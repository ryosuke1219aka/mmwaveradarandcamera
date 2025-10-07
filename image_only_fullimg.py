import os, glob, math, time, traceback, argparse
import numpy as np
from types import MethodType
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud  # 未使用でもimport整合維持
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion
from ultralytics import YOLO

# ==== BUILD MARKER / RUNTIME INFO ====
import datetime
BUILD_ID = "gt2d-v5-fullimg"
YOLO_CONF = 0.10
print(f"### BUILD {BUILD_ID} ### __file__={__file__}  now={datetime.datetime.now().isoformat(timespec='seconds')}", flush=True)

# ================== データセット設定（ROI版と同じ） ==================
NUSC_VERSION = "v1.0-trainval"
PRIMARY_DATAROOT = "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval_meta"
PART_ROOTS = [
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval_meta",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval01_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval02_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval03_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval04_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval05_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval06_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval07_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval08_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval09_blobs",
    "/Users/ryosukeakasaka/Documents/sensor/v1.0-trainval10_blobs",
]

# 全天候（ROI版と合わせて、悪天候制限はデフォ無効）
BAD_WEATHER_KEYWORDS = ["rain", "snow", "storm", "wet", "sleet", "fog", "drizzle"]
USE_BAD_WEATHER_ONLY = False

# モデル・クラスなど
YOLO_MODEL = "yolov8n.pt"
VEHICLE_CLASS_IDS = {1,2,3,5,7}  # car, bicycle(=2?) ではなく COCO準拠の車両系 (UltralyticsのID表に合わせて)
FULL_SWEEP_SHORT_SIDE = 512      # ROI版の縮小設定に合わせる

# ===== 評価用（ROI版と同じ関数群/閾値） =====
IOU_EVAL_THR = 0.50
TILE_W = 8
TILE_H = 4

DEBUG2 = True
def d2(msg: str):
    if DEBUG2:
        print(msg, flush=True)

# ---------- PATH解決（ROI版と同じロジック） ----------
def search_across_roots(relpath: str):
    for root in PART_ROOTS + [PRIMARY_DATAROOT]:
        p = os.path.join(root, relpath)
        if os.path.exists(p):
            return p
    base = os.path.basename(relpath)
    for root in PART_ROOTS + [PRIMARY_DATAROOT]:
        for p in glob.iglob(os.path.join(root, "**", base), recursive=True):
            if os.path.isfile(p):
                return p
    return None

def patch_get_sample_data_path_multi(nusc: NuScenes):
    def _get_sample_data_path_multi(self: NuScenes, token: str) -> str:
        sd = self.get('sample_data', token)
        rel = sd['filename']
        p = search_across_roots(rel)
        if p:
            return p
        raise FileNotFoundError(f"sample_data not found across roots: {rel}")
    nusc.get_sample_data_path = MethodType(_get_sample_data_path_multi, nusc)

# ---------- ユーティリティ ----------
def calculate_iou(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    areaB = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
    return inter / float(areaA + areaB - inter + 1e-6)

def _center(box):
    return ((box["x1"] + box["x2"]) * 0.5, (box["y1"] + box["y2"]) * 0.5)

def _tiles_for_image(img_wh, gw=TILE_W, gh=TILE_H):
    w, h = img_wh
    tw, th = w / gw, h / gh
    tiles = []
    for j in range(gh):
        for i in range(gw):
            x1 = int(round(i * tw))
            y1 = int(round(j * th))
            x2 = int(round((i+1) * tw))
            y2 = int(round((j+1) * th))
            tiles.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
    return tiles, tw, th

def _tile_index_for_point(x, y, w, h, gw=TILE_W, gh=TILE_H):
    ix = min(gw-1, max(0, int((x / max(1e-6,w)) * gw)))
    iy = min(gh-1, max(0, int((y / max(1e-6,h)) * gh)))
    return iy*gw + ix

def _greedy_match_iou(gts, dets, thr=IOU_EVAL_THR):
    used_det = set(); used_gt = set()
    pairs = []
    for gi, g in enumerate(gts):
        for di, d in enumerate(dets):
            iou = calculate_iou(g, d)
            if iou >= thr:
                pairs.append((iou, gi, di))
    for iou, gi, di in sorted(pairs, key=lambda x: -x[0]):
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi); used_det.add(di)
    return used_gt, used_det

def confusion_tiles(gt_boxes, det_boxes, img_wh, gw=TILE_W, gh=TILE_H, thr=IOU_EVAL_THR):
    w, h = img_wh
    tiles, tw, th = _tiles_for_image(img_wh, gw, gh)
    pos_tile = [False]*(gw*gh)
    for g in gt_boxes:
        cx = (g["x1"]+g["x2"])*0.5; cy = (g["y1"]+g["y2"])*0.5
        tidx = _tile_index_for_point(cx, cy, w, h, gw, gh)
        pos_tile[tidx] = True
    det_in_tile = [[] for _ in range(gw*gh)]
    for di, d in enumerate(det_boxes):
        cx = (d["x1"]+d["x2"])*0.5; cy = (d["y1"]+d["y2"])*0.5
        tidx = _tile_index_for_point(cx, cy, w, h, gw, gh)
        det_in_tile[tidx].append(di)
    matched_gt, matched_det = _greedy_match_iou(gt_boxes, det_boxes, thr)
    TP=TN=FP=FN=0
    for t in range(gw*gh):
        det_idxs = det_in_tile[t]
        if pos_tile[t]:
            has_tp = any(di in matched_det for di in det_idxs)
            TP += 1 if has_tp else 0
            FN += 0 if has_tp else 1
        else:
            FP += 1 if len(det_idxs)>0 else 0
            TN += 0 if len(det_idxs)>0 else 1
    return TP, TN, FP, FN

def _match_dets_to_gts_by_iou(gt_boxes, det_boxes, iou_thr=0.5):
    if not gt_boxes or not det_boxes:
        return set(), set(), []
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for di, d in enumerate(det_boxes):
            iou = calculate_iou(g, d)
            if iou >= iou_thr:
                pairs.append((iou, gi, di))
    pairs.sort(key=lambda x: -x[0])
    used_gt, used_det = set(), set()
    chosen = []
    for iou, gi, di in pairs:
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi); used_det.add(di)
        chosen.append((gi, di, iou))
    return used_gt, used_det, chosen

def box_eval_counts(gt_boxes, det_boxes, iou_thr=0.5):
    if gt_boxes is None: gt_boxes = []
    if det_boxes is None: det_boxes = []
    matched_gt, matched_det, _ = _match_dets_to_gts_by_iou(gt_boxes, det_boxes, iou_thr)
    TP = len(matched_gt)
    FP = max(0, len(det_boxes) - len(matched_det))
    FN = max(0, len(gt_boxes) - len(matched_gt))
    return TP, FP, FN

def pr_accumulate_frame(gt_boxes, det_boxes_with_conf, iou_thr, scores_list, is_tp_list, total_gt_counter):
    total_gt_counter[0] += len(gt_boxes or [])
    if not det_boxes_with_conf:
        return
    dets = sorted(det_boxes_with_conf, key=lambda d: -float(d.get("conf", 0.0)))
    matched_gt = set()
    for d in dets:
        best_iou = 0.0; best_g = -1
        for gi, g in enumerate(gt_boxes or []):
            if gi in matched_gt: 
                continue
            iou = calculate_iou(g, d)
            if iou > best_iou:
                best_iou = iou; best_g = gi
        is_tp = (best_g >= 0 and best_iou >= iou_thr)
        if is_tp:
            matched_gt.add(best_g)
        scores_list.append(float(d.get("conf", 0.0)))
        is_tp_list.append(1 if is_tp else 0)

def _compute_ap(scores, is_tp, total_gt, pr_points=101):
    if total_gt <= 0 or len(scores) == 0:
        return [], [], 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(is_tp)[order].astype(np.int32)
    fp = 1 - tp
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(1, (cum_tp + cum_fp))
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    recall_points = np.linspace(0, 1, pr_points)
    prec_at_rec = []
    for r in recall_points:
        inds = np.where(mrec >= r)[0]
        p = 0.0 if len(inds) == 0 else np.max(mpre[inds])
        prec_at_rec.append(p)
    ap = float(np.mean(prec_at_rec))
    return recalls.tolist(), precisions.tolist(), ap

# === CLI ===
def _parse_args():
    p = argparse.ArgumentParser(description="Image-only full-frame vehicle detection (NuScenes)")
    p.add_argument("--yolo-model", default=os.environ.get("YOLO_MODEL", YOLO_MODEL),
                   help="Ultralytics YOLO weights (e.g., yolov8n.pt, yolov8s.pt, custom.pt).")
    p.add_argument("--conf", type=float, default=YOLO_CONF,
                   help="YOLO confidence threshold.")
    p.add_argument("--device", default=None,
                   help="Torch device (e.g., 'cuda:0', 'mps', 'cpu'). If omitted, Ultralytics default is used.")
    p.add_argument("--eval-iou", type=float, default=0.50,
                   help="IoU threshold for box-level evaluation (TP/FP/FN).")
    p.add_argument("--eval-map", action="store_true",
                   help="Compute dataset-level PR curve and AP/mAP using detection confidences.")
    p.add_argument("--pr-curve-points", type=int, default=101,
                   help="Number of recall points for PR/AUC (101 for COCO-style).")
    return p.parse_args()

# === 天候タグ（参考集計） ===
def _tag_weather(desc_raw: str) -> str:
    if not desc_raw:
        return "clear"
    d = desc_raw.lower()
    if any(k in d for k in ["rain", "drizzle", "wet"]): return "rain"
    if any(k in d for k in ["snow", "sleet"]): return "snow"
    if any(k in d for k in ["fog", "mist", "haze"]): return "fog"
    if any(k in d for k in ["night", "dark"]): return "night"
    if any(k in d for k in ["cloud", "overcast"]): return "cloudy"
    return "clear"

# ---------- GT投影（ROI版と同等の2D化） ----------
def get_gt_2d_box(nusc: NuScenes, ann_token: str, cam_token: str, img_wh):
    try:
        _, boxes, K = nusc.get_sample_data(cam_token)
        box = None
        for b in boxes:
            if getattr(b, 'token', None) == ann_token:
                box = b; break
        if box is None:
            return None
        corners_3d = box.corners()
        depths = corners_3d[2, :]
        valid = depths > 1e-3
        if not np.any(valid):
            return None
        corners_2d = view_points(corners_3d[:, valid], np.array(K), normalize=True)
        w, h = img_wh
        xs, ys = corners_2d[0, :], corners_2d[1, :]
        x1 = int(np.clip(np.min(xs), 0, w - 1)); y1 = int(np.clip(np.min(ys), 0, h - 1))
        x2 = int(np.clip(np.max(xs), 0, w - 1)); y2 = int(np.clip(np.max(ys), 0, h - 1))
        if (x2 - x1) < 5 or (y2 - y1) < 5: return None
        if x2 <= x1 or y2 <= y1: return None
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except Exception:
        return None

# ---------- YOLO: フル画像推論（短辺512へリサイズ→逆変換） ----------
def yolo_vehicle_detections_full(model: YOLO, img_pil, conf, vehicle_ids):
    if not hasattr(yolo_vehicle_detections_full, "_banner_printed"):
        print(f"[dbg] YOLO_CONF={conf} (full image)", flush=True)
        yolo_vehicle_detections_full._banner_printed = True

    w, h = img_pil.size
    if min(w, h) > FULL_SWEEP_SHORT_SIDE:
        if w < h:
            new_w = FULL_SWEEP_SHORT_SIDE
            new_h = int(h * (new_w / w))
        else:
            new_h = FULL_SWEEP_SHORT_SIDE
            new_w = int(w * (new_h / h))
        img_small = img_pil.resize((new_w, new_h), Image.BILINEAR)
        res = model(img_small, verbose=False, conf=conf)[0]
        sx, sy = (w / new_w), (h / new_h)
        outs = []
        boxes = res.boxes.xyxy.cpu().numpy(); clss = res.boxes.cls.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            if int(clss[i]) in vehicle_ids:
                x1,y1,x2,y2 = boxes[i]
                outs.append({
                    "x1": int(x1*sx), "y1": int(y1*sy),
                    "x2": int(x2*sx), "y2": int(y2*sy),
                    "conf": float(confs[i])
                })
        return outs
    else:
        res = model(img_pil, verbose=False, conf=conf)[0]
        outs = []
        boxes = res.boxes.xyxy.cpu().numpy(); clss = res.boxes.cls.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            if int(clss[i]) in vehicle_ids:
                x1,y1,x2,y2 = boxes[i].astype(int)
                outs.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf": float(confs[i])
                })
        return outs

# ---------- シーン選別（ROI版と同じ） ----------
def filter_scenes(nusc: NuScenes):
    scenes = nusc.scene
    selected = []
    for s in scenes:
        if USE_BAD_WEATHER_ONLY:
            desc = (s.get("description") or "").lower()
            if not any(k in desc for k in BAD_WEATHER_KEYWORDS):
                continue
        sample = nusc.get("sample", s["first_sample_token"])
        if "CAM_FRONT" not in sample["data"]:
            continue
        try:
            _ = nusc.get_sample_data_path(sample["data"]["CAM_FRONT"])
            selected.append(s)
        except Exception:
            continue
    return selected

# ================== メイン ==================
def main():
    # CLI
    args = _parse_args()
    global YOLO_MODEL, YOLO_CONF
    YOLO_MODEL = args.yolo_model
    YOLO_CONF = args.conf

    print("[1/7] Load NuScenes...")
    nusc = NuScenes(version=NUSC_VERSION, dataroot=PRIMARY_DATAROOT, verbose=True)
    print("[2/7] Patch path resolver...")
    patch_get_sample_data_path_multi(nusc)

    print("[3/7] Pre-screen scenes...")
    scenes = filter_scenes(nusc)
    print(f"  -> candidate scenes: {len(scenes)}")
    if not scenes:
        print("No scenes with CAM_FRONT resolvable. Check PART_ROOTS.")
        return

    print("[4/7] Load YOLO...")
    model = YOLO(YOLO_MODEL)
    if args.device:
        try:
            model.to(args.device)
        except Exception as _e:
            print(f"[warn] Could not move model to device '{args.device}': {_e}. Using default device.", flush=True)

    # 結果集計（ROI版に揃える）
    total_ms = 0.0
    total_px = 0              # 処理ピクセル総量（常に w*h を加算）
    weather_scene_counts = {}

    # Box-level cumulative
    sum_box_tp = 0; sum_box_fp = 0; sum_box_fn = 0

    # Tile-based cumulative
    sum_tp = 0; sum_tn = 0; sum_fp = 0; sum_fn = 0

    # PR/AP
    pr_scores = []; pr_is_tp = []; pr_total_gt = [0]
    iou_eval_thr = float(getattr(args, "eval_iou", 0.50))

    # Radar/Cam先行は “画像のみ” なので参考表示用ダミー
    total_pairs = 0; radar_first = 0; cam_first = 0; simultaneous = 0

    print("[5/7] Iterate samples & measure timing...")
    dev_str = getattr(args, "device", None) or "auto"
    print(f"  CONFIG: YOLO_MODEL={YOLO_MODEL} YOLO_CONF={YOLO_CONF} DEVICE={dev_str} FULL_SWEEP_SHORT_SIDE={FULL_SWEEP_SHORT_SIDE} [BUILD {BUILD_ID}]",
          flush=True)

    for si, scene in enumerate(scenes, 1):
        wtag = _tag_weather(scene.get("description") or "")
        weather_scene_counts[wtag] = weather_scene_counts.get(wtag, 0) + 1

        token = scene["first_sample_token"]
        start_scene = time.time()
        sample_idx_in_scene = 0

        while token:
            sample = nusc.get("sample", token)
            cam_t = sample["data"].get("CAM_FRONT", None)
            if cam_t is None:
                token = sample["next"]; continue

            try:
                cam_path = nusc.get_sample_data_path(cam_t)
            except Exception:
                token = sample["next"]; continue

            try:
                img = Image.open(cam_path).convert("RGB")
                w, h = img.size
            except Exception:
                token = sample["next"]; continue

            # === GT2D（評価用） ===
            gt2d_list = []
            for ann_t in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_t)
                if 'vehicle' not in ann['category_name']:
                    continue
                gt2d = get_gt_2d_box(nusc, ann_t, cam_t, (w, h))
                if gt2d is not None:
                    gt2d_list.append(gt2d)

            # === 推論（フル画像） ===
            t0 = time.perf_counter()
            dets = yolo_vehicle_detections_full(model, img, YOLO_CONF, VEHICLE_CLASS_IDS)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            total_ms += dt_ms
            total_px += (w * h)

            # === Box-level ===
            det_boxes_plain = [{"x1":b["x1"],"y1":b["y1"],"x2":b["x2"],"y2":b["y2"]} for b in dets]
            tp_b, fp_b, fn_b = box_eval_counts(gt2d_list, det_boxes_plain, iou_thr=iou_eval_thr)
            sum_box_tp += tp_b; sum_box_fp += fp_b; sum_box_fn += fn_b

            # === PR/AP ===
            if args.eval_map:
                pr_accumulate_frame(gt2d_list, dets, iou_eval_thr, pr_scores, pr_is_tp, pr_total_gt)

            # === Tile-based ===
            tp_t, tn_t, fp_t, fn_t = confusion_tiles(gt2d_list, det_boxes_plain, (w, h))
            sum_tp += tp_t; sum_tn += tn_t; sum_fp += fp_t; sum_fn += fn_t

            if sample_idx_in_scene < 3:
                print(f"  [dbg] FULL  time={int(dt_ms)}ms  yolo={len(dets)}  anns_total={len(sample['anns'])} veh_gt2d={len(gt2d_list)}")
            sample_idx_in_scene += 1
            token = sample["next"]

        elapsed = math.ceil(time.time() - start_scene)
        print(f"[scene {si}/{len(scenes)}] {scene['name']}: time={elapsed}s (image-only)")

    # ====== サマリ出力（ROI版と同じ見出し） ======
    frames_count = max(1, len(pr_scores)) if args.eval_map else 1
    print("\n[6/7] Timing & Load Summary")
    print(f"  avg_inference_time = {total_ms / max(1,frames_count):.1f} ms/frame")
    print(f"  processed_pixel_equiv = {total_px/1e6:.2f} MPix (sum, rough compute proxy)")

    print("\n[7/7] Detection Summary")
    print(f" total_pairs={total_pairs}")
    print("  (image-onlyモードのため radar/camera 先行同時計測は未実施; ROI版結果と並列表で示すことを推奨)")

    # ===== [8/8] Tile-based Confusion =====
    total_tiles = sum_tp + sum_tn + sum_fp + sum_fn
    if total_tiles > 0:
        acc = (sum_tp + sum_tn) / total_tiles
        recall = sum_tp / max(1, (sum_tp + sum_fn))
        specificity = sum_tn / max(1, (sum_tn + sum_fp))
        precision = sum_tp / max(1, (sum_tp + sum_fp))
        f1 = 2 * precision * recall / max(1e-12, (precision + recall))
        print("\n[8/8] Tile-based Confusion (IoU>=%.2f, grid=%dx%d)" % (IOU_EVAL_THR, TILE_W, TILE_H))
        print(f"  TP={sum_tp}  TN={sum_tn}  FP={sum_fp}  FN={sum_fn}  (tiles total={total_tiles})")
        print(f"  Acc={acc:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  Specificity={specificity:.3f}  F1={f1:.3f}")
    else:
        print("\n[8/8] Tile-based Confusion: no tiles counted.")

    # ===== [9/9] Box-level (IoU-based) =====
    total_pos = sum_box_tp + sum_box_fn
    total_pred = sum_box_tp + sum_box_fp
    if (total_pos + total_pred) > 0:
        box_precision = sum_box_tp / max(1, (sum_box_tp + sum_box_fp))
        box_recall = sum_box_tp / max(1, (sum_box_tp + sum_box_fn))
        box_f1 = 2 * box_precision * box_recall / max(1e-12, (box_precision + box_recall))
        print("\n[9/9] Box-level (IoU>=%.2f)" % (iou_eval_thr,))
        print(f"  TP={sum_box_tp}  FP={sum_box_fp}  FN={sum_box_fn}")
        print(f"  Precision={box_precision:.3f}  Recall={box_recall:.3f}  F1={box_f1:.3f}")
    else:
        print("\n[9/9] Box-level: no boxes counted.")

    # ===== [10/10] Dataset PR/AP =====
    if args.eval_map:
        recalls, precisions, ap = _compute_ap(pr_scores, pr_is_tp, pr_total_gt[0], pr_points=getattr(args, "pr_curve_points", 101))
        print("\n[10/10] PR/AP (IoU>=%.2f)" % (iou_eval_thr,))
        print(f"  total_gt={pr_total_gt[0]}  dets={len(pr_scores)}  AP={ap:.3f}")
        if len(recalls) > 0 and len(precisions) > 0:
            for i in np.linspace(0, len(recalls)-1, num=5, dtype=int):
                print(f"   r={recalls[i]:.2f}  p={precisions[i]:.2f}")
        else:
            print("  (no detections or no GT; cannot compute PR)")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()