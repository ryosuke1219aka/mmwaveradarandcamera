import os, glob, math, time, traceback
import numpy as np
from types import MethodType
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
from ultralytics import YOLO

# 追加: ユーティリティ（先頭のimport付近でOK）
from nuscenes.utils.geometry_utils import transform_matrix, view_points

import argparse, csv, os

# ==== BUILD MARKER / RUNTIME INFO ====
import datetime, inspect, sys
BUILD_ID = "gt2d-v5-roi-bev"
YOLO_CONF = 0.10
print(f"### BUILD {BUILD_ID} ### __file__={__file__}  now={datetime.datetime.now().isoformat(timespec='seconds')}", flush=True)

# ================== 設定 ==================
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

# === 全天候設定（悪天候フィルタは無効化） ===
BAD_WEATHER_KEYWORDS = ["rain", "snow", "storm", "wet", "sleet", "fog", "drizzle"]
USE_BAD_WEATHER_ONLY = False   # 全天候で処理

# マッチング＆検出条件
IOU_THRESH = 0.10
RADAR_MIN_PTS = 1
NSWEEPS = 5
YOLO_MODEL = "yolov8n.pt"  # default model (can be overridden by --yolo-model)
VEHICLE_CLASS_IDS = {1,2,3,5,7}
MAX_SCENES = None

# ===== ROI 関連設定 =====
USE_ROI = True
# 少し細かいグリッドにし、最小点数もしきいを下げて「ROIを出しやすく」する
ROI_GRID = 40
ROI_MIN_PTS = 1
ROI_PAD_RATIO = 0.25
ROI_SIZE_NEAR = 220
ROI_SIZE_FAR = 80
ROI_NEAR_M = 20.0
ROI_FAR_M  = 70.0

# ROI数の上限もやや緩める
MAX_NUM_ROI = 20

# --- ROI総量 “予算”（フレームの画素の何割までROIで使うか）---
# 例: 0.15 なら画像総画素の15%を上限にROIを採用
ROI_AREA_BUDGET_RATIO = 0.15   # 0.0～1.0 の範囲で調整
# 予算超過時の優先度: 近距離(小depth)優先 → 面積の小さいROI優先
# ※より高度なスコアリング（点密度/速度/rcsなど）は後段で拡張可

# === フルスイープの方針を選べるようにする ===
#   'none'     : 一切フルスイープしない（ROI/フォールバックのみ）
#   'periodic' : FULL_SWEEP_EVERY ごとに実施（従来）
#   'adaptive' : ROIが出ないフレームが続いた時にだけ実施（推奨）
FULL_SWEEP_POLICY = "adaptive"
FULL_SWEEP_EVERY = 200           # 'periodic' 用（頻度をさらに下げる）
FULL_SWEEP_SHORT_SIDE = 512

# adaptive policy 用パラメータ
ADAPTIVE_FULL_MISS_THRESH = 5    # 連続してROIがゼロのフレーム数で実施
ADAPTIVE_FULL_MIN_GAP = 50       # 最低でもこれだけフレームはフルを空ける

# Batch size for batched ROI inference
ROI_BATCH_SIZE = 16

# --- ROI fallback settings (when radar produces no ROI) ---
FALLBACK_GRID_W = 2   # number of tiles horizontally (reduced)
FALLBACK_GRID_H = 1   # number of tiles vertically   (reduced)
FALLBACK_OVERLAP = 0.0  # 0% overlap to minimize duplicate coverage
FALLBACK_MIN_SIDE = 256  # shrink tiles if image very large (optional downscale happens elsewhere)

# ---- scene-level sweep state (for adaptive full-sweep policy) ----
class SceneSweepState:
    def __init__(self):
        self.frames_since_full = 0
        self.consecutive_roi_zero = 0

# ================== DEBUG2 トグル/ヘルパ ==================
DEBUG2 = True  # Falseにすれば全部黙ります

def d2(msg: str):
    if DEBUG2:
        print(msg, flush=True)

# ================== ユーティリティ ==================

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
    return inter / float(areaA + areaB - inter)


def _center(box):
    return ((box["x1"] + box["x2"]) * 0.5, (box["y1"] + box["y2"]) * 0.5)


def _contains(box, x, y):
    return (box["x1"] <= x <= box["x2"]) and (box["y1"] <= y <= box["y2"])

# ===== Confusion (tile-based) evaluation =====
TILE_W = 8   # 横タイル数（必要に応じて調整）
TILE_H = 4   # 縦タイル数
IOU_EVAL_THR = 0.50  # IoU閾値（評価用）

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

def _center_of(box):
    # 既存の _center を使う薄いラッパ（将来の互換性のため）
    return _center(box)

def _tile_index_for_point(x, y, w, h, gw=TILE_W, gh=TILE_H):
    ix = min(gw-1, max(0, int((x / max(1e-6,w)) * gw)))
    iy = min(gh-1, max(0, int((y / max(1e-6,h)) * gh)))
    return iy*gw + ix

def _greedy_match_iou(gts, dets, thr=IOU_EVAL_THR):
    """
    GTと検出の1対1マッチング（IoU降順で貪欲選択）。
    戻り値: (matched_gt_idx_set, matched_det_idx_set)
    """
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
    """
    タイル単位の TP/TN/FP/FN を算出。
      - 正タイル: GT中心が1つ以上
      - 負タイル: 正タイル以外
      - 検出は中心点の属するタイルで評価
      - 正タイルで IoU>=thr の“マッチ済み検出”が1つ以上→TPタイル、なければFNタイル
      - 負タイルで検出>=1→FPタイル、0→TNタイル
    """
    w, h = img_wh
    tiles, tw, th = _tiles_for_image(img_wh, gw, gh)

    # 正/負タイルラベリング（GT中心）
    pos_tile = [False] * (gw*gh)
    for g in gt_boxes:
        cx, cy = _center_of(g)
        tidx = _tile_index_for_point(cx, cy, w, h, gw, gh)
        pos_tile[tidx] = True

    # 各タイルに属する検出のインデックス
    det_in_tile = [[] for _ in range(gw*gh)]
    for di, d in enumerate(det_boxes):
        cx, cy = _center_of(d)
        tidx = _tile_index_for_point(cx, cy, w, h, gw, gh)
        det_in_tile[tidx].append(di)

    # IoUマッチ（画像全体で1対1）
    matched_gt, matched_det = _greedy_match_iou(gt_boxes, det_boxes, thr)

    TP=TN=FP=FN=0
    for t in range(gw*gh):
        det_idxs = det_in_tile[t]
        if pos_tile[t]:
            has_tp = any(di in matched_det for di in det_idxs)
            if has_tp:
                TP += 1
            else:
                FN += 1
        else:
            if len(det_idxs) > 0:
                FP += 1
            else:
                TN += 1
    return TP, TN, FP, FN

# ===== Box-level evaluation (IoU matching) + PR/AP =====

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

# === CLI arguments ===
def _parse_args():
    p = argparse.ArgumentParser(description="Radar-guided ROI vehicle detection (NuScenes)")
    p.add_argument("--yolo-model", default=os.environ.get("YOLO_MODEL", YOLO_MODEL),
                   help="Ultralytics YOLO weights (e.g., yolov8n.pt, yolov8s.pt, custom.pt). Env YOLO_MODEL also respected.")
    p.add_argument("--conf", type=float, default=YOLO_CONF,
                   help="YOLO confidence threshold (default from YOLO_CONF).")
    p.add_argument("--device", default=None,
                   help="Torch device for YOLO (e.g., 'cuda:0', 'mps', 'cpu'). If omitted, Ultralytics default is used.")
    p.add_argument("--roi-budget", type=float, default=float(os.environ.get("ROI_BUDGET", ROI_AREA_BUDGET_RATIO)),
                   help="ROI area budget ratio (0.0-1.0). Can also set via env ROI_BUDGET.")
    p.add_argument("--eval-iou", type=float, default=0.50,
                    help="IoU threshold for box-level evaluation (TP/FP/FN). Default=0.50.")
    p.add_argument("--eval-map", action="store_true",
                    help="Compute dataset-level PR curve and AP/mAP using detection confidences.")
    p.add_argument("--pr-curve-points", type=int, default=101,
                    help="Number of recall points for PR/AUC (101 for COCO-style).")
    return p.parse_args()


# === 天候タグ（ざっくり分類：可視化用） ===

def _tag_weather(desc_raw: str) -> str:
    if not desc_raw:
        return "clear"
    d = desc_raw.lower()
    if any(k in d for k in ["rain", "drizzle", "wet"]):
        return "rain"
    if any(k in d for k in ["snow", "sleet"]):
        return "snow"
    if any(k in d for k in ["fog", "mist", "haze"]):
        return "fog"
    if any(k in d for k in ["night", "dark"]):
        return "night"
    if any(k in d for k in ["cloud", "overcast"]):
        return "cloudy"
    return "clear"


# ---------- レーダ→global ----------

# --- custom radar file resolver and loader ---
def _resolve_sd_path(nusc: NuScenes, sd_token: str) -> str:
    """
    sample_data の token から実ファイルの絶対パスを探す。
    NuScenes 標準の dataroot 連結ではなく、PART_ROOTS を横断して解決する。
    """
    sd = nusc.get('sample_data', sd_token)
    rel = sd['filename']
    p = search_across_roots(rel)
    if p is None:
        raise FileNotFoundError(f"Cannot resolve sample_data file across roots: {rel}")
    d2(f"[dbg2/patch] resolve radar file -> {p}")
    return p


def _radar_multisweep_points_global(nusc: NuScenes, sample: dict, channel: str, nsweeps: int):
    """
    RADAR の過去 nsweeps を遡り、各スイープを global 座標に変換して連結して返す。
    返り値:
      pts_global: (3, N)
      attrs: dict(vx, vy, rcs) 各 shape=(N,)
    """
    if channel not in sample['data']:
        return None, {}

    cur_token = sample['data'][channel]
    pts_list = []
    vx_list, vy_list, rcs_list = [], [], []

    count = 0
    while cur_token and count < nsweeps:
        try:
            sd = nusc.get('sample_data', cur_token)
            fpath = _resolve_sd_path(nusc, cur_token)

            # 読み込み
            pc = RadarPointCloud.from_file(fpath)

            # sensor -> ego (その時刻)
            cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
            pc.translate(np.array(cs['translation']))

            # ego -> global (その時刻)
            pose = nusc.get('ego_pose', sd['ego_pose_token'])
            pc.rotate(Quaternion(pose['rotation']).rotation_matrix)
            pc.translate(np.array(pose['translation']))

            # 連結（必要な次元だけ保持）
            pts_list.append(pc.points[:3, :])
            rcs_list.append(pc.points[6, :])
            vx_list.append(pc.points[8, :])
            vy_list.append(pc.points[9, :])

            count += 1
            cur_token = sd['prev']  # 過去方向へ
        except Exception as e:
            d2(f"[dbg2/c0] sweep load error: {e}")
            break

    if count == 0 or len(pts_list) == 0:
        return None, {}

    pts_global = np.concatenate(pts_list, axis=1)
    attrs = {
        "vx": np.concatenate(vx_list, axis=0),
        "vy": np.concatenate(vy_list, axis=0),
        "rcs": np.concatenate(rcs_list, axis=0),
    }
    d2(f"[dbg2/c1] (custom multi) sweeps={count} pts_raw={pts_global.shape[1]}")
    return pts_global, attrs


def radar_points_global(nusc: NuScenes, sample, nsweeps=NSWEEPS):
    """
    nsweepsぶんの RADAR_FRONT 点群を取得し、ref=RADAR_FRONT のキーサンプル基準で
    レーダ座標系→グローバル座標系へ変換して返す。
    """
    radar_token = sample['data'].get('RADAR_FRONT', None)
    if radar_token is None:
        d2("[dbg2/c0] RADAR_FRONT token missing")
        return None, {}

    try:
        pts_global, attrs = _radar_multisweep_points_global(nusc, sample, 'RADAR_FRONT', nsweeps)
        if pts_global is None:
            return None, {}
    except Exception as e:
        d2(f"[dbg2/c0] from_file_multisweep(custom) error: {e}")
        return None, {}

    # すでに global で連結済み
    d2(f"[dbg2/c1] radar_sweeps={nsweeps} pts_raw={pts_global.shape[1]}")
    return pts_global, attrs


# ---------- global→camera→image 投影 ----------

def project_points_to_cam(nusc: NuScenes, sample, cam_token, pts_global):
    """
    global(3xN) を 指定カメラの画像平面に投影して (u,v,depth) を返す。
    """
    if pts_global is None or pts_global.shape[1] == 0:
        d2("[dbg2/d0] pts_global empty")
        return None, None, None

    sd_cam = nusc.get('sample_data', cam_token)
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
    pose_cam = nusc.get('ego_pose', sd_cam['ego_pose_token'])

    T_ego_from_global = transform_matrix(pose_cam['translation'], Quaternion(pose_cam['rotation'])).T
    T_cam_from_ego = transform_matrix(cs_cam['translation'], Quaternion(cs_cam['rotation'])).T
    T_cam_from_global = T_cam_from_ego @ T_ego_from_global

    pts_h = np.vstack([pts_global, np.ones((1, pts_global.shape[1]))])
    pts_cam_h = T_cam_from_global @ pts_h
    pts_cam = pts_cam_h[:3, :]

    K = np.array(cs_cam['camera_intrinsic'])
    uv = view_points(pts_cam, K, normalize=True)  # 3xN
    u, v, d = uv[0, :], uv[1, :], pts_cam[2, :]
    d2(f"[dbg2/d1] proj pts_cam={pts_cam.shape[1]} depth>0={int(np.sum(d>0))}")
    return u, v, d


# ---------- ROI生成（レーダ投影→グリッド集約） ----------

def build_rois_from_radar(nusc: NuScenes, sample, cam_token, img_wh):
    """レーダー投影点を画素グリッドに集約してROI矩形の配列を作る。"""
    w, h = img_wh
    pts_g, attrs = radar_points_global(nusc, sample, nsweeps=NSWEEPS)
    if pts_g is None:
        d2("[dbg2/b0] radar_points_global -> None")
        return []
    u, v, d = project_points_to_cam(nusc, sample, cam_token, pts_g)
    if u is None:
        d2("[dbg2/b0] project_points_to_cam -> None")
        return []

    # 画像内＆手前の点だけ
    valid = (d > 1.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    n_in = int(np.sum(valid))
    n_raw = u.shape[0]
    if n_in == 0:
        d2(f"[dbg2/b1] proj_in_ratio=0.00 (in=0, raw={n_raw})")
        return []
    u, v, d = u[valid], v[valid], d[valid]
    d2(f"[dbg2/b1] proj_in_ratio={n_in/max(1,n_raw):.2f} (in={n_in}, raw={n_raw})")

    # 画素グリッドで集約
    gx = (u // ROI_GRID).astype(np.int32)
    gy = (v // ROI_GRID).astype(np.int32)
    key = gx + 10000 * gy
    uniq, inv = np.unique(key, return_inverse=True)
    d2(f"[dbg2/b2] grid_cells={len(uniq)}")

    rois_raw = 0
    rois = []
    for cell_id in range(len(uniq)):
        idxs = np.where(inv == cell_id)[0]
        if idxs.size < ROI_MIN_PTS:
            continue
        rois_raw += 1
        u_cell = u[idxs]; v_cell = v[idxs]; d_cell = d[idxs]
        d_med = float(np.median(d_cell))
        if d_med <= ROI_NEAR_M:
            base = ROI_SIZE_NEAR
        elif d_med >= ROI_FAR_M:
            base = ROI_SIZE_FAR
        else:
            t = (d_med - ROI_NEAR_M) / (ROI_FAR_M - ROI_NEAR_M)
            base = (1 - t) * ROI_SIZE_NEAR + t * ROI_SIZE_FAR

        spread = (np.std(u_cell) + np.std(v_cell)) * 0.5
        size = int(max(32, base + spread * 2.0))
        cx = int(np.clip(np.mean(u_cell), 0, w - 1))
        cy = int(np.clip(np.mean(v_cell), 0, h - 1))
        x1 = int(max(0, cx - size//2)); y1 = int(max(0, cy - size//2))
        x2 = int(min(w-1, cx + size//2)); y2 = int(min(h-1, cy + size//2))

        padw = int((x2 - x1) * ROI_PAD_RATIO); padh = int((y2 - y1) * ROI_PAD_RATIO)
        x1 = max(0, x1 - padw); y1 = max(0, y1 - padh)
        x2 = min(w-1, x2 + padw); y2 = min(h-1, y2 + padh)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        rois.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "depth": d_med})

    if not rois:
        d2(f"[dbg2/b3] ROIs raw={rois_raw} -> kept=0 (minsize/minpts/filters)")
        return []

    # 近距離優先でソート→重複マージ
    rois.sort(key=lambda r: r["depth"])
    merged = []
    for r in rois:
        keep = True
        for m in merged:
            inter_x1 = max(r["x1"], m["x1"]); inter_y1 = max(r["y1"], m["y1"])
            inter_x2 = min(r["x2"], m["x2"]); inter_y2 = min(r["y2"], m["y2"])
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_r = (r["x2"]-r["x1"]) * (r["y2"]-r["y1"]) 
            area_m = (m["x2"]-m["x1"]) * (m["y2"]-m["y1"]) 
            iou = inter / float(area_r + area_m - inter + 1e-6)
            if iou > 0.3:
                m["x1"] = min(m["x1"], r["x1"]); m["y1"] = min(m["y1"], r["y1"]) 
                m["x2"] = max(m["x2"], r["x2"]); m["y2"] = max(m["y2"], r["y2"]) 
                keep = False
                break
        if keep:
            merged.append(r)
        if len(merged) >= MAX_NUM_ROI:
            break

    # --- ROI面積“予算”の適用 ---
    # 画像総画素に対して ROI の合計面積が上限(ROI_AREA_BUDGET_RATIO)を超えないように抑制
    if ROI_AREA_BUDGET_RATIO is not None and ROI_AREA_BUDGET_RATIO > 0.0:
        budget_px = int(ROI_AREA_BUDGET_RATIO * w * h)
        # 近距離(小depth)優先、次に面積の小さい順で並べ替え
        merged.sort(key=lambda r: (r["depth"], (r["x2"]-r["x1"]) * (r["y2"]-r["y1"])))
        kept = []
        area_sum = 0
        for r in merged:
            a = max(0, r["x2"]-r["x1"]) * max(0, r["y2"]-r["y1"])
            # 少なくとも1枚は必ず採用（予算が極端に小さい設定への対策）
            if (area_sum + a) <= budget_px or len(kept) == 0:
                kept.append(r)
                area_sum += a
            # 予算超過したら打ち切り
            if area_sum >= budget_px:
                break
        # 上限枚数も適用（念のため）
        if len(kept) > MAX_NUM_ROI:
            kept = kept[:MAX_NUM_ROI]
        # デバッグ出力
        d2(f"[dbg2/b4-budget] ROI budget applied: kept={len(kept)} area%={int(area_sum*100/max(1,w*h))}% (budget={int(ROI_AREA_BUDGET_RATIO*100)}%)")
        merged = kept

    # ROIが極端に少ない場合は、粗いグリッドで少し補う（フル回避のため）
    if len(merged) < 2:
        extra = build_coarse_fallback_rois(img_wh)
        # 近距離優先で先頭数個だけ追加
        for e in extra[:max(0, 4 - len(merged))]:
            merged.append(e)
    d2(f"[dbg2/b4] ROIs raw={rois_raw} merged={len(merged)} (MAX={MAX_NUM_ROI})")
    return merged


# ---------- ROI fallback tile builder ----------
def build_coarse_fallback_rois(img_wh, grid_scale: float = 1.0):
    """When radar yields no ROI, cover the image with a coarse grid of tiles.
    Returns a list of boxes {x1,y1,x2,y2,depth} (depth is dummy for sorting compatibility).
    """
    w, h = img_wh
    gw = max(1, int(round(FALLBACK_GRID_W * grid_scale)))
    gh = max(1, int(round(FALLBACK_GRID_H * grid_scale)))
    # base tile size
    tw = w / gw
    th = h / gh
    # overlap in pixels
    ox = tw * FALLBACK_OVERLAP
    oy = th * FALLBACK_OVERLAP
    rois = []
    for gy in range(gh):
        for gx in range(gw):
            x1 = int(max(0, gx * tw - ox/2))
            y1 = int(max(0, gy * th - oy/2))
            x2 = int(min(w, (gx+1) * tw + ox/2))
            y2 = int(min(h, (gy+1) * th + oy/2))
            # guard against tiny tiles
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            rois.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "depth": 9999.0})
    d2(f"[dbg2/roi-fallback] coarse tiles n={len(rois)} grid={gw}x{gh} overlap={int(FALLBACK_OVERLAP*100)}%")
    return rois


# --- Helper: Apply ROI area budget to fallback tiles ---
def _apply_roi_budget(rois, w, h, budget_ratio, max_num=MAX_NUM_ROI):
    """
    Apply an area budget to a list of ROI boxes so that the total covered pixel area
    does not exceed budget_ratio * (w*h). Selection priority: smaller depth first
    (closer objects), then smaller area. Ensures at least one ROI is kept if list non-empty.
    """
    if not rois or budget_ratio is None or budget_ratio <= 0.0:
        return rois
    budget_px = int(budget_ratio * w * h)
    # sort by (depth asc, area asc)
    rois_sorted = sorted(rois, key=lambda r: (r.get("depth", 9999.0),
                                              (r["x2"]-r["x1"]) * (r["y2"]-r["y1"])))
    kept, area_sum = [], 0
    for r in rois_sorted:
        a = max(0, r["x2"]-r["x1"]) * max(0, r["y2"]-r["y1"])
        if (area_sum + a) <= budget_px or len(kept) == 0:
            kept.append(r)
            area_sum += a
        if area_sum >= budget_px or len(kept) >= max_num:
            break
    return kept


# ---------- 既存GT投影 ----------

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
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except Exception:
        return None


# ---------- YOLO呼び出し（ROI対応） ----------

def yolo_vehicle_detections_full(model: YOLO, img_pil):
    if not hasattr(yolo_vehicle_detections_full, "_banner_printed"):
        print(f"[dbg] YOLO_CONF={YOLO_CONF} (full image)", flush=True)
        yolo_vehicle_detections_full._banner_printed = True
    res = model(img_pil, verbose=False, conf=YOLO_CONF)[0]
    boxes = res.boxes.xyxy.cpu().numpy(); clss = res.boxes.cls.cpu().numpy()
    outs = []
    for i in range(len(boxes)):
        if int(clss[i]) in VEHICLE_CLASS_IDS:
            x1,y1,x2,y2 = boxes[i].astype(int)
            conf = float(res.boxes.conf.cpu().numpy()[i])
            outs.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "conf": conf
            })
    return outs


def yolo_vehicle_detections_roi(model: YOLO, img_pil, rois):
    """複数ROIを順にクロップして推論→座標を元画像系へ戻して結合"""
    all_out = []
    for r in rois:
        crop = img_pil.crop((r["x1"], r["y1"], r["x2"], r["y2"]))
        res = model(crop, verbose=False, conf=YOLO_CONF)[0]
        boxes = res.boxes.xyxy.cpu().numpy(); clss = res.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            if int(clss[i]) in VEHICLE_CLASS_IDS:
                x1,y1,x2,y2 = boxes[i].astype(int)
                conf = float(res.boxes.conf.cpu().numpy()[i])
                all_out.append({
                    "x1": int(x1 + r["x1"]), "y1": int(y1 + r["y1"]),
                    "x2": int(x2 + r["x1"]), "y2": int(y2 + r["y1"]),
                    "conf": conf
                })
    return all_out


# Batched ROI inference: crops multiple ROIs and runs them through YOLO in mini-batches.
def yolo_vehicle_detections_roi_batched(model: YOLO, img_pil, rois, batch_size: int = ROI_BATCH_SIZE):
    """
    Batched ROI inference: crops multiple ROIs and runs them through YOLO in mini-batches.
    Args:
        model: Ultralytics YOLO model.
        img_pil: PIL.Image of the full frame.
        rois: list of dicts with keys x1,y1,x2,y2.
        batch_size: number of crops per model forward.
    Returns:
        list of boxes in full-image coordinates (dicts with x1,y1,x2,y2).
    """
    if not rois:
        return []

    outs = []

    try:
        for start in range(0, len(rois), max(1, batch_size)):
            end = min(len(rois), start + max(1, batch_size))
            batch_rois = rois[start:end]
            batch_crops = [img_pil.crop((r["x1"], r["y1"], r["x2"], r["y2"])) for r in batch_rois]

            # Run a single forward pass for this mini-batch
            results = model(batch_crops, verbose=False, conf=YOLO_CONF)

            # Map detections back to full-image coordinates
            for j, res in enumerate(results):
                r = batch_rois[j]
                x_off, y_off = r["x1"], r["y1"]
                boxes = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                for i in range(len(boxes)):
                    if int(clss[i]) in VEHICLE_CLASS_IDS:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        conf = float(res.boxes.conf.cpu().numpy()[i])
                        outs.append({
                            "x1": int(x1 + x_off), "y1": int(y1 + y_off),
                            "x2": int(x2 + x_off), "y2": int(y2 + y_off),
                            "conf": conf
                        })
        return outs
    except Exception as e:
        # Fallback: if anything goes wrong with batched path, use the original per-ROI loop
        d2(f"[warn] batched ROI inference failed, falling back to per-ROI loop: {e}")
        return yolo_vehicle_detections_roi(model, img_pil, rois)


def yolo_vehicle_detections_any(model: YOLO, img_pil, sample_idx_in_scene, rois, state: "SceneSweepState"):
    """
    Policy order:
      1) ROIがあれば ROI 推論
      2) ROIが無ければ coarse fallback（グリッド）で ROI 風推論
      3) FULL_SWEEP_POLICY に応じてフルスイープ（none: しない / periodic / adaptive）
    Returns: (detections, used_full: bool, roi_area_px: int)
    """
    w, h = img_pil.size

    # Helper: run full-image inference (downscaled if needed)
    def _run_full():
        if min(w, h) > FULL_SWEEP_SHORT_SIDE:
            if w < h:
                new_w = FULL_SWEEP_SHORT_SIDE
                new_h = int(h * (new_w / w))
            else:
                new_h = FULL_SWEEP_SHORT_SIDE
                new_w = int(w * (new_h / h))
            img_small = img_pil.resize((new_w, new_h), Image.BILINEAR)
            outs_small = yolo_vehicle_detections_full(model, img_small)
            sx, sy = (w / new_w), (h / new_h)
            outs = [{"x1": int(b["x1"]*sx), "y1": int(b["y1"]*sy), "x2": int(b["x2"]*sx), "y2": int(b["y2"]*sy)} for b in outs_small]
        else:
            outs = yolo_vehicle_detections_full(model, img_pil)
        return outs

    if not USE_ROI:
        outs = _run_full()
        state.frames_since_full = 0
        return outs, True, 0

    # 1) Prefer radar ROIs
    if rois:
        outs = yolo_vehicle_detections_roi_batched(model, img_pil, rois, ROI_BATCH_SIZE)
        roi_px = sum(max(0, r["x2"]-r["x1"]) * max(0, r["y2"]-r["y1"]) for r in rois)
        state.frames_since_full += 1
        state.consecutive_roi_zero = 0
        return outs, False, roi_px

    # 2) No radar ROI -> build fallback tiles (slightly denser if long time since full)
    grid_scale = 1.0 if state.frames_since_full < ADAPTIVE_FULL_MIN_GAP else 1.5
    fallback_rois = build_coarse_fallback_rois((w, h), grid_scale=grid_scale)
    # Apply ROI area budget to fallback tiles to avoid near-full-image coverage
    fallback_rois = _apply_roi_budget(fallback_rois, w, h, ROI_AREA_BUDGET_RATIO, MAX_NUM_ROI)
    outs = yolo_vehicle_detections_roi_batched(model, img_pil, fallback_rois, ROI_BATCH_SIZE)
    roi_px = sum(max(0, r["x2"]-r["x1"]) * max(0, r["y2"]-r["y1"]) for r in fallback_rois)

    # ヒットが無ければ「ROIゼロ」扱いでカウンタを進める
    state.frames_since_full += 1
    state.consecutive_roi_zero += 1

    # 3) Check policy for full sweep
    do_full = False
    if FULL_SWEEP_POLICY == "none":
        do_full = False
    elif FULL_SWEEP_POLICY == "periodic":
        do_full = (sample_idx_in_scene % max(1, FULL_SWEEP_EVERY) == 0)
    elif FULL_SWEEP_POLICY == "adaptive":
        do_full = (state.frames_since_full >= ADAPTIVE_FULL_MIN_GAP and
                   state.consecutive_roi_zero >= ADAPTIVE_FULL_MISS_THRESH)
    else:
        do_full = False

    if do_full:
        outs_full = _run_full()
        state.frames_since_full = 0
        state.consecutive_roi_zero = 0
        return outs_full, True, 0

    return outs, False, roi_px


# ---------- ★BEV風ゲーティング ----------

def check_radar_in_box(nusc: NuScenes, sample: dict, ann_token: str, nsweeps=NSWEEPS) -> int:
    """
    レーダー点が対象GTボックスの中にあるかを判定（XY主体＋緩いZ）。
    """
    try:
        pts_g, _ = radar_points_global(nusc, sample, nsweeps=nsweeps)
        if pts_g is None or pts_g.shape[1] == 0:
            return 0

        box = nusc.get_box(ann_token)  # has .center, .wlh (w,l,h), .rotation_matrix
        R = box.rotation_matrix
        c = box.center.reshape(3,1)
        w, l, h = box.wlh

        pts_local = R.T @ (pts_g - c)  # 3xN

        scale = 1.6
        half_x = (w*scale)/2.0   # 横（幅）
        half_y = (l*scale)/2.0   # 奥（長さ）

        inside_xy = (np.abs(pts_local[0,:]) <= half_x) & (np.abs(pts_local[1,:]) <= half_y)
        inside_z  = (np.abs(pts_local[2,:]) <= 3.0)  # レーダーZの粗さを吸収
        inside = inside_xy & inside_z
        return int(np.count_nonzero(inside))
    except Exception:
        return 0


# === シーン選別（全天候：I/O解決のみ確認） ===

def filter_scenes(nusc: NuScenes):
    scenes = nusc.scene
    selected = []
    for s in scenes:
        if USE_BAD_WEATHER_ONLY:
            desc = (s.get("description") or "").lower()
            if not any(k in desc for k in BAD_WEATHER_KEYWORDS):
                continue
        sample = nusc.get("sample", s["first_sample_token"])
        if "CAM_FRONT" not in sample["data"] or "RADAR_FRONT" not in sample["data"]:
            continue
        try:
            _ = nusc.get_sample_data_path(sample["data"]["CAM_FRONT"])
            _ = nusc.get_sample_data_path(sample["data"]["RADAR_FRONT"])
            selected.append(s)
        except Exception:
            continue
        if MAX_SCENES and len(selected) >= MAX_SCENES:
            break
    return selected

### 追加: シンプルな追跡用クラス ###
class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, detections):
        # 既存トラックと新規検出のマッチング
        matches = []
        used_det_indices = set()
        for i, track in enumerate(self.tracks):
            best_iou = 0
            best_det_idx = -1
            for j, det in enumerate(detections):
                if j in used_det_indices:
                    continue
                # calculate_iouはグローバルに定義されている関数を使用
                iou = calculate_iou(track['box'], det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = j
            
            if best_iou > self.iou_threshold:
                matches.append((i, best_det_idx))
                used_det_indices.add(best_det_idx)

        # マッチしたトラックを更新
        matched_track_indices = {m[0] for m in matches}
        for track_idx, det_idx in matches:
            self.tracks[track_idx]['box'] = detections[det_idx]
            self.tracks[track_idx]['age'] = 0

        # マッチしなかったトラックの年齢を増やす
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track['age'] += 1

        # 古くなったトラックを削除
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        # マッチしなかった検出から新しいトラックを作成
        for i, det in enumerate(detections):
            if i not in used_det_indices:
                self.tracks.append({'id': self.next_id, 'box': det, 'age': 0})
                self.next_id += 1
        
        return self.tracks
    
    def merge_overlapping_rois(rois, iou_threshold=0.5):
        if not rois:
            return []
    
        # 面積の大きい順にソート
        rois.sort(key=lambda r: (r['x2'] - r['x1']) * (r['y2'] - r['y1']), reverse=True)
    
        merged = []
        while rois:
            current_roi = rois.pop(0)
            remaining_rois = []
            for other_roi in rois:
                iou = calculate_iou(current_roi, other_roi)
                if iou > iou_threshold:
                    # 重なっていたら統合
                    current_roi['x1'] = min(current_roi['x1'], other_roi['x1'])
                    current_roi['y1'] = min(current_roi['y1'], other_roi['y1'])
                    current_roi['x2'] = max(current_roi['x2'], other_roi['x2'])
                    current_roi['y2'] = max(current_roi['y2'], other_roi['y2'])
                else:
                    remaining_rois.append(other_roi)
            merged.append(current_roi)
            rois = remaining_rois
        
        return merged

# ================== メイン ==================

def main():
    # parse CLI options and override defaults
    args = _parse_args()
    global YOLO_MODEL, YOLO_CONF, ROI_AREA_BUDGET_RATIO
    YOLO_MODEL = args.yolo_model
    YOLO_CONF = args.conf
    if args.roi_budget is not None:
        ROI_AREA_BUDGET_RATIO = max(0.0, min(1.0, args.roi_budget))

    print("[1/7] Load NuScenes...")
    nusc = NuScenes(version=NUSC_VERSION, dataroot=PRIMARY_DATAROOT, verbose=True)
    print("[2/7] Patch path resolver...")
    patch_get_sample_data_path_multi(nusc)

    print("[3/7] Pre-screen scenes...")
    scenes = filter_scenes(nusc)
    print(f"  -> candidate scenes: {len(scenes)}")
    if not scenes:
        print("No scenes with both CAM_FRONT and RADAR_FRONT resolvable. Check PART_ROOTS.")
        return

    print("[4/7] Load YOLO...")
    model = YOLO(YOLO_MODEL)
    if args.device:
        try:
            model.to(args.device)
        except Exception as _e:
            print(f"[warn] Could not move model to device '{args.device}': {_e}. Using default device.", flush=True)

    # 結果集計用
    total_pairs = 0
    radar_first = 0
    cam_first = 0
    simultaneous = 0
    radar_leads = []

    # タイルベース混同行列の累積
    sum_tp = 0
    sum_tn = 0
    sum_fp = 0
    sum_fn = 0

    # 速度/負荷観測
    total_ms = 0.0
    total_px = 0         # 処理した総ピクセル数（ROIなら合計面積）
    total_full_calls = 0
    total_roi_calls = 0
    
    ### 追加: パフォーマンス計測用のリスト ###
    roi_gen_times = []
    yolo_inf_times = []


    # 天候別シーン数
    weather_scene_counts = {}

    # Box-level cumulative counters
    sum_box_tp = 0
    sum_box_fp = 0
    sum_box_fn = 0

    # Dataset-level PR/AP accumulators
    pr_scores = []
    pr_is_tp = []
    pr_total_gt = [0]  # mutable counter for total GT count
    iou_eval_thr = float(getattr(args, "eval_iou", 0.50))

    print("[5/7] Iterate samples & measure timing...")
    dev_str = getattr(args, "device", None) or "auto"
    print(f"  CONFIG: IOU_THRESH={IOU_THRESH} NSWEEPS={NSWEEPS} RADAR_MIN_PTS={RADAR_MIN_PTS} YOLO_MODEL={YOLO_MODEL} YOLO_CONF={YOLO_CONF} DEVICE={dev_str} USE_ROI={USE_ROI} ROI_BUDGET={int(ROI_AREA_BUDGET_RATIO*100)}% [BUILD {BUILD_ID}]", 
          flush=True)

    for si, scene in enumerate(scenes, 1):
        # 天候ラベル集計
        wtag = _tag_weather(scene.get("description") or "")
        weather_scene_counts[wtag] = weather_scene_counts.get(wtag, 0) + 1

        token = scene["first_sample_token"]
        vehicle_hist = {}
        start_scene = time.time()
        sample_idx_in_scene = 0
        sweep_state = SceneSweepState()
        
        ### 修正点1: シーンごとにTrackerを初期化 ###
        tracker = SimpleTracker()

        while token:
            sample = nusc.get("sample", token)
            ts = sample["timestamp"]

            cam_t = sample["data"]["CAM_FRONT"]
            try:
                cam_path = nusc.get_sample_data_path(cam_t)
            except Exception:
                token = sample["next"]; continue

            try:
                img = Image.open(cam_path).convert("RGB")
                w, h = img.size
            except Exception:
                token = sample["next"]; continue

            # --- 1. 追跡情報から「予測ROI」を生成 ---
            predicted_rois = []
            for track in tracker.tracks:
                box = track['box']
                pad_w = (box['x2'] - box['x1']) * 0.1
                pad_h = (box['y2'] - box['y1']) * 0.1
                predicted_rois.append({
                    'x1': int(max(0, box['x1'] - pad_w)), 
                    'y1': int(max(0, box['y1'] - pad_h)), 
                    'x2': int(min(w - 1, box['x2'] + pad_w)), 
                    'y2': int(min(h - 1, box['y2'] + pad_h)), 
                    'depth': 5.0
                })

            # --- 2. レーダーROIと予測ROIを統合 ---
            t_roi_start = time.perf_counter()
            radar_rois = []
            if USE_ROI:
                radar_rois = build_rois_from_radar(nusc, sample, cam_t, (w, h))
            t_roi_end = time.perf_counter()
            roi_gen_times.append((t_roi_end - t_roi_start) * 1000.0)
            
            combined_rois = radar_rois + predicted_rois
            final_rois = merge_overlapping_rois(combined_rois) # 重複除去を実行

            # ... (デバッグ出力など。rois変数をcombined_roisに置き換えるのを推奨) ...
            
            # === このフレームのGT 2Dボックス蓄積（タイル評価用） ===
            gt2d_list = []

            ### 変更: YOLO推論の時間を計測 ###
            t_yolo_start = time.perf_counter()
            yolo_boxes, used_full, used_roi_px = yolo_vehicle_detections_any(model, img, sample_idx_in_scene, combined_rois, sweep_state)
            t_yolo_end = time.perf_counter()
            
            ### 修正点2: dt_msの計算とyolo_inf_timesへの追加 ###
            dt_ms = (t_yolo_end - t_yolo_start) * 1000.0
            yolo_inf_times.append(dt_ms)
            total_ms += dt_ms

            # --- 4. 検出結果でTrackerを更新 ---
            plain_boxes = [{"x1":b["x1"],"y1":b["y1"],"x2":b["x2"],"y2":b["y2"]} for b in yolo_boxes]
            tracker.update(plain_boxes)

            if used_full:
                total_full_calls += 1
                total_px += (w * h)
                # safety: already handled inside yolo_vehicle_detections_any, but keep consistent
                # sweep_state.frames_since_full = 0
                # sweep_state.consecutive_roi_zero = 0
            else:
                total_roi_calls += 1
                total_px += used_roi_px # used_roi_pxはyolo_vehicle_detections_anyから返される

            # === DEBUG counters ===
            dbg = {
                'yolo': len(yolo_boxes),
                'anns_total': len(sample['anns']),
                'anns_vehicle': 0,
                'gt2d_ok': 0,
                'iou_hit': 0,
                'n_roi': len(rois),
                'used_full': used_full,
                'inference_ms': int(dt_ms),
                'roi_pixel_ratio_%': 0
            }
            if not used_full:
                dbg['roi_pixel_ratio_%'] = int((used_roi_px * 100) / max(1, (w*h)))

            # === 先行/同時の集計 ===
            for ann_t in sample['anns']:
                ann = nusc.get('sample_annotation', ann_t)
                if 'vehicle' not in ann['category_name']:
                    continue
                inst = ann['instance_token']
                dbg['anns_vehicle'] += 1
                rec = vehicle_hist.setdefault(inst, {'first_radar_ts':None, 'first_camera_ts':None})

                if rec['first_radar_ts'] is None:
                    npts = check_radar_in_box(nusc, sample, ann_t, nsweeps=NSWEEPS)
                    if npts >= RADAR_MIN_PTS:
                        rec['first_radar_ts'] = ts

                if rec['first_camera_ts'] is None:
                    gt2d = get_gt_2d_box(nusc, ann_t, cam_t, (w, h))
                    if gt2d is not None:
                        # タイル評価用に保持
                        gt2d_list.append(gt2d)
                        # 先行/同時の集計用ヒット判定
                        for det in yolo_boxes:
                            iou = calculate_iou(gt2d, det)
                            hit = (iou >= IOU_THRESH) or _contains(gt2d, *_center(det))
                            if hit:
                                dbg['iou_hit'] += 1
                                rec['first_camera_ts'] = ts
                                break
                    else:
                        # GTが2Dに投影できない場合はスキップ
                        pass
            
                        # === Box-level evaluation (per frame) ===
            if gt2d_list:
                det_boxes_plain = [{"x1":b["x1"],"y1":b["y1"],"x2":b["x2"],"y2":b["y2"]} for b in yolo_boxes]
                tp_b, fp_b, fn_b = box_eval_counts(gt2d_list, det_boxes_plain, iou_thr=iou_eval_thr)
                sum_box_tp += tp_b; sum_box_fp += fp_b; sum_box_fn += fn_b

                # PR/AP accumulation if requested (needs confidences)
                if args.eval_map:
                    pr_accumulate_frame(gt2d_list, yolo_boxes, iou_eval_thr, pr_scores, pr_is_tp, pr_total_gt)


            # === タイルベース混同行列をフレーム単位で加算 ===
            if gt2d_list:
                tp_t, tn_t, fp_t, fn_t = confusion_tiles(gt2d_list, yolo_boxes, (w, h))
                sum_tp += tp_t
                sum_tn += tn_t
                sum_fp += fp_t
                sum_fn += fn_t

            if sample_idx_in_scene < 3:
                mode = "FULL" if used_full else f"ROI(n={len(rois)}, {dbg['roi_pixel_ratio_%']}%)"
                print(f"  [dbg] {mode}  time={dbg['inference_ms']}ms  yolo={dbg['yolo']}  anns={dbg['anns_total']} veh={dbg['anns_vehicle']} gt2d_ok={dbg['gt2d_ok']} iou_hit={dbg['iou_hit']}")
            sample_idx_in_scene += 1
            token = sample["next"]

        # シーン集計
        scene_pairs = 0; sc_r=0; sc_c=0; sc_s=0
        for rec in vehicle_hist.values():
            if rec['first_radar_ts'] and rec['first_camera_ts']:
                dt = (rec['first_camera_ts'] - rec['first_radar_ts'])/1e6
                scene_pairs += 1; total_pairs += 1
                if dt > 0.001:
                    radar_first += 1; sc_r += 1; radar_leads.append(dt)
                elif dt < -0.001:
                    cam_first += 1; sc_c += 1
                else:
                    simultaneous += 1; sc_s += 1

        elapsed = math.ceil(time.time() - start_scene)
        print(f"[scene {si}/{len(scenes)}] {scene['name']}: "
              f"pairs={scene_pairs}  radar_first={sc_r}  cam_first={sc_c}  sim={sc_s}  time={elapsed}s")

    print("\n[6/7] Timing & Load Summary")
    ### 変更: `total_ms` の平均計算部分を `yolo_inf_times` の平均に置き換え ###
    avg_total_inference_time = np.mean(yolo_inf_times) if yolo_inf_times else 0
    print(f"  avg_inference_time = {avg_total_inference_time:.1f} ms/frame")
    print(f"  calls: full={total_full_calls}  roi={total_roi_calls}")
    print(f"  processed_pixel_equiv = {total_px/1e6:.2f} MPix (sum, for rough compute proxy)")

    print("\n[7/7] Detection Summary")
    print(f" total_pairs={total_pairs}")
    if total_pairs > 0:
        print(f"  radar_first: {radar_first} ({radar_first/total_pairs:.1%})")
        print(f"  cam_first:   {cam_first} ({cam_first/total_pairs:.1%})")
        print(f"  simultaneous:{simultaneous} ({simultaneous/total_pairs:.1%})")
        if radar_leads:
            print(f"  radar lead avg={np.mean(radar_leads):.3f}s max={np.max(radar_leads):.3f}s")
    else:
        print("  No matched pairs (check thresholds / data availability).")

    # ===== [8/8] Tile-based Confusion Matrix =====
    total_tiles = sum_tp + sum_tn + sum_fp + sum_fn
    if total_tiles > 0:
        acc = (sum_tp + sum_tn) / total_tiles
        recall = sum_tp / max(1, (sum_tp + sum_fn))       # = TPR
        specificity = sum_tn / max(1, (sum_tn + sum_fp))  # = TNR
        precision = sum_tp / max(1, (sum_tp + sum_fp))
        f1 = 2 * precision * recall / max(1e-12, (precision + recall))
        print("\n[8/8] Tile-based Confusion (IoU>=%.2f, grid=%dx%d)" % (IOU_EVAL_THR, TILE_W, TILE_H))
        print(f"  TP={sum_tp}  TN={sum_tn}  FP={sum_fp}  FN={sum_fn}  (tiles total={total_tiles})")
        print(f"  Acc={acc:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  Specificity={specificity:.3f}  F1={f1:.3f}")
    else:
        print("\n[8/8] Tile-based Confusion: no tiles counted (check TILE_W/H settings).")

    # 参考：天候別内訳
    if weather_scene_counts:
        print("\n[Appendix] Scene counts by weather tag (rough):")
        for k,v in sorted(weather_scene_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k:>7}: {v}")
    
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

    # ===== [10/10] Dataset PR/AP (confidence-aware) =====
    if args.eval_map:
        recalls, precisions, ap = _compute_ap(pr_scores, pr_is_tp, pr_total_gt[0], pr_points=getattr(args, "pr_curve_points", 101))
        print("\n[10/10] PR/AP (IoU>=%.2f)" % (iou_eval_thr,))
        print(f"  total_gt={pr_total_gt[0]}  dets={len(pr_scores)}  AP={ap:.3f}")
        if len(recalls) > 0 and len(precisions) > 0:
            for i in np.linspace(0, len(recalls)-1, num=5, dtype=int):
                print(f"   r={recalls[i]:.2f}  p={precisions[i]:.2f}")
        else:
            print("  (no detections or no GT; cannot compute PR)")

    ### 追加: パフォーマンス計測結果の最終出力 ###
    print("\n[Appendix] Performance Profiling (avg per frame)")
    avg_roi_gen = np.mean(roi_gen_times) if roi_gen_times else 0
    avg_yolo_inf = np.mean(yolo_inf_times) if yolo_inf_times else 0
    
    print(f"  ROI Generation Time : {avg_roi_gen:.2f} ms")
    print(f"  YOLO Inference Time : {avg_yolo_inf:.2f} ms")
    # USE_ROIがFalseの場合はROI生成時間は0なので、合計時間はYOLO推論時間と同じになる
    if USE_ROI:
        print(f"  ------------------------------------")
        print(f"  Total (ROI Gen + YOLO): {avg_roi_gen + avg_yolo_inf:.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()