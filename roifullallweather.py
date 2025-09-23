

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
YOLO_MODEL = "yolov8s.pt"
VEHICLE_CLASS_IDS = {1,2,3,5,7}
MAX_SCENES = None

# ===== ROI 関連設定 =====
USE_ROI = True
ROI_GRID = 48
ROI_MIN_PTS = 2
ROI_PAD_RATIO = 0.25
ROI_SIZE_NEAR = 220
ROI_SIZE_FAR = 80
ROI_NEAR_M = 20.0
ROI_FAR_M  = 70.0
MAX_NUM_ROI = 12
FULL_SWEEP_EVERY = 10
FULL_SWEEP_SHORT_SIDE = 512

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
        rois.append({"x1":x1, "y1":y1, "x2":x2, "y2":y2, "depth": d_med})

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
    d2(f"[dbg2/b4] ROIs raw={rois_raw} merged={len(merged)} (MAX={MAX_NUM_ROI})")
    return merged


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
            outs.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2)})
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
                all_out.append({
                    "x1": int(x1 + r["x1"]), "y1": int(y1 + r["y1"]),
                    "x2": int(x2 + r["x1"]), "y2": int(y2 + r["y1"]) 
                })
    return all_out


def yolo_vehicle_detections_any(model: YOLO, img_pil, sample_idx_in_scene, rois):
    """
    ROIが有効: ROIあり→ROI推論 / ROIなし→全体スイープ
    ROIを使わない設定: 常に全体
    全体スイープは FULL_SWEEP_EVERY ごとに必ず走らせる（盲点対策）
    """
    use_full = (not USE_ROI) or (not rois)
    force_full = (sample_idx_in_scene % FULL_SWEEP_EVERY == 0)

    if use_full or force_full:
        # 解像度を落として実施（短辺=FULL_SWEEP_SHORT_SIDE）
        w, h = img_pil.size
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
            outs = []
            for b in outs_small:
                outs.append({
                    "x1": int(b["x1"] * sx), "y1": int(b["y1"] * sy),
                    "x2": int(b["x2"] * sx), "y2": int(b["y2"] * sy)
                })
        else:
            outs = yolo_vehicle_detections_full(model, img_pil)
        return outs, True  # full=true
    else:
        outs = yolo_vehicle_detections_roi(model, img_pil, rois)
        return outs, False  # full=false


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


# ================== メイン ==================

def main():
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

    # 結果集計用
    total_pairs = 0
    radar_first = 0
    cam_first = 0
    simultaneous = 0
    radar_leads = []

    # 速度/負荷観測
    total_ms = 0.0
    total_px = 0         # 処理した総ピクセル数（ROIなら合計面積）
    total_full_calls = 0
    total_roi_calls = 0

    # 天候別シーン数
    weather_scene_counts = {}

    print("[5/7] Iterate samples & measure timing...")
    print(f"  CONFIG: IOU_THRESH={IOU_THRESH} NSWEEPS={NSWEEPS} RADAR_MIN_PTS={RADAR_MIN_PTS} YOLO_MODEL={YOLO_MODEL} YOLO_CONF={YOLO_CONF} USE_ROI={USE_ROI} [BUILD {BUILD_ID}]", flush=True)

    for si, scene in enumerate(scenes, 1):
        # 天候ラベル集計
        wtag = _tag_weather(scene.get("description") or "")
        weather_scene_counts[wtag] = weather_scene_counts.get(wtag, 0) + 1

        token = scene["first_sample_token"]
        vehicle_hist = {}  # instance_token -> {'first_radar_ts':None,'first_camera_ts':None}
        start_scene = time.time()
        sample_idx_in_scene = 0

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

            # ---- ROI生成（必要時のみ）----
            rois = []
            if USE_ROI:
                rois = build_rois_from_radar(nusc, sample, cam_t, (w, h))
                if sample_idx_in_scene < 3:
                    px = sum((r["x2"]-r["x1"]) * (r["y2"]-r["y1"]) for r in rois) if rois else 0
                    d2(f"[dbg2/b5] scene={scene['name']} sample_idx={sample_idx_in_scene} "
                       f"ROI_n={len(rois)} ROI_px%={int(px*100/(w*h)) if (w*h)>0 else 0}")

            # ---- YOLO呼び出し（ROI/全体スイープ切替）----
            t0 = time.perf_counter()
            yolo_boxes, used_full = yolo_vehicle_detections_any(model, img, sample_idx_in_scene, rois)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            total_ms += dt_ms

            if used_full:
                total_full_calls += 1
                total_px += (w * h)
            else:
                total_roi_calls += 1
                roi_px = 0
                for r in rois:
                    roi_px += max(0, r["x2"]-r["x1"]) * max(0, r["y2"]-r["y1"])
                total_px += roi_px

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
            if not used_full and rois:
                roi_px = sum((r["x2"]-r["x1"]) * (r["y2"]-r["y1"]) for r in rois)
                dbg['roi_pixel_ratio_%'] = int(roi_px * 100 / (w*h))

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
                        for det in yolo_boxes:
                            iou = calculate_iou(gt2d, det)
                            hit = (iou >= IOU_THRESH) or _contains(gt2d, *_center(det))
                            if hit:
                                dbg['iou_hit'] += 1
                                rec['first_camera_ts'] = ts
                                break

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
    print(f"  avg_inference_time = {total_ms / max(1,(total_full_calls+total_roi_calls)):.1f} ms/frame")
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

    # 参考：天候別内訳
    if weather_scene_counts:
        print("\n[Appendix] Scene counts by weather tag (rough):")
        for k,v in sorted(weather_scene_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k:>7}: {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()