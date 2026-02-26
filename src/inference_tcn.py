"""
Real-time TCN inference with rtmlib skeleton extraction.

rtmlib Wholebody → 133 keypoints → face 제거 (65 joints) or body 17 → 프레임 버퍼 → TCN 추론

65-joint layout (face 제거):
    0-16:  Body (COCO 17)
   17-22:  Feet (6)
   23-43:  Left Hand (21)
   44-64:  Right Hand (21)

Usage:
    # 웹캠
    python src/inference_tcn.py

    # 비디오 파일
    python src/inference_tcn.py --source video.mp4

    # 이미지 (프레임 반복하여 시퀀스 생성)
    python src/inference_tcn.py --source image.jpg
"""

import sys
import os
import argparse
import time
from collections import deque

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import cv2
import torch
import numpy as np

from models.tcn import TCN
from data.ntu_dataset import NTU_ACTION_NAMES


# ===== 클래스별 표시 색상 (BGR) =====
CLASS_COLORS = {
    "idle":             (128, 128, 128),  # gray
    "waving":           (0, 165, 255),    # orange
    "hands_up_single":  (0, 0, 255),      # red
    "hands_up_both":    (0, 100, 255),    # dark orange
    "pointing":         (0, 255, 0),      # green
    "unknown":          (80, 80, 80),     # dark gray
}


# ===== Track state 관리 =====
class TrackState:
    """각 track_id별 상태를 관리하는 클래스"""
    def __init__(self, buf_size=60):
        self.buffer = deque(maxlen=buf_size)  # normalized keypoint frames
        self.pred_cls = "unknown"
        self.pred_conf = 0.0
        self.pred_probs = {}
        self.last_seen_frame = 0
        self.last_depth_z = None
        self.bbox_area = 0  # for priority in hands_up selection


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time TCN action recognition")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: webcam index (0,1,...) or file path")
    parser.add_argument("--model", type=str, default=None,
                        help="TCN checkpoint (default: models/best_tcn_xsub.pth)")
    parser.add_argument("--rtmlib_path", type=str,
                        default="rtmlib")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["balanced", "performance", "lightweight"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--kpt_thr", type=float, default=0.4,
                        help="Keypoint threshold for skeleton drawing")
    parser.add_argument("--buf_size", type=int, default=60,
                        help="Frame buffer size for TCN (default: 60)")
    parser.add_argument("--infer_every", type=int, default=5,
                        help="Run TCN every N frames (default: 5)")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────

def load_tcn(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = TCN(
        input_dim=checkpoint.get("input_dim", 34),
        num_classes=checkpoint.get("num_classes", 5),
        hidden_dims=checkpoint.get("hidden_dims", [64, 128, 128, 256]),
        kernel_size=checkpoint.get("kernel_size", 5),
        dropout=checkpoint.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    info = {
        "num_classes": checkpoint.get("num_classes", 5),
        "num_joints": checkpoint.get("num_joints", 17),
        "input_dim": checkpoint.get("input_dim", 34),
        "max_frames": checkpoint.get("max_frames", 120),
        "epoch": checkpoint.get("epoch", "?"),
        "val_acc": checkpoint.get("val_acc", "?"),
    }
    return model, info


# ─────────────────────────────────────────────
#  Per-frame skeleton preprocessing
#  (학습 시 ntu_dataset.py의 _normalize_skeleton과 동일)
# ─────────────────────────────────────────────

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


# RTMPose 133 → 65 joints (face 제거)
_KEEP_IDXS_65 = list(range(0, 23)) + list(range(91, 133))


def extract_skeleton(keypoints, scores, num_joints=65, person_idx=0):
    """
    rtmlib 133-keypoint output → body 17 or wholebody 65 joints.

    Args:
        keypoints: (N_persons, 133, 2)
        scores:    (N_persons, 133)
        num_joints: 17 (body only) or 65 (body+feet+hands)
        person_idx: 어떤 사람을 쓸지

    Returns:
        kp: (num_joints, 2) or None
        sc: (num_joints,) or None
    """
    if keypoints is None or len(keypoints) == 0:
        return None, None

    pidx = min(person_idx, len(keypoints) - 1)

    if num_joints == 65:
        kp = keypoints[pidx][_KEEP_IDXS_65].copy()   # (65, 2)
        sc = scores[pidx][_KEEP_IDXS_65].copy()       # (65,)
    else:
        kp = keypoints[pidx][:17].copy()   # (17, 2)
        sc = scores[pidx][:17].copy()      # (17,)

    return kp, sc


def normalize_frame(kp, sc, min_score=0.3):
    """
    단일 프레임 skeleton 정규화 (학습 시와 동일).

    Args:
        kp: (V, 2) — x, y (pixel coords), V=17 or 65
        sc: (V,)   — confidence

    Returns:
        kp_norm: (V, 2) — normalized coords
    """
    kp = kp.copy()

    ls = kp[LEFT_SHOULDER]
    rs = kp[RIGHT_SHOULDER]

    # 어깨 중점 이동
    center = (ls + rs) / 2.0
    kp -= center

    # 어깨 거리 스케일링
    scale = np.linalg.norm(kp[LEFT_SHOULDER] - kp[RIGHT_SHOULDER])
    if scale > 1e-6:
        kp /= scale

    # Low confidence → zero
    low_conf = sc < min_score
    kp[low_conf] = 0.0

    return kp


# ─────────────────────────────────────────────
#  Frame buffer → TCN input
# ─────────────────────────────────────────────

def buffer_to_tensor(frame_buffer, max_frames, device, num_joints=65):
    """
    frame_buffer: deque of (V, 2) numpy arrays (V=17 or 65)

    Returns:
        features: (1, V*2, T) tensor
        mask:     (1, T) tensor
    """
    T = len(frame_buffer)
    input_dim = num_joints * 2
    arr = np.stack(list(frame_buffer), axis=0)  # (T, V, 2)

    # Pad or crop to max_frames
    if T >= max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        arr = arr[indices]
        mask = np.ones(max_frames, dtype=np.float32)
    else:
        padded = np.zeros((max_frames, num_joints, 2), dtype=np.float32)
        padded[:T] = arr
        arr = padded
        mask = np.zeros(max_frames, dtype=np.float32)
        mask[:T] = 1.0

    # (max_frames, V, 2) → (V*2, max_frames)
    features = arr.reshape(max_frames, input_dim).T  # (input_dim, max_frames)

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    return features, mask


# ─────────────────────────────────────────────
#  TCN inference
# ─────────────────────────────────────────────
#  Pose-based probability adjustment
# ─────────────────────────────────────────────

# Joint indices (COCO 17)
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10

def compute_pose_features(frame_buffer):
    """
    Analyze recent frames to extract pose-level features:
      - wrist_above: is either wrist above its shoulder? (spatial)
      - temporal_std: how much are wrists moving? (temporal)

    Returns dict of features.
    """
    if len(frame_buffer) < 5:
        return {"wrist_above": False, "n_hands_up": 0, "temporal_std": 1.0}

    recent = list(frame_buffer)[-30:]  # last 30 normalized frames
    arr = np.stack(recent, axis=0)  # (T, 17, 2)

    # --- Spatial: wrist above shoulder (in normalized coords, y < 0 = above center) ---
    last = arr[-1]  # latest frame
    l_wrist_y = last[_L_WRIST, 1]
    r_wrist_y = last[_R_WRIST, 1]
    l_shoulder_y = last[_L_SHOULDER, 1]
    r_shoulder_y = last[_R_SHOULDER, 1]

    l_up = l_wrist_y < l_shoulder_y - 0.3  # significantly above shoulder
    r_up = r_wrist_y < r_shoulder_y - 0.3
    n_hands_up = int(l_up) + int(r_up)

    # --- Temporal: wrist movement std (low = static hand raise, high = waving) ---
    l_wrist_xy = arr[:, _L_WRIST, :]  # (T, 2)
    r_wrist_xy = arr[:, _R_WRIST, :]
    l_std = np.std(l_wrist_xy, axis=0).mean()
    r_std = np.std(r_wrist_xy, axis=0).mean()
    temporal_std = max(l_std, r_std)

    return {
        "wrist_above": (n_hands_up >= 1),
        "n_hands_up": n_hands_up,
        "temporal_std": temporal_std,
    }


def adjust_probs(probs_dict, pose_features, num_classes):
    """
    Adjust TCN probabilities using pose-based heuristics.

    Key insight:
      - hands_up_single: wrist above shoulder + LOW temporal movement
      - waving: wrist above shoulder + HIGH temporal movement
      - hands_up_both: BOTH wrists above shoulder + LOW movement
    """
    if not pose_features["wrist_above"]:
        return probs_dict  # no hand above shoulder, trust TCN

    temporal_std = pose_features["temporal_std"]
    n_hands = pose_features["n_hands_up"]

    # Threshold: below this = static (hands up), above = dynamic (waving)
    STATIC_THR = 0.15
    DYNAMIC_THR = 0.25

    adjusted = dict(probs_dict)

    if temporal_std < STATIC_THR:
        # Static hand raise → boost hands_up, suppress waving
        if n_hands == 1:
            boost_cls = "hands_up_single"
        else:
            boost_cls = "hands_up_both"

        # Transfer probability from waving → hands_up
        transfer = adjusted.get("waving", 0) * 0.6
        adjusted["waving"] = adjusted.get("waving", 0) - transfer
        adjusted[boost_cls] = adjusted.get(boost_cls, 0) + transfer

    elif temporal_std > DYNAMIC_THR:
        # Clear waving motion → boost waving slightly
        transfer = adjusted.get("hands_up_single", 0) * 0.3
        adjusted["hands_up_single"] = adjusted.get("hands_up_single", 0) - transfer
        adjusted["waving"] = adjusted.get("waving", 0) + transfer

    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


# ─────────────────────────────────────────────

@torch.no_grad()
def predict_tcn(model, frame_buffer, max_frames, num_classes, device, num_joints=65):
    """
    Returns: (class_name, confidence, probs_dict)
    """
    if len(frame_buffer) < 5:
        return "unknown", 0.0, {}

    features, mask = buffer_to_tensor(frame_buffer, max_frames, device, num_joints)
    logits = model(features, mask)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    probs_dict = {
        NTU_ACTION_NAMES.get(i, f"class_{i}"): probs[i].item()
        for i in range(num_classes)
    }

    # Pose-based adjustment
    pose_features = compute_pose_features(frame_buffer)
    probs_dict = adjust_probs(probs_dict, pose_features, num_classes)

    pred_cls = max(probs_dict, key=probs_dict.get)
    confidence = probs_dict[pred_cls]

    return pred_cls, confidence, probs_dict


# ─────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────

def draw_label(frame, position, class_name, confidence, buf_len, buf_max):
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    label = f"{class_name} {confidence:.0%}"

    cx, cy = int(position[0]), int(position[1])

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (cx - tw // 2 - 5, cy - th - 10),
                  (cx + tw // 2 + 5, cy + 5), color, -1)
    cv2.putText(frame, label, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Buffer indicator
    bar_text = f"buf: {buf_len}/{buf_max}"
    cv2.putText(frame, bar_text, (cx - tw // 2, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_hud(frame, fps, pose_ms, tcn_ms, buf_len, buf_max, probs_dict=None):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Pose: {pose_ms:.0f}ms  TCN: {tcn_ms:.0f}ms",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Buffer: {buf_len}/{buf_max}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Draw per-class probability bars
    if probs_dict:
        y_start = 115
        bar_w = 150
        for i, (cls_name, prob) in enumerate(probs_dict.items()):
            y = y_start + i * 22
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            # Bar background
            cv2.rectangle(frame, (10, y - 12), (10 + bar_w, y + 4), (40, 40, 40), -1)
            # Bar fill
            cv2.rectangle(frame, (10, y - 12), (10 + int(bar_w * prob), y + 4), color, -1)
            # Label
            cv2.putText(frame, f"{cls_name[:12]:12s} {prob:.0%}", (10 + bar_w + 5, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = args.model or os.path.join(ROOT_DIR, "models", "best_tcn_xsub.pth")

    # ===== Load TCN model =====
    print(f"Loading TCN model: {model_path}")
    model, model_info = load_tcn(model_path, device)
    max_frames = model_info["max_frames"]
    num_classes = model_info["num_classes"]
    num_joints = model_info["num_joints"]
    print(f"  Classes: {num_classes}, Joints: {num_joints}, Max frames: {max_frames}")
    print(f"  Trained epoch: {model_info['epoch']}, Val acc: {model_info['val_acc']}")

    # ===== Init rtmlib =====
    sys.path.insert(0, args.rtmlib_path)
    from rtmlib import PoseTracker, Wholebody, draw_skeleton

    print(f"Initializing PoseTracker (mode: {args.mode})...")
    wholebody = PoseTracker(
        Wholebody,
        det_frequency=7,
        to_openpose=False,
        mode=args.mode,
        backend="onnxruntime",
        device=args.device,
        return_track_ids=True,
    )

    # ===== Video source =====
    source = int(args.source) if args.source.isdigit() else args.source
    is_camera = isinstance(source, int)
    is_image = (not is_camera and
                any(str(source).lower().endswith(ext)
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']))

    # ===== Image mode =====
    if is_image:
        print(f"Image mode: {source}")
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"Error: Cannot read image: {source}")
            return

        # 이미지 → 동일 프레임을 반복하여 시퀀스 생성
        buf = deque(maxlen=args.buf_size)
        result = wholebody(frame)
        
        # return_track_ids=True이므로 track_ids도 받음
        if len(result) == 3:
            keypoints, scores, track_ids = result
        else:
            keypoints, scores = result
            track_ids = []

        if keypoints is None or len(keypoints) == 0:
            print("No person detected.")
            return

        kp_body, sc_body = extract_skeleton(keypoints, scores, num_joints=num_joints, person_idx=0)
        kp_norm = normalize_frame(kp_body, sc_body)

        # 동일 프레임으로 버퍼 채우기
        for _ in range(args.buf_size):
            buf.append(kp_norm)

        class_name, confidence, probs_dict = predict_tcn(
            model, buf, max_frames, num_classes, device, num_joints=num_joints
        )

        # Draw
        img_show = frame.copy()
        img_show = draw_skeleton(img_show, keypoints, scores,
                                 openpose_skeleton=False, kpt_thr=args.kpt_thr)
        center = (keypoints[0][LEFT_SHOULDER] + keypoints[0][RIGHT_SHOULDER]) / 2.0
        label_pos = (center[0], center[1] - 50)
        draw_label(img_show, label_pos, class_name, confidence,
                   len(buf), args.buf_size)

        print(f"\nResult: {class_name} ({confidence:.1%})")
        for c, p in probs_dict.items():
            print(f"  {c}: {p:.4f}")

        cv2.imshow("TCN Action Recognition", img_show)
        print("\nPress any key to quit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ===== Video / Camera mode =====
    if is_camera:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        return

    if is_camera:
        print("Warming up camera...")
        for _ in range(5):
            cap.read()

    # Track-wise frame buffers and states
    tracks = {}  # tid -> TrackState
    tid_to_detidx = {}  # tid -> detection index in current frame (for depth)

    # Current target
    active_target_tid = None

    frame_count = 0
    fps_list = []
    fail_count = 0
    tcn_ms = 0.0
    TTL_FRAMES = 30  # Remove track if not seen for 30 frames

    print(f"\nStarted! Buffer size: {args.buf_size}, "
          f"Infer every: {args.infer_every} frames")
    print("Press 'q' to quit, 'r' to reset tracks.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            fail_count += 1
            if is_camera and fail_count < 30:
                time.sleep(0.1)
                continue
            break
        fail_count = 0
        frame_count += 1

        t_start = time.time()

        # ---- Pose detection ----
        result = wholebody(frame)
        
        # return_track_ids=True이므로 track_ids 포함
        if len(result) == 3:
            keypoints, scores, track_ids = result
        else:
            # 혹시 모를 호환성
            keypoints, scores = result
            track_ids = list(range(len(keypoints) if keypoints is not None else 0))
        
        t_pose = time.time()

        img_show = frame.copy()
        tid_to_detidx = {}  # reset for this frame

        if keypoints is not None and len(keypoints) > 0:
            # Draw skeleton
            img_show = draw_skeleton(img_show, keypoints, scores,
                                     openpose_skeleton=False, kpt_thr=args.kpt_thr)

            # 각 detection을 track에 업데이트
            for det_idx, tid in enumerate(track_ids):
                tid_to_detidx[tid] = det_idx
                
                # Extract & normalize skeleton
                kp_body, sc_body = extract_skeleton(keypoints, scores, num_joints=num_joints, person_idx=det_idx)
                if kp_body is not None:
                    kp_norm = normalize_frame(kp_body, sc_body)
                    
                    # Track이 없으면 생성
                    if tid not in tracks:
                        tracks[tid] = TrackState(buf_size=args.buf_size)
                    
                    # 버퍼에 추가 및 상태 업데이트
                    tracks[tid].buffer.append(kp_norm)
                    tracks[tid].last_seen_frame = frame_count
                    
                    # bbox area 계산 (depth priority용)
                    kp_px = keypoints[det_idx][:17]  # 17 body joints
                    x_min, x_max = kp_px[:, 0].min(), kp_px[:, 0].max()
                    y_min, y_max = kp_px[:, 1].min(), kp_px[:, 1].max()
                    tracks[tid].bbox_area = (x_max - x_min) * (y_max - y_min)

            # ---- TCN inference (every N frames) ----
            if frame_count % args.infer_every == 0:
                t_tcn_start = time.time()
                
                # 모든 active track에 대해 TCN 추론
                for tid, track in list(tracks.items()):
                    if len(track.buffer) >= 5:
                        track.pred_cls, track.pred_conf, track.pred_probs = predict_tcn(
                            model, track.buffer, max_frames, num_classes, device,
                            num_joints=num_joints
                        )
                
                tcn_ms = (time.time() - t_tcn_start) * 1000

            # ---- hands_up 후보 선택 ----
            hands_up_candidates = []
            for tid, track in tracks.items():
                if track.pred_cls in ["hands_up_single", "hands_up_both"] and track.pred_conf > 0.7:
                    hands_up_candidates.append((tid, track))
            
            # 우선순위: hands_up_both > hands_up_single, bbox 크기 큰 순
            hands_up_candidates.sort(
                key=lambda x: (
                    x[1].pred_cls == "hands_up_both",  # hands_up_both가 True = 1
                    x[1].bbox_area
                ),
                reverse=True
            )
            
            # 최우선 타겟 선택
            if hands_up_candidates:
                active_target_tid = hands_up_candidates[0][0]
            else:
                active_target_tid = None

            # Draw labels for all tracks
            for tid, track in tracks.items():
                if tid in tid_to_detidx:
                    det_idx = tid_to_detidx[tid]
                    center = (keypoints[det_idx][LEFT_SHOULDER] + keypoints[det_idx][RIGHT_SHOULDER]) / 2.0
                    label_pos = (center[0], center[1] - 50)
                    
                    # Active target은 강조 표시
                    if tid == active_target_tid:
                        color = (0, 255, 255)  # cyan - highlight
                        label_text = f"TID:{tid} {track.pred_cls} {track.pred_conf:.0%} [TARGET]"
                    else:
                        label_text = f"TID:{tid} {track.pred_cls} {track.pred_conf:.0%}"
                    
                    color = CLASS_COLORS.get(track.pred_cls, (255, 255, 255))
                    if tid == active_target_tid:
                        color = (0, 255, 255)
                    
                    # Draw box around person if it's the target
                    if tid == active_target_tid and det_idx < len(keypoints):
                        kp_px = keypoints[det_idx][:17]
                        x_min = int(kp_px[:, 0].min())
                        x_max = int(kp_px[:, 0].max())
                        y_min = int(kp_px[:, 1].min())
                        y_max = int(kp_px[:, 1].max())
                        cv2.rectangle(img_show, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    
                    draw_label(img_show, label_pos, track.pred_cls, track.pred_conf,
                               len(track.buffer), args.buf_size)

        # ---- Clean up old tracks ----
        dead_tids = [tid for tid, track in tracks.items() 
                      if frame_count - track.last_seen_frame > TTL_FRAMES]
        for tid in dead_tids:
            del tracks[tid]
            if active_target_tid == tid:
                active_target_tid = None

        t_end = time.time()

        # FPS
        fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        pose_ms = (t_pose - t_start) * 1000
        
        # HUD: 모든 track의 class 확률
        all_probs = {}
        if tracks:
            # 현재 활성 track들의 probs를 취합 (첫 track 기준으로 표시)
            first_track = next(iter(tracks.values()))
            all_probs = first_track.pred_probs
        
        draw_hud(img_show, avg_fps, pose_ms, tcn_ms,
                 sum(len(t.buffer) for t in tracks.values()), args.buf_size * len(tracks), all_probs)

        cv2.imshow("TCN Action Recognition", img_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracks.clear()
            active_target_tid = None
            print("[Reset] All tracks cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
