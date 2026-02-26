"""
MLP Posture Recognition inference (sitting / standing / lying).

Supports:
    1) .npy skeleton file   → 단일 파일 추론
    2) webcam / video        → 실시간 추론

Usage:
    # .npy 파일
    python src/inference.py skeleton.npy

    # 웹캠 (기본 카메라)
    python src/inference.py --webcam

    # 비디오 파일
    python src/inference.py --webcam --source video.mp4

    # 이미지
    python src/inference.py --webcam --source image.jpg
"""

import sys
import os
import time

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import argparse
import torch
import numpy as np
import cv2

from utils.normalize_skeleton import normalize_skeletons
from utils.skeleton_ops import keep_body_only
from utils.compute_pairwise_distance import extract_feature_from_xy
from models.mlp import MLP
from data.dataset import NUM_CLASSES, IDX_TO_CLASS, _process_skeleton
from utils.track_state import TrackState, TrackManager, match_centers_to_tracks


INPUT_DIM = 50

# ===== 클래스별 표시 색상 (BGR) =====
CLASS_COLORS = {
    "sitting":  (0, 165, 255),    # orange
    "standing": (0, 255, 0),      # green
    "lying":    (255, 0, 0),      # blue
    "unknown":  (80, 80, 80),     # dark gray
}


# ─────────────────────────────────────────────
#  .npy file inference (기존)
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(model, skeleton_path, device):
    """
    단일 .npy 파일에서 skeleton을 로드하여 행동을 예측합니다.

    Args:
        model: 학습된 MLP 모델
        skeleton_path: .npy 파일 경로 (shape: (N_persons, 133, 3))
        device: torch device

    Returns:
        results: list of dicts with 'person_idx', 'class', 'confidence', 'probabilities'
    """
    skeletons = np.load(skeleton_path)  # (N, 133, 3)

    if skeletons.ndim == 2:
        skeletons = skeletons[np.newaxis]  # (1, 133, 3)

    results = []

    for person_idx, person in enumerate(skeletons):
        feature = _process_skeleton(person, augment=False)

        if feature is None:
            results.append({
                "person_idx": person_idx,
                "class": "unknown",
                "confidence": 0.0,
                "probabilities": {},
                "error": "Normalization failed (low confidence keypoints)",
            })
            continue

        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

        output = model(feature_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        results.append({
            "person_idx": person_idx,
            "class": IDX_TO_CLASS[pred_idx],
            "confidence": confidence,
            "probabilities": {IDX_TO_CLASS[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)},
        })

    return results


# ─────────────────────────────────────────────
#  Webcam / video real-time inference
# ─────────────────────────────────────────────

def init_pose_tracker(rtmlib_path, device, backend, mode):
    sys.path.insert(0, rtmlib_path)
    from rtmlib import PoseTracker, Wholebody

    tracker = PoseTracker(
        Wholebody,
        det_frequency=1,
        to_openpose=False,
        mode=mode,
        backend=backend,
        device=device,
        tracking=False,
    )
    return tracker


@torch.no_grad()
def predict_frame(model, keypoints_133, scores_133, device):
    """
    단일 프레임의 rtmlib 133-keypoint → MLP posture 추론.

    Args:
        keypoints_133: (133, 2)
        scores_133:    (133,)

    Returns:
        class_name: str
        confidence: float
        probs_dict: dict {class_name: prob}
    """
    # (133, 2) + (133, 1) → (133, 3) 형태로 결합
    person = np.concatenate([
        keypoints_133.astype(np.float32),
        scores_133[:, None].astype(np.float32),
    ], axis=-1)  # (133, 3)

    feature = _process_skeleton(person, augment=False)
    if feature is None:
        return "unknown", 0.0, {}

    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(feature_tensor)
    probs = torch.softmax(output, dim=1).squeeze(0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    probs_dict = {
        IDX_TO_CLASS[i]: probs[i].item() for i in range(NUM_CLASSES)
    }
    return IDX_TO_CLASS[pred_idx], confidence, probs_dict


@torch.no_grad()
def predict_frames(model, keypoints, scores, device, max_persons=8):
    """
    Multi-person posture inference in one batch.
    Args:
        keypoints: (N,133,2)
        scores:    (N,133)
    Returns:
        results: list[dict] ordered by person index
    """
    if keypoints is None or scores is None:
        return []

    n_det = min(len(keypoints), len(scores), max_persons)
    feats = []
    meta = []
    for pidx in range(n_det):
        person = np.concatenate([
            keypoints[pidx].astype(np.float32),
            scores[pidx][:, None].astype(np.float32),
        ], axis=-1)
        feature = _process_skeleton(person, augment=False)
        if feature is None:
            meta.append((pidx, None))
            continue
        feats.append(feature)
        meta.append((pidx, len(feats) - 1))

    if not feats:
        return [{"person_idx": pidx, "class": "unknown", "confidence": 0.0, "probabilities": {}}
                for pidx in range(n_det)]

    feat_tensor = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32).to(device)
    outputs = model(feat_tensor)  # (B, C)
    probs_all = torch.softmax(outputs, dim=1).detach().cpu().numpy()

    results = []
    for pidx, bidx in meta:
        if bidx is None:
            results.append({
                "person_idx": pidx,
                "class": "unknown",
                "confidence": 0.0,
                "probabilities": {},
            })
            continue
        probs = probs_all[bidx]
        pred_idx = int(np.argmax(probs))
        results.append({
            "person_idx": pidx,
            "class": IDX_TO_CLASS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        })

    return results


def draw_label(frame, position, class_name, confidence, prefix=None, extra_text=None):
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    base = f"{class_name} {confidence:.0%}"
    label = f"{prefix} {base}" if prefix else base

    cx, cy = int(position[0]), int(position[1])
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (cx - tw // 2 - 5, cy - th - 10),
                  (cx + tw // 2 + 5, cy + 5), color, -1)
    cv2.putText(frame, label, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if extra_text:
        cv2.putText(frame, extra_text, (cx - tw // 2, cy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def draw_hud(frame, fps, pose_ms, mlp_ms, probs_dict=None):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Pose: {pose_ms:.0f}ms  MLP: {mlp_ms:.0f}ms",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if probs_dict:
        y_start = 90
        bar_w = 150
        for i, (cls_name, prob) in enumerate(probs_dict.items()):
            y = y_start + i * 28
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(frame, (10, y - 14), (10 + bar_w, y + 6), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, y - 14), (10 + int(bar_w * prob), y + 6), color, -1)
            cv2.putText(frame, f"{cls_name:10s} {prob:.0%}", (10 + bar_w + 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def run_webcam(model, device, args):
    """웹캠/비디오 실시간 MLP posture 추론."""

    # rtmlib import
    sys.path.insert(0, args.rtmlib_path)
    from rtmlib import draw_skeleton

    print("Initializing pose tracker...")
    tracker = init_pose_tracker(
        args.rtmlib_path, args.pose_device, args.backend, args.mode
    )
    # Optional runtime knobs
    try:
        tracker.det_frequency = max(1, int(args.det_frequency))
    except Exception:
        pass

    # Source
    source = int(args.source) if args.source.isdigit() else args.source
    is_camera = isinstance(source, int)
    is_image = (not is_camera and
                any(str(source).lower().endswith(ext)
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']))

    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    # ===== Image mode =====
    if is_image:
        print(f"Image mode: {source}")
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"Error: Cannot read image: {source}")
            return

        keypoints, scores = tracker(frame)
        if keypoints is None or len(keypoints) == 0:
            print("No person detected.")
            return

        img_show = frame.copy()
        img_show = draw_skeleton(img_show, keypoints, scores,
                                 openpose_skeleton=False, kpt_thr=args.kpt_thr)

        for pidx in range(len(keypoints)):
            cls_name, conf, probs_dict = predict_frame(
                model, keypoints[pidx], scores[pidx], device
            )
            center = (keypoints[pidx][LEFT_SHOULDER] + keypoints[pidx][RIGHT_SHOULDER]) / 2.0
            draw_label(img_show, (center[0], center[1] - 60), cls_name, conf, prefix=f"P{pidx}")

            print(f"\n  Person {pidx}: {cls_name} ({conf:.1%})")
            for c, p in probs_dict.items():
                print(f"    {c}: {p:.4f}")

        cv2.imshow("MLP Posture Recognition", img_show)
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

    current_probs = {}
    mlp_ms = 0.0
    frame_count = 0
    next_tid = 0
    fps_list = []
    fail_count = 0
    tracks = TrackManager()
    ttl_frames = max(1, int(args.track_ttl))

    print(f"\nStarted! Press 'q' to quit, 'i' to print track states.\n")

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

        # Pose detection
        keypoints, scores = tracker(frame)
        t_pose = time.time()

        img_show = frame.copy()

        visible_tids = set()
        if keypoints is not None and scores is not None and len(keypoints) > 0:
            img_show = draw_skeleton(img_show, keypoints, scores,
                                     openpose_skeleton=False, kpt_thr=args.kpt_thr)

            # Local track association
            n_det = min(len(keypoints), len(scores), args.max_persons_infer)
            det_centers = []
            for det_idx in range(n_det):
                center = (keypoints[det_idx][LEFT_SHOULDER] + keypoints[det_idx][RIGHT_SHOULDER]) / 2.0
                det_centers.append((float(center[0]), float(center[1])))
            det_to_tid, next_tid = match_centers_to_tracks(
                det_centers, tracks, frame_count, ttl_frames, args.track_match_dist, next_tid
            )

            # MLP inference for multiple persons (batched)
            t_mlp_start = time.time()
            results = predict_frames(
                model, keypoints, scores, device, max_persons=args.max_persons_infer
            )
            mlp_ms = (time.time() - t_mlp_start) * 1000

            # Draw labels for each inferred person
            for r in results:
                pidx = r["person_idx"]
                if pidx >= len(det_to_tid):
                    continue
                tid = int(det_to_tid[pidx])
                visible_tids.add(tid)
                if tid not in tracks:
                    tracks[tid] = TrackState(buf_size=1)
                tracks[tid].pred_cls = r["class"]
                tracks[tid].pred_conf = r["confidence"]
                tracks[tid].pred_probs = r["probabilities"]
                tracks[tid].last_seen_frame = frame_count
                center = (keypoints[pidx][LEFT_SHOULDER] + keypoints[pidx][RIGHT_SHOULDER]) / 2.0
                tracks[tid].last_center = (float(center[0]), float(center[1]))
                tracks[tid].missing_frames = 0
                kp_px = keypoints[pidx][:17]
                x_min, x_max = kp_px[:, 0].min(), kp_px[:, 0].max()
                y_min, y_max = kp_px[:, 1].min(), kp_px[:, 1].max()
                w = float(x_max - x_min)
                h = float(y_max - y_min)
                tracks[tid].last_bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
                tracks[tid].bbox_wh = (w, h)
                tracks[tid].bbox_area = w * h
                st = tracks.get_track_state(tid)
                if st is not None and st.get("last_center") is not None:
                    sx, sy = st["last_center"]
                    extra = f"xy=({sx:.1f},{sy:.1f})"
                    cls_name = st["pred_cls"]
                    conf = st["pred_conf"]
                else:
                    extra = None
                    cls_name = r["class"]
                    conf = r["confidence"]
                draw_label(
                    img_show,
                    (center[0], center[1] - 60),
                    cls_name,
                    conf,
                    prefix=f"TID:{tid}",
                    extra_text=extra,
                )

            # HUD probability bars: highest-confidence visible person
            if results:
                best = max(results, key=lambda x: x["confidence"])
                current_probs = best["probabilities"]
            else:
                current_probs = {}
        else:
            mlp_ms = 0.0

        # Update missing counters and cleanup stale tracks
        for tid, tr in tracks.items():
            if tid not in visible_tids:
                tr.missing_frames = frame_count - tr.last_seen_frame
        dead_tids = [tid for tid, tr in tracks.items() if tr.missing_frames > ttl_frames]
        for tid in dead_tids:
            del tracks[tid]

        t_end = time.time()

        fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        pose_ms = (t_pose - t_start) * 1000
        draw_hud(img_show, avg_fps, pose_ms, mlp_ms, current_probs)

        cv2.imshow("MLP Posture Recognition", img_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            all_states = tracks.get_all_states()
            if not all_states:
                print("[Tracks] none")
            else:
                print("[Tracks]")
                for tid, st in sorted(all_states.items(), key=lambda x: x[0]):
                    center = st["last_center"]
                    center_str = "None" if center is None else f"({center[0]:.1f},{center[1]:.1f})"
                    print(
                        f"  tid={tid} cls={st['pred_cls']} conf={st['pred_conf']:.2f} "
                        f"miss={st['missing_frames']} center={center_str}"
                    )

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MLP Posture Recognition (sitting/standing/lying)"
    )
    parser.add_argument("skeleton", type=str, nargs="?", default=None,
                        help="Path to skeleton .npy file (for file mode)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: models/best_model.pth)")

    # Webcam mode
    parser.add_argument("--webcam", action="store_true",
                        help="Enable webcam/video real-time inference")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: webcam index (0,1,...) or file path")
    parser.add_argument("--rtmlib_path", type=str,
                        default="rtmlib")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["balanced", "performance", "lightweight"])
    parser.add_argument("--pose_device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="onnxruntime")
    parser.add_argument("--kpt_thr", type=float, default=0.4)
    parser.add_argument("--max_persons_infer", type=int, default=8,
                        help="Max persons to run posture inference on per frame")
    parser.add_argument("--det_frequency", type=int, default=1,
                        help="Pose detector frequency for tracker (lower is more stable)")
    parser.add_argument("--track_match_dist", type=float, default=120.0,
                        help="Pixel distance threshold for local track matching")
    parser.add_argument("--track_ttl", type=int, default=20,
                        help="Drop track if not seen for N frames")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or os.path.join(ROOT_DIR, "models", "best_model.pth")

    # Load model
    print(f"Loading MLP model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = MLP(
        input_dim=checkpoint.get("input_dim", INPUT_DIM),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Classes: {NUM_CLASSES} ({', '.join(IDX_TO_CLASS.values())})")

    if args.webcam:
        # ===== Webcam / video real-time mode =====
        run_webcam(model, device, args)
    elif args.skeleton:
        # ===== .npy file mode (기존) =====
        results = predict(model, args.skeleton, device)

        basename = os.path.basename(args.skeleton)
        print(f"\n{'=' * 50}")
        print(f"  Inference: {basename}")
        print(f"{'=' * 50}")

        for r in results:
            print(f"\n  Person {r['person_idx']}:")
            if "error" in r:
                print(f"    ⚠ {r['error']}")
                continue
            print(f"    Prediction : {r['class']} ({r['confidence']:.1%})")
            print(f"    Probabilities:")
            for cls_name, prob in r["probabilities"].items():
                bar = "█" * int(prob * 30)
                print(f"      {cls_name:10s} {prob:.4f} {bar}")

        print()
    else:
        parser.print_help()
        print("\n  Error: --webcam 또는 skeleton .npy 파일 경로를 지정하세요.")


if __name__ == "__main__":
    main()
