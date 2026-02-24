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
        det_frequency=7,
        to_openpose=False,
        mode=mode,
        backend=backend,
        device=device,
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


def draw_label(frame, position, class_name, confidence):
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    label = f"{class_name} {confidence:.0%}"

    cx, cy = int(position[0]), int(position[1])
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (cx - tw // 2 - 5, cy - th - 10),
                  (cx + tw // 2 + 5, cy + 5), color, -1)
    cv2.putText(frame, label, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


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
            draw_label(img_show, (center[0], center[1] - 60), cls_name, conf)

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

    current_class = "unknown"
    current_conf = 0.0
    current_probs = {}
    mlp_ms = 0.0
    frame_count = 0
    fps_list = []
    fail_count = 0

    print(f"\nStarted! Press 'q' to quit.\n")

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

        if keypoints is not None and len(keypoints) > 0:
            img_show = draw_skeleton(img_show, keypoints, scores,
                                     openpose_skeleton=False, kpt_thr=args.kpt_thr)

            # MLP inference for first person
            t_mlp_start = time.time()
            current_class, current_conf, current_probs = predict_frame(
                model, keypoints[0], scores[0], device
            )
            mlp_ms = (time.time() - t_mlp_start) * 1000

            center = (keypoints[0][LEFT_SHOULDER] + keypoints[0][RIGHT_SHOULDER]) / 2.0
            draw_label(img_show, (center[0], center[1] - 60), current_class, current_conf)

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
                        default="/home/gw/robocup_ws/src/rtmlib")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["balanced", "performance", "lightweight"])
    parser.add_argument("--pose_device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="onnxruntime")
    parser.add_argument("--kpt_thr", type=float, default=0.4)

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
