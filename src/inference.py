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
import cv2

from models.mlp import MLP
from data.dataset import NUM_CLASSES, IDX_TO_CLASS
from runtime.mlp_runtime import predict_mlp_frame, predict_mlp_frames, predict_skeleton_file
from utils.track_state import TrackState, TrackManager, match_centers_to_tracks
from runtime.realtime_inference_common import (
    format_track_state_line,
    get_shoulder_center,
    init_pose_tracker,
    open_video_capture,
    parse_video_source,
    update_track_from_keypoints,
    warmup_capture,
)
from runtime.realtime_viz import draw_basic_hud, draw_prob_bars, draw_track_label


INPUT_DIM = 50

# ===== 클래스별 표시 색상 (BGR) =====
CLASS_COLORS = {
    "sitting":  (0, 165, 255),    # orange
    "standing": (0, 255, 0),      # green
    "lying":    (255, 0, 0),      # blue
    "unknown":  (80, 80, 80),     # dark gray
}

def run_webcam(model, device, args):
    """웹캠/비디오 실시간 MLP posture 추론."""

    # rtmlib import
    sys.path.insert(0, args.rtmlib_path)
    from rtmlib import draw_skeleton

    print("Initializing pose tracker...")
    tracker = init_pose_tracker(
        rtmlib_path=args.rtmlib_path,
        mode=args.mode,
        device=args.pose_device,
        backend=args.backend,
        det_frequency=args.det_frequency,
        tracking=False,
        return_track_ids=False,
    )

    # Source
    source, is_camera, is_image = parse_video_source(args.source)

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
            cls_name, conf, probs_dict = predict_mlp_frame(
                model, keypoints[pidx], scores[pidx], device
            )
            center = get_shoulder_center(keypoints[pidx])
            draw_track_label(
                img_show,
                (center[0], center[1] - 60),
                cls_name,
                conf,
                CLASS_COLORS,
                prefix=f"P{pidx}",
                font_scale=0.9,
            )

            print(f"\n  Person {pidx}: {cls_name} ({conf:.1%})")
            for c, p in probs_dict.items():
                print(f"    {c}: {p:.4f}")

        cv2.imshow("MLP Posture Recognition", img_show)
        print("\nPress any key to quit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ===== Video / Camera mode =====
    cap = open_video_capture(source, is_camera)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        return

    if is_camera:
        print("Warming up camera...")
        warmup_capture(cap, num_frames=5)

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
                det_centers.append(get_shoulder_center(keypoints[det_idx]))
            det_to_tid, next_tid = match_centers_to_tracks(
                det_centers, tracks, frame_count, ttl_frames, args.track_match_dist, next_tid
            )

            # MLP inference for multiple persons (batched)
            t_mlp_start = time.time()
            results = predict_mlp_frames(
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
                center = get_shoulder_center(keypoints[pidx])
                kp_px = keypoints[pidx][:17]
                update_track_from_keypoints(tracks[tid], kp_px, center, frame_count)
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
                draw_track_label(
                    img_show,
                    (center[0], center[1] - 60),
                    cls_name,
                    conf,
                    CLASS_COLORS,
                    prefix=f"TID:{tid}",
                    extra_lines=[extra] if extra else None,
                    font_scale=0.9,
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
        draw_basic_hud(img_show, avg_fps, f"Pose: {pose_ms:.0f}ms  MLP: {mlp_ms:.0f}ms")
        draw_prob_bars(img_show, current_probs, CLASS_COLORS, start_y=90, row_h=28, label_max_chars=10)

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
                    print(format_track_state_line(tid, st, include_buffer=False))

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
        results = predict_skeleton_file(model, args.skeleton, device)

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
