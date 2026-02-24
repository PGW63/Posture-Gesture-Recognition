"""
웹캠/DroidCam으로 스켈레톤 시퀀스를 수집하여 NTU pkl 형식으로 저장.

사용법:
    # 웹캠
    python src/collect_data.py --label hands_up

    # DroidCam (IP 카메라)
    python src/collect_data.py --label hands_up --source http://192.168.0.10:4747/video

    # DroidCam (가상 디바이스)
    python src/collect_data.py --label hands_up --source 2

조작:
    SPACE  — 카운트다운 후 녹화 시작 / 녹화 중지 (토글)
    a      — 자동 녹화 모드 (녹화→대기→녹화 반복, 'a'로 중지)
    s      — 현재까지 수집한 데이터를 pkl로 저장
    q      — 저장하고 종료
    r      — 마지막 녹화 삭제 (undo)

녹화 중 약 3~5초 동작을 반복하면 됩니다.
각 라벨별로 최소 50~100개 샘플 수집 권장.
"""

import sys
import os
import argparse
import time
import pickle
from collections import deque

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Collect skeleton sequences from webcam")
    parser.add_argument("--label", type=str, required=True,
                        help="Action label name (e.g. idle, hands_up, hand_waving, pointing)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "custom_data"),
                        help="Output directory for collected data")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (default: 0)")
    parser.add_argument("--rtmlib_path", type=str,
                        default="/home/gw/robocup_ws/src/rtmlib")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["balanced", "performance", "lightweight"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_frames", type=int, default=90,
                        help="Max frames per clip (default: 90, ~3sec at 30fps)")
    parser.add_argument("--min_frames", type=int, default=15,
                        help="Min frames for valid clip (default: 15)")
    parser.add_argument("--countdown", type=int, default=5,
                        help="Countdown seconds before recording starts (default: 5)")
    parser.add_argument("--auto_pause", type=int, default=3,
                        help="Pause seconds between auto-record clips (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    )

    # ===== Camera =====
    is_url = args.source.startswith("http") or args.source.startswith("rtsp")
    source = int(args.source) if args.source.isdigit() else args.source

    if is_url:
        # DroidCam IP camera (HTTP/RTSP)
        print(f"Connecting to IP camera: {args.source}")
        cap = cv2.VideoCapture(source)
    else:
        # Local webcam (V4L2)
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.source}")
        return

    print("Warming up camera...")
    for _ in range(5):
        cap.read()

    # ===== State =====
    recording = False
    countdown_active = False
    countdown_start = 0
    auto_mode = False
    auto_state = "idle"       # "idle", "countdown", "recording", "pause"
    auto_state_start = 0
    current_clip_kp = []       # list of (17, 2) keypoint arrays
    current_clip_sc = []       # list of (17,) score arrays
    all_clips = []             # list of (keypoint, keypoint_score, total_frames)

    # Load existing data for this label (if any)
    pkl_path = os.path.join(args.output_dir, f"{args.label}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            existing = pickle.load(f)
        all_clips = existing
        print(f"Loaded {len(all_clips)} existing clips for '{args.label}'")
    else:
        print(f"Starting fresh collection for '{args.label}'")

    print(f"\n{'='*50}")
    print(f"  Collecting: {args.label}")
    print(f"  Max frames/clip: {args.clip_frames}")
    print(f"  Countdown: {args.countdown}s")
    print(f"  Existing clips: {len(all_clips)}")
    print(f"{'='*50}")
    print(f"  SPACE = countdown → record (toggle)")
    print(f"  a     = auto-record mode (toggle)")
    print(f"  s     = save to pkl")
    print(f"  r     = undo last clip")
    print(f"  q     = save & quit")
    print(f"{'='*50}\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        now = time.time()
        keypoints, scores = wholebody(frame)
        img_show = frame.copy()
        h, w = img_show.shape[:2]

        person_detected = keypoints is not None and len(keypoints) > 0

        if person_detected:
            img_show = draw_skeleton(img_show, keypoints, scores,
                                     openpose_skeleton=False, kpt_thr=0.4)

        # ===== Countdown handling =====
        if countdown_active:
            elapsed = now - countdown_start
            remaining = args.countdown - elapsed
            if remaining <= 0:
                # Countdown finished → start recording
                countdown_active = False
                recording = True
                current_clip_kp = []
                current_clip_sc = []
                print(f"  ● Recording started (clip #{len(all_clips) + 1})")
            else:
                # Show big countdown number
                count_num = int(remaining) + 1
                cv2.putText(img_show, str(count_num),
                            (w // 2 - 40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8)
                cv2.putText(img_show, "GET READY!",
                            (w // 2 - 120, h // 2 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ===== Auto-record mode state machine =====
        if auto_mode and not countdown_active:
            if auto_state == "idle":
                # Start first countdown
                auto_state = "countdown"
                auto_state_start = now
                countdown_active = True
                countdown_start = now
                print(f"  ▶ Auto-mode: countdown {args.countdown}s...")

            elif auto_state == "countdown":
                # Waiting for countdown (handled above)
                if not countdown_active:
                    auto_state = "recording"
                    auto_state_start = now

            elif auto_state == "recording":
                # Recording is handled below; check if clip was auto-saved
                if not recording:
                    # Clip finished → go to pause
                    auto_state = "pause"
                    auto_state_start = now
                    print(f"  ⏸ Auto-mode: pause {args.auto_pause}s...")

            elif auto_state == "pause":
                elapsed = now - auto_state_start
                if elapsed >= args.auto_pause:
                    # Pause done → start next countdown
                    auto_state = "countdown"
                    auto_state_start = now
                    countdown_active = True
                    countdown_start = now
                    print(f"  ▶ Auto-mode: countdown {args.countdown}s...")

        # ===== Recording =====
        if recording and person_detected:
            kp = keypoints[0][:17].copy()   # (17, 2)
            sc = scores[0][:17].copy()       # (17,)
            current_clip_kp.append(kp)
            current_clip_sc.append(sc)

            # Auto-stop at max frames
            if len(current_clip_kp) >= args.clip_frames:
                _save_clip(current_clip_kp, current_clip_sc, all_clips,
                           args.min_frames, args.label)
                current_clip_kp = []
                current_clip_sc = []
                recording = False

        # ===== UI =====
        # Status bar
        if countdown_active:
            status_color = (0, 255, 255)
            status_text = f"⏱ COUNTDOWN {int(args.countdown - (now - countdown_start)) + 1}s"
        elif recording:
            status_color = (0, 0, 255)
            status_text = f"● REC  {len(current_clip_kp)}/{args.clip_frames}f"
        elif auto_mode and auto_state == "pause":
            remaining_pause = args.auto_pause - (now - auto_state_start)
            status_color = (255, 165, 0)
            status_text = f"⏸ PAUSE {remaining_pause:.1f}s"
        else:
            status_color = (0, 200, 0)
            status_text = "■ IDLE"

        auto_indicator = " [AUTO]" if auto_mode else ""

        cv2.rectangle(img_show, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(img_show, status_text + auto_indicator, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        info = f"Label: {args.label}  |  Clips: {len(all_clips)}"
        cv2.putText(img_show, info, (w - 300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if not person_detected:
            cv2.putText(img_show, "No person detected!",
                        (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(f"Collect: {args.label}", img_show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Toggle recording (with countdown)
            if auto_mode:
                print("  ✗ Auto-mode active. Press 'a' to stop first.")
            elif countdown_active:
                # Cancel countdown
                countdown_active = False
                print("  ✗ Countdown cancelled")
            elif not recording:
                # Start countdown
                countdown_active = True
                countdown_start = now
                print(f"  ⏱ Countdown {args.countdown}s...")
            else:
                # Stop recording → save clip
                _save_clip(current_clip_kp, current_clip_sc, all_clips,
                           args.min_frames, args.label)
                current_clip_kp = []
                current_clip_sc = []
                recording = False

        elif key == ord('a'):  # Toggle auto-record mode
            if auto_mode:
                # Stop auto mode
                auto_mode = False
                auto_state = "idle"
                if countdown_active:
                    countdown_active = False
                if recording:
                    _save_clip(current_clip_kp, current_clip_sc, all_clips,
                               args.min_frames, args.label)
                    current_clip_kp = []
                    current_clip_sc = []
                    recording = False
                print("  ■ Auto-mode stopped")
            else:
                auto_mode = True
                auto_state = "idle"
                print(f"  ▶ Auto-mode started! "
                      f"(record {args.clip_frames}f → pause {args.auto_pause}s → repeat)")

        elif key == ord('s'):  # Save
            _save_pkl(all_clips, pkl_path, args.label)

        elif key == ord('r'):  # Undo last
            if all_clips:
                removed = all_clips.pop()
                print(f"  ↩ Removed last clip ({removed['total_frames']}f). "
                      f"Remaining: {len(all_clips)}")
            else:
                print("  ✗ No clips to remove")

        elif key == ord('q'):  # Quit
            if recording and current_clip_kp:
                _save_clip(current_clip_kp, current_clip_sc, all_clips,
                           args.min_frames, args.label)
            _save_pkl(all_clips, pkl_path, args.label)
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def _save_clip(clip_kp, clip_sc, all_clips, min_frames, label_name):
    """현재 클립을 all_clips에 추가."""
    n = len(clip_kp)
    if n < min_frames:
        print(f"  ✗ Too short ({n}f < {min_frames}f), discarded")
        return

    kp_arr = np.stack(clip_kp, axis=0)   # (T, 17, 2)
    sc_arr = np.stack(clip_sc, axis=0)    # (T, 17)

    # NTU pkl 형식과 동일: (M, T, V, C)
    kp_arr = kp_arr[np.newaxis, ...]           # (1, T, 17, 2)
    sc_arr = sc_arr[np.newaxis, ...]           # (1, T, 17)

    clip = {
        "frame_dir": f"custom_{label_name}_{len(all_clips):04d}",
        "total_frames": n,
        "keypoint": kp_arr.astype(np.float16),
        "keypoint_score": sc_arr.astype(np.float16),
    }
    all_clips.append(clip)
    print(f"  ✓ Clip saved: {n} frames (total: {len(all_clips)} clips)")


def _save_pkl(all_clips, pkl_path, label_name):
    """클립 리스트를 pkl로 저장."""
    if not all_clips:
        print("  ✗ No clips to save")
        return

    with open(pkl_path, "wb") as f:
        pickle.dump(all_clips, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_frames = sum(c["total_frames"] for c in all_clips)
    print(f"\n  ★ Saved {len(all_clips)} clips ({total_frames} total frames)")
    print(f"    → {pkl_path}\n")


if __name__ == "__main__":
    main()
