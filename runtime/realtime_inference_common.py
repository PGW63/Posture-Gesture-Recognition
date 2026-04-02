import sys
import cv2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def init_pose_tracker(
    rtmlib_path,
    mode,
    device,
    backend="onnxruntime",
    det_frequency=1,
    tracking=False,
    return_track_ids=False,
):
    sys.path.insert(0, rtmlib_path)
    from rtmlib import PoseTracker, Wholebody

    return PoseTracker(
        Wholebody,
        det_frequency=max(1, int(det_frequency)),
        to_openpose=False,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        return_track_ids=return_track_ids,
    )


def parse_video_source(source_arg):
    source = int(source_arg) if str(source_arg).isdigit() else source_arg
    is_camera = isinstance(source, int)
    is_image = (not is_camera and str(source).lower().endswith(IMAGE_EXTENSIONS))
    return source, is_camera, is_image


def open_video_capture(source, is_camera):
    if is_camera:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


def warmup_capture(cap, num_frames=5):
    for _ in range(max(0, int(num_frames))):
        cap.read()


def get_shoulder_center(person_keypoints, left_shoulder=5, right_shoulder=6):
    center = (person_keypoints[left_shoulder] + person_keypoints[right_shoulder]) * 0.5
    return float(center[0]), float(center[1])


def update_track_from_keypoints(
    track,
    keypoints_xy,
    center,
    frame_count,
    keypoint_scores=None,
    score_thr=0.15,
):
    track.last_seen_frame = frame_count
    track.last_center = (float(center[0]), float(center[1]))
    track.missing_frames = 0

    pts = keypoints_xy
    if keypoint_scores is not None and len(keypoint_scores) == len(keypoints_xy):
        valid = keypoint_scores >= float(score_thr)
        if valid.any():
            pts = keypoints_xy[valid]

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    w = float(x_max - x_min)
    h = float(y_max - y_min)

    track.last_bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
    track.bbox_wh = (w, h)
    track.bbox_area = w * h


def format_track_state_line(tid, state, include_buffer=False):
    center = state["last_center"]
    center_str = "None" if center is None else f"({center[0]:.1f},{center[1]:.1f})"
    buf_str = f"buf={state['buffer_len']} " if include_buffer else ""
    return (
        f"  tid={tid} cls={state['pred_cls']} conf={state['pred_conf']:.2f} "
        f"{buf_str}miss={state['missing_frames']} center={center_str}"
    )
