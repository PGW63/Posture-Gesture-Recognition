import cv2


def draw_detection_overlay(
    frame,
    bbox_xyxy,
    center_xy,
    class_name,
    confidence,
    class_colors,
    track_id=None,
    extra_lines=None,
    font_scale=0.7,
    thickness=2,
):
    color = class_colors.get(class_name, (255, 255, 255))
    frame_h, frame_w = frame.shape[:2]

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if center_xy is not None:
        cx, cy = int(center_xy[0]), int(center_xy[1])
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
    elif bbox_xyxy is not None:
        cx = int((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5)
        cy = int((bbox_xyxy[1] + bbox_xyxy[3]) * 0.5)
    else:
        return

    label_y = cy - 50
    if bbox_xyxy is not None:
        label_y = max(24, int(bbox_xyxy[1]) - 10)
    label_x = min(max(cx, 40), max(40, frame_w - 40))
    label_y = min(max(label_y, 24), max(24, frame_h - 24))

    prefix = f"TID:{track_id}" if track_id is not None else None
    draw_track_label(
        frame,
        (label_x, label_y),
        class_name,
        confidence,
        class_colors,
        prefix=prefix,
        extra_lines=extra_lines,
        font_scale=font_scale,
        thickness=thickness,
    )


def draw_track_label(
    frame,
    position,
    class_name,
    confidence,
    class_colors,
    prefix=None,
    extra_lines=None,
    font_scale=0.8,
    thickness=2,
):
    color = class_colors.get(class_name, (255, 255, 255))
    base = f"{class_name} {confidence:.0%}"
    label = f"{prefix} {base}" if prefix else base

    frame_h, frame_w = frame.shape[:2]
    cx, cy = int(position[0]), int(position[1])
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cx = min(max(cx, tw // 2 + 6), max(tw // 2 + 6, frame_w - tw // 2 - 6))
    cy = min(max(cy, th + 12), max(th + 12, frame_h - 6))
    cv2.rectangle(
        frame,
        (cx - tw // 2 - 5, cy - th - 10),
        (cx + tw // 2 + 5, cy + 5),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (cx - tw // 2, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    if not extra_lines:
        return

    for i, line in enumerate(extra_lines, start=1):
        text_y = min(cy + 20 + (i - 1) * 18, frame_h - 6)
        cv2.putText(
            frame,
            str(line),
            (cx - tw // 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def draw_basic_hud(frame, fps, stage_text, extra_lines=None):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, stage_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y = 85
    if extra_lines:
        for line in extra_lines:
            cv2.putText(frame, str(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 24


def draw_prob_bars(
    frame,
    probs_dict,
    class_colors,
    start_y=90,
    row_h=24,
    bar_w=150,
    label_max_chars=12,
):
    if not probs_dict:
        return

    for i, (cls_name, prob) in enumerate(probs_dict.items()):
        y = start_y + i * row_h
        color = class_colors.get(cls_name, (200, 200, 200))
        cv2.rectangle(frame, (10, y - 12), (10 + bar_w, y + 4), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, y - 12), (10 + int(bar_w * prob), y + 4), color, -1)
        label = f"{cls_name[:label_max_chars]:{label_max_chars}s} {prob:.0%}"
        cv2.putText(frame, label, (10 + bar_w + 5, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
