import os
import cv2
import numpy as np
import argparse
from glob import glob
import sys
from tqdm import tqdm


def _initialize(rtmlib_path):
    sys.path.append(rtmlib_path)

    from rtmlib import PoseTracker, Wholebody
    global wholebody

    device = "cuda"
    backend = "onnxruntime"  # opencv, onnxruntime, openvino
    openpose_skeleton = False

    wholebody = PoseTracker(
        Wholebody,
        det_frequency=7,
        to_openpose=openpose_skeleton,
        mode='performance',
        backend=backend,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract skeletons from images and save as numpy arrays."
    )
    parser.add_argument(
        "--rtmlib_path",
        type=str,
        default="/home/gw/robocup_ws/src/rtmlib",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/gw/robocup_ws/src/Learning",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpg",
        help="Format of input images (default: jpg)"
    )
    args = parser.parse_args()

    _initialize(args.rtmlib_path)

    data_path = os.path.join(args.root, "data")
    image_path = os.path.join(data_path, "images")
    skeleton_path = os.path.join(data_path, "skeletons")
    os.makedirs(skeleton_path, exist_ok=True)

    image_list = glob(os.path.join(image_path, f"*.{args.image_format}"))

    total_images = 0
    total_saved = 0
    no_detection = 0

    for img_path in tqdm(image_list, desc="Processing images"):
        total_images += 1

        image = cv2.imread(img_path)
        if image is None:
            tqdm.write(f"[WARNING] Failed to read image: {img_path}")
            continue

        keypoints, scores = wholebody(image)

        # 사람 검출 실패
        if keypoints is None or len(keypoints) == 0:
            no_detection += 1
            tqdm.write(f"[INFO] No person detected: {img_path}")
            continue

        num_people = len(keypoints)
        skeletons = []

        for i in range(num_people):
            person_kp = keypoints[i]          # (num_keypoints, 2)
            person_score = scores[i]          # (num_keypoints,)

            # (num_keypoints, 3) -> x, y, score
            person_skeleton = np.concatenate(
                [person_kp, person_score[..., None]],
                axis=-1
            )

            skeletons.append(person_skeleton)

        skeletons = np.array(skeletons)  # (num_people, num_keypoints, 3)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(skeleton_path, f"{base_name}.npy")

        np.save(save_path, skeletons)

        total_saved += 1
        tqdm.write(f"[OK] Saved {num_people} skeleton(s) from {base_name}")

    print("\n===== SUMMARY =====")
    print(f"Total images processed : {total_images}")
    print(f"Skeleton files saved   : {total_saved}")
    print(f"No detection images    : {no_detection}")
    print("===================")
