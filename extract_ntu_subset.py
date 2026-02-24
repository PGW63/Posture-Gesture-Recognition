"""
Extract 5-class subset from NTU120 2D skeleton pkl.

Classes:
    0: idle           — NTU A011(reading), A012(writing), A028(phone_call),
                        A029(playing_phone), A030(typing) 에서 균등 샘플링
    1: waving         — NTU A023 hand_waving
    2: hands_up_single — (placeholder, custom data에서 나중에 병합)
    3: hands_up_both  — NTU A095 capitulate
    4: pointing       — NTU A031 pointing

Usage:
    python extract_ntu_subset.py
    python extract_ntu_subset.py --src /path/to/ntu120_2d.pkl --out ntu_5class.pkl
    python extract_ntu_subset.py --idle_per_class 200
"""

import pickle
import argparse
import numpy as np
from collections import Counter


# NTU original label (0-indexed): action class mapping
# A023 = label 22, A031 = label 30, A095 = label 94
NTU_WAVING = 22       # A023: hand waving
NTU_POINTING = 30     # A031: pointing to something
NTU_CAPITULATE = 94   # A095: capitulate (both hands up)

# Idle source classes (relatively still / seated actions)
IDLE_SOURCES = [10, 11, 27, 28, 29]
# A011: reading, A012: writing, A028: phone call, A029: playing phone, A030: typing

# New label mapping
NEW_LABELS = {
    "idle": 0,
    "waving": 1,
    # "hands_up_single": 2,  # from custom data
    "hands_up_both": 3,
    "pointing": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 5-class subset from NTU120")
    parser.add_argument("--src", type=str, default="/home/gw/Downloads/ntu120_2d.pkl",
                        help="Path to ntu120_2d.pkl")
    parser.add_argument("--out", type=str, default="ntu_5class.pkl",
                        help="Output pkl path")
    parser.add_argument("--idle_per_class", type=int, default=200,
                        help="Number of idle samples from each source class")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading NTU data: {args.src}")
    with open(args.src, "rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]
    split = data["split"]

    # Build frame_dir → index mapping
    frame2idx = {}
    for idx, a in enumerate(annotations):
        frame2idx[a["frame_dir"]] = idx

    # Convert split string lists to index lists
    split_idx = {}
    for name, items in split.items():
        idxs = [frame2idx[s] for s in items if s in frame2idx]
        split_idx[name] = set(idxs)
        print(f"  {name}: {len(idxs)} samples")

    # ===== Collect samples by new class =====
    # Group annotations by original label
    label_to_indices = {}
    for idx, a in enumerate(annotations):
        label = a["label"]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # --- Idle: sample from multiple classes ---
    idle_indices = []
    for src_label in IDLE_SOURCES:
        src_indices = label_to_indices.get(src_label, [])
        n_sample = min(args.idle_per_class, len(src_indices))
        sampled = np.random.choice(src_indices, size=n_sample, replace=False).tolist()
        idle_indices.extend(sampled)
        print(f"  Idle from label {src_label}: {n_sample} samples")
    print(f"  Total idle: {len(idle_indices)}")

    # --- Direct NTU classes ---
    waving_indices = label_to_indices.get(NTU_WAVING, [])
    hands_up_both_indices = label_to_indices.get(NTU_CAPITULATE, [])
    pointing_indices = label_to_indices.get(NTU_POINTING, [])

    print(f"  Waving (A023): {len(waving_indices)} samples")
    print(f"  Hands up both (A095): {len(hands_up_both_indices)} samples")
    print(f"  Pointing (A031): {len(pointing_indices)} samples")

    # ===== Build new annotation list =====
    # Collect all (old_idx, new_label) pairs
    all_pairs = []
    for idx in idle_indices:
        all_pairs.append((idx, NEW_LABELS["idle"]))
    for idx in waving_indices:
        all_pairs.append((idx, NEW_LABELS["waving"]))
    for idx in hands_up_both_indices:
        all_pairs.append((idx, NEW_LABELS["hands_up_both"]))
    for idx in pointing_indices:
        all_pairs.append((idx, NEW_LABELS["pointing"]))

    # Sort by old index for reproducibility
    all_pairs.sort(key=lambda x: x[0])

    # Build new annotations
    old_to_new_idx = {}
    new_annotations = []
    for new_idx, (old_idx, new_label) in enumerate(all_pairs):
        ann = annotations[old_idx].copy()
        ann["label"] = new_label
        ann["original_label"] = annotations[old_idx]["label"]
        new_annotations.append(ann)
        old_to_new_idx[old_idx] = new_idx

    print(f"\nTotal NTU samples: {len(new_annotations)}")

    # ===== Rebuild splits =====
    new_split = {}
    for name, old_idxs in split_idx.items():
        new_idxs = [old_to_new_idx[i] for i in old_idxs if i in old_to_new_idx]
        new_split[name] = new_idxs
        # Count per class
        class_counts = Counter(new_annotations[i]["label"] for i in new_idxs)
        print(f"  {name}: {len(new_idxs)} samples — {dict(sorted(class_counts.items()))}")

    # ===== Save =====
    out = {
        "annotations": new_annotations,
        "split": new_split,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved: {args.out}")
    print(f"Label mapping: {NEW_LABELS}")
    print(f"(Class 2 = hands_up_single will be added from custom data)")


if __name__ == "__main__":
    main()
