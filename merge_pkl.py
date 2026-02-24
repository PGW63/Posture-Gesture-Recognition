"""
Merge NTU subset pkl and custom skeleton pkl into a single 5-class dataset.

Classes:
    0: idle             — NTU (reading, writing, phone_call, playing_phone, typing)
    1: waving           — NTU A023
    2: hands_up_single  — Custom data (한손 들기)
    3: hands_up_both    — NTU A095
    4: pointing         — NTU A031

Usage:
    python merge_pkl.py
    python merge_pkl.py --ntu ntu_5class.pkl --custom custom_handup.pkl --out merged_5class.pkl
"""

import pickle
import argparse
import numpy as np
from collections import Counter


CLASS_NAMES = {
    0: "idle",
    1: "waving",
    2: "hands_up_single",
    3: "hands_up_both",
    4: "pointing",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge NTU and custom pkl")
    parser.add_argument("--ntu", type=str, default="ntu_5class.pkl",
                        help="NTU subset pkl (classes 0,1,3,4)")
    parser.add_argument("--custom", type=str, default="custom_handup.pkl",
                        help="Custom skeleton pkl (class 2)")
    parser.add_argument("--out", type=str, default="merged_5class.pkl",
                        help="Output merged pkl")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load NTU subset
    print(f"Loading NTU subset: {args.ntu}")
    with open(args.ntu, "rb") as f:
        ntu_data = pickle.load(f)

    ntu_ann = ntu_data["annotations"]
    ntu_split = ntu_data["split"]
    n_ntu = len(ntu_ann)
    print(f"  NTU: {n_ntu} samples")

    # Load custom data
    print(f"Loading custom data: {args.custom}")
    with open(args.custom, "rb") as f:
        custom_data = pickle.load(f)

    custom_ann = custom_data["annotations"]
    custom_split = custom_data["split"]
    n_custom = len(custom_ann)
    print(f"  Custom: {n_custom} samples")

    # ===== Merge annotations =====
    # NTU annotations come first (indices 0 .. n_ntu-1)
    # Custom annotations come after (indices n_ntu .. n_ntu+n_custom-1)
    merged_ann = ntu_ann + custom_ann

    # ===== Merge splits =====
    # Custom split indices need to be offset by n_ntu
    merged_split = {}
    for split_name in ntu_split.keys():
        ntu_indices = ntu_split[split_name]
        custom_indices = [i + n_ntu for i in custom_split.get(split_name, [])]
        merged_split[split_name] = ntu_indices + custom_indices

    # ===== Print statistics =====
    print(f"\n{'='*60}")
    print(f"  Merged dataset: {len(merged_ann)} total samples")
    print(f"{'='*60}")

    for split_name, indices in merged_split.items():
        class_counts = Counter(merged_ann[i]["label"] for i in indices)
        print(f"\n  {split_name} ({len(indices)} samples):")
        for cls_id in sorted(class_counts.keys()):
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"    {cls_id}: {cls_name:20s} — {class_counts[cls_id]}")

    # ===== Save =====
    out = {
        "annotations": merged_ann,
        "split": merged_split,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved: {args.out}")
    print(f"Class mapping: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
