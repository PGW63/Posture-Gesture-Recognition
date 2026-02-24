import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.normalize_skeleton import normalize_skeletons
from utils.skeleton_ops import keep_body_only
from utils.compute_pairwise_distance import extract_feature_from_xy


# Action → 3-class mapping
ACTION_TO_CLASS = {
    "squat": "sitting",
    "sit": "sitting",
    "run": "standing",
    "stretch": "standing",
    "jump": "standing",
    "bendover": "standing",
    "stand": "standing",
    "walk": "standing",
    "lying": "lying",
}

CLASS_TO_IDX = {
    "sitting": 0,
    "standing": 1,
    "lying": 2,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

# Feature dimension: 17 joints × 2 (x,y) + 16 bone distances = 50
FEATURE_DIM = None  # set dynamically after first sample


def _get_label_from_annotation(ann_path):
    """
    JSON annotation에서 첫 번째 person의 action을 읽어 3-class label로 변환.
    Returns label index (int) or None if no valid action found.
    """
    with open(ann_path, "r") as f:
        ann = json.load(f)

    if not ann.get("persons"):
        return None

    person = ann["persons"][0]
    actions = person.get("actions", {})

    for action_name, is_active in actions.items():
        if is_active == 1 and action_name in ACTION_TO_CLASS:
            mapped_class = ACTION_TO_CLASS[action_name]
            return CLASS_TO_IDX[mapped_class]

    return None


def _process_skeleton(person, augment=False):
    """
    단일 person skeleton (133, 3) → feature vector.
    Returns feature (numpy array) or None if normalization fails.
    """
    # Keep body joints only
    person = keep_body_only(person)

    # Extract body 17 joints
    person_body = person[:17]  # (17, 3)

    # Normalize
    norm_result = normalize_skeletons(person_body[np.newaxis])
    if len(norm_result) == 0:
        return None

    # norm_result[0] is (17*2,) = (34,) flattened xy
    normalized_xy = norm_result[0].reshape(17, 2)  # (17, 2)

    # Data augmentation (train time only)
    if augment:
        # Random horizontal flip
        if np.random.random() < 0.5:
            normalized_xy[:, 0] *= -1
            # Swap left/right joint pairs (COCO convention)
            swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for i, j in swap_pairs:
                normalized_xy[[i, j]] = normalized_xy[[j, i]]

        # Add small random noise
        noise = np.random.normal(0, 0.02, normalized_xy.shape)
        normalized_xy = normalized_xy + noise

    # Extract features: flattened xy + bone distances
    feature = extract_feature_from_xy(normalized_xy)  # (34 + 16,) = (50,)

    return feature


class SkeletonDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=False):
        """
        Args:
            root_dir: Learning 프로젝트 루트
            split: "train", "val", or "test"
            augment: True면 data augmentation 적용 (train에만 권장)
        """
        self.root_dir = root_dir
        self.augment = augment
        self.skeleton_dir = os.path.join(root_dir, "data", "skeletons")
        self.annotation_dir = os.path.join(root_dir, "data", "Annotations")
        split_file = os.path.join(root_dir, "data", "ImageSets", f"{split}.txt")

        with open(split_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]

        # Build valid (skeleton_path, label) pairs for lazy augmentation
        # or precomputed (feature, label) for non-augmented
        self.samples = []
        self.label_counts = [0] * NUM_CLASSES
        self.feature_dim = None
        skipped = 0

        for name in names:
            skeleton_path = os.path.join(self.skeleton_dir, f"{name}.npy")
            ann_path = os.path.join(self.annotation_dir, f"{name}.json")

            if not os.path.exists(skeleton_path) or not os.path.exists(ann_path):
                skipped += 1
                continue

            label = _get_label_from_annotation(ann_path)
            if label is None:
                skipped += 1
                continue

            if augment:
                # Store paths for on-the-fly augmentation
                self.samples.append((skeleton_path, label))
                self.label_counts[label] += 1
            else:
                # Precompute features
                skeletons = np.load(skeleton_path)
                if skeletons is None or len(skeletons) == 0:
                    skipped += 1
                    continue

                feature = _process_skeleton(skeletons[0], augment=False)
                if feature is None:
                    skipped += 1
                    continue

                if self.feature_dim is None:
                    self.feature_dim = len(feature)

                self.samples.append((torch.tensor(feature, dtype=torch.float32), label))
                self.label_counts[label] += 1

        if skipped > 0:
            print(f"[{split}] Skipped {skipped} samples (missing file or failed normalization)")
        print(f"[{split}] Loaded {len(self.samples)} samples — "
              f"sitting: {self.label_counts[0]}, "
              f"standing: {self.label_counts[1]}, "
              f"lying: {self.label_counts[2]}")
        if self.feature_dim:
            print(f"[{split}] Feature dim: {self.feature_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.augment:
            skeleton_path, label = self.samples[idx]
            skeletons = np.load(skeleton_path)
            feature = _process_skeleton(skeletons[0], augment=True)
            if feature is None:
                # Fallback: return zeros if augmented version fails
                feature = np.zeros(50, dtype=np.float32)
            return torch.tensor(feature, dtype=torch.float32), label
        else:
            feature, label = self.samples[idx]
            return feature, label

    def get_class_weights(self):
        """역빈도 기반 클래스 가중치 계산."""
        total = sum(self.label_counts)
        weights = []
        for count in self.label_counts:
            if count > 0:
                weights.append(total / (NUM_CLASSES * count))
            else:
                weights.append(1.0)
        return torch.FloatTensor(weights)
