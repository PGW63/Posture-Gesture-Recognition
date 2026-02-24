import numpy as np


# ===== RTMPose WholeBody 133 기준 =====

BODY_IDXS = list(range(0, 17))       # 17 joints
FOOT_IDXS = list(range(17, 23))      # 6 joints
FACE_IDXS = list(range(23, 91))      # 68 joints
LEFT_HAND_IDXS = list(range(91, 112))   # 21 joints
RIGHT_HAND_IDXS = list(range(112, 133)) # 21 joints

# ===== Wholebody without face: 65 joints =====
# 133에서 얼굴(23~90) 제거 → body(17) + feet(6) + hands(42) = 65
BODY_HANDS_FEET_IDXS = list(range(0, 23)) + list(range(91, 133))  # 65 indices
NUM_JOINTS_WHOLEBODY = 65  # = len(BODY_HANDS_FEET_IDXS)

# 65-joint 인덱스 매핑 (face 제거 후 재배열):
#   0-16:  Body (COCO 17)
#  17-22:  Feet (left big toe, left small toe, left heel,
#                right big toe, right small toe, right heel)
#  23-43:  Left Hand  (wrist, thumb×4, index×4, middle×4, ring×4, pinky×4)
#  44-64:  Right Hand (wrist, thumb×4, index×4, middle×4, ring×4, pinky×4)

# Left-Right swap pairs for data augmentation (65-joint)
SWAP_PAIRS_17 = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
]
SWAP_PAIRS_65 = (
    SWAP_PAIRS_17                                  # body
    + [(17, 20), (18, 21), (19, 22)]               # feet (left ↔ right)
    + [(23 + i, 44 + i) for i in range(21)]        # left hand ↔ right hand
)


def remove_indices(person, indices):
    """
    person: (K, 3)
    indices: list[int]
    """
    new_person = person.copy()
    new_person[indices] = 0.0
    return new_person


def remove_face(person):
    return remove_indices(person, FACE_IDXS)


def remove_hands(person):
    return remove_indices(person, LEFT_HAND_IDXS + RIGHT_HAND_IDXS)


def remove_feet(person):
    return remove_indices(person, FOOT_IDXS)


def keep_body_only(person):
    """
    얼굴, 손, 발 제거 → body 17개만 사용
    """
    remove = FACE_IDXS + LEFT_HAND_IDXS + RIGHT_HAND_IDXS + FOOT_IDXS
    return remove_indices(person, remove)


def extract_body_hands_feet(person):
    """
    133 keypoints에서 얼굴 제거 → 65 joints (body + feet + hands) 추출.

    person: (133, C)  C=2 or C=3
    Returns: (65, C)
    """
    return person[BODY_HANDS_FEET_IDXS].copy()


def get_swap_pairs(num_joints):
    """num_joints에 맞는 left-right swap pair 반환."""
    if num_joints == 17:
        return SWAP_PAIRS_17
    elif num_joints == 65:
        return SWAP_PAIRS_65
    else:
        # fallback: body-only swap
        return SWAP_PAIRS_17
