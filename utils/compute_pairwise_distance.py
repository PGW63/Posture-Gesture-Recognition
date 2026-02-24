import numpy as np


# ===== Bone connection pairs (COCO 17 body keypoints) =====
# 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
# 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
# 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
# 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

BONE_PAIRS = [
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (5, 6),               # shoulder width
    (11, 12),             # hip width
    (5, 11), (6, 12),     # torso (shoulder to hip)
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (0, 5), (0, 6),       # nose to shoulders (head tilt)
    (5, 12), (6, 11),     # cross torso (diagonal)
]


# ===== Bone pairs for 65-joint wholebody (face 제거) =====
# 65-joint layout:
#   0-16:  Body (COCO 17)
#  17-22:  Feet (L big toe, L small toe, L heel, R big toe, R small toe, R heel)
#  23-43:  Left Hand  (wrist + 5 fingers × 4 joints)
#  44-64:  Right Hand (wrist + 5 fingers × 4 joints)

BONE_PAIRS_65 = [
    # --- Body core (same as 17-joint) ---
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (5, 6),               # shoulder width
    (11, 12),             # hip width
    (5, 11), (6, 12),     # torso
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (0, 5), (0, 6),       # head tilt
    (5, 12), (6, 11),     # cross torso

    # --- Feet ---
    (15, 19), (15, 17),   # left ankle → heel, big toe
    (16, 22), (16, 20),   # right ankle → heel, big toe
    (19, 17), (19, 18),   # left heel → toes
    (22, 20), (22, 21),   # right heel → toes

    # --- Left Hand (body wrist 9 → hand wrist 23 → fingers) ---
    (9, 23),              # body wrist → hand wrist
    (23, 27),             # wrist → thumb tip
    (23, 31),             # wrist → index tip
    (23, 35),             # wrist → middle tip
    (23, 39),             # wrist → ring tip
    (23, 43),             # wrist → pinky tip

    # --- Right Hand (body wrist 10 → hand wrist 44 → fingers) ---
    (10, 44),             # body wrist → hand wrist
    (44, 48),             # wrist → thumb tip
    (44, 52),             # wrist → index tip
    (44, 56),             # wrist → middle tip
    (44, 60),             # wrist → ring tip
    (44, 64),             # wrist → pinky tip
]



def compute_bone_distances(xy, pairs=BONE_PAIRS):
    """
    xy: (K, 2) normalized skeleton
    return: (len(pairs),)
    """
    dists = []

    for i, j in pairs:
        d = np.linalg.norm(xy[i] - xy[j])
        dists.append(d)

    return np.array(dists)


def compute_all_pairwise_distances(xy):
    """
    xy: (K, 2)
    return: (K*(K-1)//2,)
    """
    K = xy.shape[0]
    dists = []

    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(xy[i] - xy[j])
            dists.append(d)

    return np.array(dists)


def extract_feature_from_xy(xy, use_all_pairs=False, pairs=None):
    """
    xy: normalized (K,2)

    return:
        feature vector = [flattened_xy + pairwise distances]
    """

    coord_feat = xy.flatten()

    if use_all_pairs:
        dist_feat = compute_all_pairwise_distances(xy)
    else:
        # Auto-select bone pairs based on num joints
        if pairs is None:
            K = xy.shape[0]
            if K == 65:
                pairs = BONE_PAIRS_65
            else:
                pairs = BONE_PAIRS
        dist_feat = compute_bone_distances(xy, pairs)

    return np.concatenate([coord_feat, dist_feat])
