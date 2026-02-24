import numpy as np

def normalize_skeletons(skeletons, min_score=0.3):
    """
    skeletons: (N, K, 3)
    return: list of normalized skeletons (each is (K*2,))
    """

    if skeletons is None or len(skeletons) == 0:
        return []

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    normalized_list = []

    for person in skeletons:
        xy = person[:, :2].copy()
        score = person[:, 2]

        # 필수 keypoint confidence 체크
        if (
            score[LEFT_SHOULDER] < min_score
            or score[RIGHT_SHOULDER] < min_score
        ):
            continue

        # center: 어깨 중점
        center = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2.0
        xy -= center

        # scale: 어깨 거리
        scale = np.linalg.norm(
            xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER]
        )

        if scale < 1e-6:
            continue

        xy /= scale

        normalized_list.append(xy.flatten())

    return normalized_list
