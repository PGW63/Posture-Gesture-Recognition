import numpy as np
import torch

from data.dataset import NUM_CLASSES, IDX_TO_CLASS, _process_skeleton


@torch.no_grad()
def predict_skeleton_file(model, skeleton_path, device):
    skeletons = np.load(skeleton_path)  # (N, 133, 3)
    if skeletons.ndim == 2:
        skeletons = skeletons[np.newaxis]

    results = []
    for person_idx, person in enumerate(skeletons):
        feature = _process_skeleton(person, augment=False)
        if feature is None:
            results.append(
                {
                    "person_idx": person_idx,
                    "class": "unknown",
                    "confidence": 0.0,
                    "probabilities": {},
                    "error": "Normalization failed (low confidence keypoints)",
                }
            )
            continue

        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(feature_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        results.append(
            {
                "person_idx": person_idx,
                "class": IDX_TO_CLASS[pred_idx],
                "confidence": confidence,
                "probabilities": {IDX_TO_CLASS[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)},
            }
        )
    return results


@torch.no_grad()
def predict_mlp_frame(model, keypoints_133, scores_133, device):
    person = np.concatenate(
        [keypoints_133.astype(np.float32), scores_133[:, None].astype(np.float32)],
        axis=-1,
    )
    feature = _process_skeleton(person, augment=False)
    if feature is None:
        return "unknown", 0.0, {}

    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(feature_tensor)
    probs = torch.softmax(output, dim=1).squeeze(0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    probs_dict = {IDX_TO_CLASS[i]: probs[i].item() for i in range(NUM_CLASSES)}
    return IDX_TO_CLASS[pred_idx], confidence, probs_dict


@torch.no_grad()
def predict_mlp_frames(model, keypoints, scores, device, max_persons=8):
    if keypoints is None or scores is None:
        return []

    n_det = min(len(keypoints), len(scores), max_persons)
    feats = []
    meta = []
    for pidx in range(n_det):
        person = np.concatenate(
            [keypoints[pidx].astype(np.float32), scores[pidx][:, None].astype(np.float32)],
            axis=-1,
        )
        feature = _process_skeleton(person, augment=False)
        if feature is None:
            meta.append((pidx, None))
            continue
        feats.append(feature)
        meta.append((pidx, len(feats) - 1))

    if not feats:
        return [
            {"person_idx": pidx, "class": "unknown", "confidence": 0.0, "probabilities": {}}
            for pidx in range(n_det)
        ]

    feat_tensor = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32).to(device)
    outputs = model(feat_tensor)
    probs_all = torch.softmax(outputs, dim=1).detach().cpu().numpy()

    results = []
    for pidx, bidx in meta:
        if bidx is None:
            results.append(
                {"person_idx": pidx, "class": "unknown", "confidence": 0.0, "probabilities": {}}
            )
            continue
        probs = probs_all[bidx]
        pred_idx = int(np.argmax(probs))
        results.append(
            {
                "person_idx": pidx,
                "class": IDX_TO_CLASS[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            }
        )
    return results

