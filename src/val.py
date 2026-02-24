import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import SkeletonDataset, NUM_CLASSES, IDX_TO_CLASS
from models.mlp import MLP


INPUT_DIM = 50
BATCH_SIZE = 64


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def print_report(preds, labels, accuracy, split_name):
    print(f"\n{'=' * 50}")
    print(f"  {split_name} Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(labels))}/{len(labels)})")

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    print(f"\n  Confusion Matrix:")
    header = "  " + "True\\Pred".ljust(12) + "".join(IDX_TO_CLASS[i].ljust(12) for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = "  " + IDX_TO_CLASS[i].ljust(12) + "".join(str(cm[i][j]).ljust(12) for j in range(NUM_CLASSES))
        print(row)

    print(f"\n  Per-Class Metrics:")
    for i in range(NUM_CLASSES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"    {IDX_TO_CLASS[i]:10s}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model on val/test split")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: val)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: models/best_model.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or os.path.join(ROOT_DIR, "models", "best_model.pth")

    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = MLP(
        input_dim=checkpoint.get("input_dim", INPUT_DIM),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Trained at epoch {checkpoint.get('epoch', '?')} (Val Acc: {checkpoint.get('val_acc', '?'):.4f})")

    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = SkeletonDataset(ROOT_DIR, split=args.split)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Evaluate
    accuracy, preds, labels = evaluate(model, loader, device)
    print_report(preds, labels, accuracy, args.split.upper())


if __name__ == "__main__":
    main()
