import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SkeletonDataset, NUM_CLASSES, IDX_TO_CLASS
from models.mlp import MLP


# ===== Hyperparameters =====
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10  # early stopping patience
INPUT_DIM = 50  # 17 body joints × 2 (x, y) + 16 bone distances
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(ROOT_DIR, "models", "best_model.pth")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(features)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def print_confusion_matrix(preds, labels):
    """Print confusion matrix and per-class metrics."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    print("\n===== Confusion Matrix =====")
    header = "True\\Pred".ljust(12) + "".join(IDX_TO_CLASS[i].ljust(12) for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = IDX_TO_CLASS[i].ljust(12) + "".join(str(cm[i][j]).ljust(12) for j in range(NUM_CLASSES))
        print(row)

    print("\n===== Per-Class Metrics =====")
    for i in range(NUM_CLASSES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {IDX_TO_CLASS[i]:10s}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Root dir: {ROOT_DIR}")
    print()

    # ===== Load datasets =====
    print("Loading datasets...")
    train_dataset = SkeletonDataset(ROOT_DIR, split="train", augment=False)
    val_dataset = SkeletonDataset(ROOT_DIR, split="val")
    test_dataset = SkeletonDataset(ROOT_DIR, split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ===== Class weights for imbalanced data =====
    class_weights = train_dataset.get_class_weights().to(DEVICE)
    print(f"\nClass weights: {class_weights.tolist()}")

    # ===== Model =====
    model = MLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"\nModel:\n{model}\n")

    # ===== Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ===== Training loop =====
    best_val_acc = 0.0
    patience_counter = 0

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:3d}/{EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "input_dim": INPUT_DIM,
                "num_classes": NUM_CLASSES,
            }, SAVE_PATH)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ===== Test evaluation =====
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.4f})")

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
    print_confusion_matrix(test_preds, test_labels)

    print(f"\n✓ Training complete! Best model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
