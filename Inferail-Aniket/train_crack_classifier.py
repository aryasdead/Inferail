"""InfeRail Railway Track Crack Classifier Training Pipeline.

Trains the ResNet-18 binary classification model using the cracked and uncracked
railway line images with Albumentations data augmentation and transfer learning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path so detection module is discoverable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.crack_classifier import CrackClassifier, save_model, CLASS_NAMES
from detection.preprocessing import (
    preprocess_for_training,
    get_training_augmentations,
    validate_image,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Preparation & Multi-Scale Patch Extraction
# ──────────────────────────────────────────────────────────────────────────────

class RailCrackDataset(Dataset):
    """PyTorch Dataset that generates augmented letterboxed views and track crops.

    For small industrial visual inspection datasets, extracting multi-scale crops
    along the rail track surface combined with Albumentations spatial and lighting
    augmentations prevents overfitting and teaches the network scale-invariant crack
    discontinuity features vs. intact rail steel.
    """

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, str]],
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
    ):
        """Args:
            samples: List of (image_rgb_uint8, label_int, source_name).
            target_size: Target resolution (H, W).
            is_training: If True, applies training augmentations.
        """
        self.samples = samples
        self.target_size = target_size
        self.is_training = is_training
        self.aug = get_training_augmentations(img_size=target_size[0], is_classifier=True) if is_training else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_rgb, label, _ = self.samples[idx]

        # Use InfeRail's preprocessing pipeline: CLAHE + Bilateral Denoise + Letterbox + Normalize
        res = preprocess_for_training(
            img_rgb,
            target_size=self.target_size,
            use_clahe=True,
            use_denoise=True,
            normalize_imagenet=True,
            augment=self.is_training,
            augmentation_pipeline=self.aug,
        )

        tensor = torch.from_numpy(res["tensor"]).float()  # (3, 224, 224)
        target = torch.tensor([float(label)], dtype=torch.float32)  # (1,)
        return tensor, target


def build_dataset_samples(
    repo_base: Path,
) -> Tuple[List[Tuple[np.ndarray, int, str]], List[Tuple[np.ndarray, int, str]]]:
    """Scan and load cracked and uncracked images with patch extraction.

    Returns:
        train_samples, val_samples: Lists of (image_rgb, label, name).
    """
    # Look for images across possible locations
    search_dirs = [
        repo_base.parent / "cracked",
        repo_base.parent / "uncracked",
        repo_base / "dataset" / "sample_test_images",
    ]

    cracked_images: List[Tuple[np.ndarray, str]] = []
    uncracked_images: List[Tuple[np.ndarray, str]] = []

    # 1. Primary cracked directory
    cracked_dir = repo_base.parent / "cracked"
    if cracked_dir.is_dir():
        for p in cracked_dir.glob("*.*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    img = validate_image(p)
                    cracked_images.append((img, p.name))
                except Exception as e:
                    print(f"Warning: could not load {p}: {e}")

    # 2. Primary uncracked directory
    uncracked_dir = repo_base.parent / "uncracked"
    if uncracked_dir.is_dir():
        for p in uncracked_dir.glob("*.*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    img = validate_image(p)
                    uncracked_images.append((img, p.name))
                except Exception as e:
                    print(f"Warning: could not load {p}: {e}")

    # 3. Supplemental sample images
    samples_dir = repo_base / "dataset" / "sample_test_images"
    if samples_dir.is_dir():
        crack_sample = samples_dir / "track_sample_crack_shadow.jpg"
        if crack_sample.is_file():
            cracked_images.append((validate_image(crack_sample), crack_sample.name))

        for clean_name in [
            "track_sample_clean.jpg",
            "track_sample_rust_surface.jpg",
            "track_sample_low_light.jpg",
            "track_sample_specular_glare.jpg",
        ]:
            p = samples_dir / clean_name
            if p.is_file():
                uncracked_images.append((validate_image(p), p.name))

    print(f"Loaded {len(cracked_images)} raw cracked images and {len(uncracked_images)} raw uncracked images.")

    # Multi-crop generator for high-resolution images
    def generate_views(images: List[Tuple[np.ndarray, str]], label: int) -> List[Tuple[np.ndarray, int, str]]:
        samples = []
        for img, name in images:
            # Full image view
            samples.append((img, label, f"{name}_full"))

            h, w = img.shape[:2]
            # If high-resolution, extract multi-scale overlapping crops along the rail surface
            if h >= 400 and w >= 400:
                crop_sizes = [min(h, w) // 2, int(min(h, w) * 0.75)]
                for c_size in crop_sizes:
                    if c_size < 150:
                        continue
                    # Center and offset crops
                    step_y = max(1, (h - c_size) // 3)
                    step_x = max(1, (w - c_size) // 3)
                    for y in range(0, h - c_size + 1, step_y):
                        for x in range(0, w - c_size + 1, step_x):
                            crop = img[y : y + c_size, x : x + c_size]
                            samples.append((crop, label, f"{name}_crop_{y}_{x}"))

        return samples

    all_cracked_views = generate_views(cracked_images, label=1)
    all_uncracked_views = generate_views(uncracked_images, label=0)

    print(f"Generated {len(all_cracked_views)} cracked views and {len(all_uncracked_views)} uncracked views.")

    # Stratified split: 80% train, 20% validation
    random.seed(42)
    random.shuffle(all_cracked_views)
    random.shuffle(all_uncracked_views)

    c_split = max(1, int(len(all_cracked_views) * 0.8))
    u_split = max(1, int(len(all_uncracked_views) * 0.8))

    train_samples = all_cracked_views[:c_split] + all_uncracked_views[:u_split]
    val_samples = all_cracked_views[c_split:] + all_uncracked_views[u_split:]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    epochs_head: int = 15,
    epochs_finetune: int = 15,
    batch_size: int = 8,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: Optional[str] = None,
) -> Tuple[CrackClassifier, Dict[str, Any]]:
    """Execute two-phase transfer learning training with validation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load data
    train_samples, val_samples = build_dataset_samples(REPO_ROOT)
    train_ds = RailCrackDataset(train_samples, is_training=True)
    val_ds = RailCrackDataset(val_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset summary: {len(train_ds)} training samples, {len(val_ds)} validation samples.")

    # 1. Initialize CrackClassifier with ImageNet pretrained backbone
    model = CrackClassifier(pretrained=True, freeze_backbone=True)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer_head = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=1e-2,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_weights = None

    print("\n" + "=" * 60)
    print("PHASE 1: Training Classification Head (Backbone Frozen)")
    print("=" * 60)

    for epoch in range(1, epochs_head + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_head.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer_head.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_ds)

        # Validation
        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs_head:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.1f}% | "
            f"Val F1: {val_f1*100:.1f}%"
        )

    print("\n" + "=" * 60)
    print("PHASE 2: Fine-Tuning High-Level Backbone Layers (layer3, layer4)")
    print("=" * 60)

    # Unfreeze higher convolutional layers (layer3 and layer4) for domain adaptation
    # Keep early conv layers (conv1, layer1, layer2) frozen to retain edge/texture filters
    for i, child in enumerate(model.backbone.children()):
        if i >= 6:  # layer3 (idx 6), layer4 (idx 7), avgpool (idx 8)
            for param in child.parameters():
                param.requires_grad = True

    optimizer_ft = torch.optim.AdamW(
        [
            {"params": model.backbone[6:].parameters(), "lr": lr_finetune * 0.5},
            {"params": model.classifier_head.parameters(), "lr": lr_finetune},
        ],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=epochs_finetune, eta_min=1e-6)

    for epoch in range(1, epochs_finetune + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_ft.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer_ft.step()
            total_loss += loss.item() * len(y)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs_finetune:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.1f}% | "
            f"Val F1: {val_f1*100:.1f}%"
        )

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # Final evaluation
    val_loss, val_acc, val_f1, cm = evaluate_model(model, val_loader, criterion, device)
    print("\n" + "=" * 60)
    print("FINAL VALIDATION REPORT")
    print("=" * 60)
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation F1-Score: {val_f1*100:.2f}%")
    print(f"Confusion Matrix:    TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")

    metrics = {
        "final_val_loss": round(val_loss, 4),
        "final_val_accuracy": round(val_acc, 4),
        "final_val_f1": round(val_f1, 4),
        "confusion_matrix": cm,
        "history": history,
    }

    return model, metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> Tuple[float, float, float, Dict[str, int]]:
    """Compute validation loss, accuracy, F1 score, and confusion matrix."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * len(y)

            predicted_labels = (preds >= threshold).long().squeeze(-1).cpu().tolist()
            if isinstance(predicted_labels, int):
                predicted_labels = [predicted_labels]
            actual_labels = y.long().squeeze(-1).cpu().tolist()
            if isinstance(actual_labels, int):
                actual_labels = [actual_labels]

            all_preds.extend(predicted_labels)
            all_targets.extend(actual_labels)

    avg_loss = total_loss / max(1, len(all_targets))

    # Confusion matrix
    tp = sum(1 for p, a in zip(all_preds, all_targets) if p == 1 and a == 1)
    tn = sum(1 for p, a in zip(all_preds, all_targets) if p == 0 and a == 0)
    fp = sum(1 for p, a in zip(all_preds, all_targets) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(all_preds, all_targets) if p == 0 and a == 1)

    acc = (tp + tn) / max(1, len(all_targets))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-7, precision + recall)

    cm = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return avg_loss, acc, f1, cm


# ──────────────────────────────────────────────────────────────────────────────
# Holdout Evaluation & Testing
# ──────────────────────────────────────────────────────────────────────────────

def test_holdout_images(
    model: CrackClassifier,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run trained model inference on all holdout test images and format predictions."""
    model.eval()
    from detection.crack_classifier import classify_image

    test_dirs = [
        REPO_ROOT.parent / "TEST IMAGES" / "cracked",
        REPO_ROOT.parent / "TEST IMAGES" / "UNCRACKED",
    ]

    results = []
    print("\n" + "=" * 80)
    print("HOLDOUT TEST PREDICTION BENCHMARK")
    print("=" * 80)
    print(f"{'Image File':<45} | {'Ground Truth':<12} | {'Prediction':<12} | {'Confidence':<10} | {'Probability':<10}")
    print("-" * 80)

    for t_dir in test_dirs:
        if not t_dir.is_dir():
            continue
        expected = "non-cracked" if "uncracked" in t_dir.name.lower() else "cracked"
        for img_path in sorted(t_dir.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            res = classify_image(img_path, model=model, device=device)
            pred = res["prediction"]
            conf = res["confidence"]
            prob = res["crack_probability"]

            match_symbol = "✓" if pred == expected else "✗"
            print(f"{img_path.name[:45]:<45} | {expected:<12} | {pred:<12} | {conf*100:>8.2f}% | {prob*100:>8.2f}% {match_symbol}")

            results.append({
                "path": str(img_path),
                "name": img_path.name,
                "ground_truth": expected,
                "prediction": pred,
                "confidence": conf,
                "crack_probability": prob,
                "is_correct": (pred == expected),
            })

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    print("-" * 80)
    print(f"Holdout Benchmark Accuracy: {correct}/{total} ({correct/max(1, total)*100:.1f}%)")
    print("=" * 80)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("InfeRail Crack Classifier Model Training Pipeline")
    print("=" * 60)

    # Train model
    model, metrics = train_model(
        epochs_head=15,
        epochs_finetune=15,
        batch_size=8,
        lr_head=1e-3,
        lr_finetune=1e-4,
        device=device,
    )

    # Save trained checkpoint to models directory
    model_save_path = REPO_ROOT / "models" / "crack_classifier.pth"
    saved_path = save_model(
        model,
        model_save_path,
        metadata={
            "trained_epochs": 30,
            "architecture": "ResNet-18 (ImageNet fine-tuned)",
            "final_val_loss": metrics["final_val_loss"],
            "final_val_accuracy": metrics["final_val_accuracy"],
            "final_val_f1": metrics["final_val_f1"],
            "training_device": device,
        },
    )
    print(f"\nSuccessfully saved trained checkpoint to: {saved_path}")

    # Evaluate on holdout test set
    test_results = test_holdout_images(model, device=device)

    # Save summary report
    report_path = REPO_ROOT / "outputs" / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "metrics": metrics,
        "test_results": test_results,
        "model_path": str(model_save_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Saved complete training & evaluation report to: {report_path}")
    print(f"Total training workflow completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()

