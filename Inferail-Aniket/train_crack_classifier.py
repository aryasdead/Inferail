"""InfeRail Railway Track Crack Classifier Training Pipeline.

Trains the ResNet-18 binary classification model using the cracked and uncracked
railway line images with Albumentations data augmentation and transfer learning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
import os
import random
import shutil
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
    epochs_head: Optional[int] = None,
    epochs_finetune: int = 15,
    batch_size: int = 8,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[CrackClassifier, Dict[str, Any]]:
    """Execute two-phase transfer learning training with validation.
    
    Supports continuing training from an existing checkpoint (warm-start)
    or training from fresh ImageNet pretrained weights.
    """
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

    criterion = nn.BCELoss()
    prior_epochs = 0

    # 1. Initialize CrackClassifier
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        print(f"\nResuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model = CrackClassifier(pretrained=False, freeze_backbone=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        metadata = checkpoint.get("metadata", {})
        prior_epochs = metadata.get("trained_epochs", 0)
        print(f"Model previously trained for {prior_epochs} epochs.")

        # Baseline evaluation before continuing training
        base_loss, base_acc, base_f1, base_cm = evaluate_model(model, val_loader, criterion, device)
        print(f"Pre-training baseline validation: Loss={base_loss:.4f}, Acc={base_acc*100:.1f}%, F1={base_f1*100:.1f}%")

        if epochs_head is None:
            epochs_head = 5  # Quick calibration of head to new data distribution
    else:
        print("\nInitializing fresh model with ImageNet pretrained backbone.")
        model = CrackClassifier(pretrained=True, freeze_backbone=True)
        model.to(device)
        if epochs_head is None:
            epochs_head = 15

    # Freeze backbone for Phase 1
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.classifier_head.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=1e-2,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_weights = None

    print("\n" + "=" * 60)
    print(f"PHASE 1: Training/Calibrating Classification Head ({epochs_head} epochs, Backbone Frozen)")
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

    total_epochs = prior_epochs + epochs_head + epochs_finetune
    metrics = {
        "final_val_loss": round(val_loss, 4),
        "final_val_accuracy": round(val_acc, 4),
        "final_val_f1": round(val_f1, 4),
        "confusion_matrix": cm,
        "history": history,
        "epochs_this_session": epochs_head + epochs_finetune,
        "total_trained_epochs": total_epochs,
        "resumed_from": str(checkpoint_path) if checkpoint_path else None,
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


def test_dataset_images(
    model: CrackClassifier,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Evaluate trained model on all original cracked and uncracked images."""
    model.eval()
    from detection.crack_classifier import classify_image

    eval_dirs = [
        (REPO_ROOT.parent / "cracked", "cracked"),
        (REPO_ROOT.parent / "uncracked", "non-cracked"),
    ]

    results = []
    print("\n" + "=" * 80)
    print("DATASET IMAGES EVALUATION BENCHMARK (CRACKED & UNCRACKED)")
    print("=" * 80)
    print(f"{'Image File':<35} | {'Ground Truth':<12} | {'Prediction':<12} | {'Confidence':<10} | {'Probability':<10}")
    print("-" * 80)

    for folder, expected in eval_dirs:
        if not folder.is_dir():
            continue
        for img_path in sorted(folder.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            res = classify_image(img_path, model=model, device=device)
            pred = res["prediction"]
            conf = res["confidence"]
            prob = res["crack_probability"]
            match_symbol = "✓" if pred == expected else "✗"
            print(f"{img_path.name[:35]:<35} | {expected:<12} | {pred:<12} | {conf*100:>8.2f}% | {prob*100:>8.2f}% {match_symbol}")

            results.append({
                "path": str(img_path),
                "name": img_path.name,
                "ground_truth": expected,
                "prediction": pred,
                "confidence": conf,
                "crack_probability": prob,
                "is_correct": (pred == expected),
            })

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    print("-" * 80)
    print(f"Dataset Benchmark Accuracy: {correct}/{total} ({correct/max(1, total)*100:.1f}%)")
    print("=" * 80)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="InfeRail Crack Classifier Model Training Pipeline")
    parser.add_argument("--epochs-head", type=int, default=None, help="Head calibration epochs (default: 5 if resuming, 15 if from scratch)")
    parser.add_argument("--epochs-finetune", type=int, default=15, help="Fine-tuning epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head learning rate (default: 1e-3)")
    parser.add_argument("--lr-finetune", type=float, default=1e-4, help="Fine-tune learning rate (default: 1e-4)")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (ImageNet backbone) instead of resuming checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (default: models/crack_classifier.pth if present)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("InfeRail Crack Classifier Model Training Pipeline")
    print("=" * 60)

    # Determine checkpoint
    model_save_path = REPO_ROOT / "models" / "crack_classifier.pth"
    resume_path = None
    if not args.scratch:
        if args.checkpoint is not None and Path(args.checkpoint).is_file():
            resume_path = Path(args.checkpoint)
        elif model_save_path.is_file():
            resume_path = model_save_path

    # Back up existing checkpoint if resuming
    if resume_path and resume_path.is_file():
        backup_path = model_save_path.with_name("crack_classifier_prev.pth")
        try:
            shutil.copy2(resume_path, backup_path)
            print(f"Backed up prior checkpoint to: {backup_path.name}")
        except Exception as e:
            print(f"Notice: could not backup checkpoint: {e}")

    # Train model
    model, metrics = train_model(
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        device=device,
        checkpoint_path=resume_path,
    )

    # Save trained checkpoint to models directory
    saved_path = save_model(
        model,
        model_save_path,
        metadata={
            "trained_epochs": metrics.get("total_trained_epochs", 30),
            "architecture": "ResNet-18 (ImageNet fine-tuned)",
            "final_val_loss": metrics["final_val_loss"],
            "final_val_accuracy": metrics["final_val_accuracy"],
            "final_val_f1": metrics["final_val_f1"],
            "training_device": device,
            "resumed_from": str(resume_path) if resume_path else None,
        },
    )
    print(f"\nSuccessfully saved trained checkpoint to: {saved_path}")

    # Evaluate on dataset images & holdout test images
    dataset_results = test_dataset_images(model, device=device)
    holdout_results = test_holdout_images(model, device=device)

    # Save summary report
    report_path = REPO_ROOT / "outputs" / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "metrics": metrics,
        "dataset_results": dataset_results,
        "holdout_results": holdout_results,
        "model_path": str(model_save_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved complete training & evaluation report to: {report_path}")
    print(f"Total training workflow completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()

