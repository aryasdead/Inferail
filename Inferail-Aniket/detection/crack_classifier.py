"""InfeRail Crack Classifier — Binary Image Classification Model.

Owned by: Team 1 (Computer Vision & AI Pipeline Lead)
Description:
    Provides a ResNet-18 based binary classifier for railway track surface
    crack detection. The model consumes preprocessed tensors produced by
    the InfeRail preprocessing pipeline and outputs a prediction
    (cracked / non-cracked) along with a confidence score.

Model Architecture:
    - Backbone: ResNet-18 (torchvision, ImageNet-pretrained weights available)
    - Head: AdaptiveAvgPool → Dropout(0.3) → Linear(512, 1) → Sigmoid
    - Output: Single probability value in [0, 1]
        • ≥ 0.5 → "cracked"
        • < 0.5 → "non-cracked"

Why ResNet-18:
    1. The preprocessing pipeline already applies ImageNet normalization
       (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) which
       aligns perfectly with ResNet pretrained weights.
    2. Lightweight (~11M params) — fast inference on edge devices / drone
       hardware used for track inspections.
    3. Strong transfer learning performance on industrial visual defect
       detection tasks with as few as 500 labeled images.
    4. 224×224 native input resolution matches the default target size
       in the preprocessing pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: Dict[int, str] = {0: "non-cracked", 1: "cracked"}
DEFAULT_THRESHOLD: float = 0.5
MODEL_INPUT_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / 'models' / 'crack_classifier.pth'


# ──────────────────────────────────────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class CrackClassifier(nn.Module):
        """ResNet-18 backbone with a binary classification head for crack detection.

        Architecture
        ────────────
        Input:  (B, 3, 224, 224) — ImageNet-normalized float32 tensor.
        Backbone:  ResNet-18 convolutional layers (conv1 → layer4).
        Head:
            AdaptiveAvgPool2d(1, 1) → Flatten
            → Dropout(p=0.3)
            → Linear(512, 1)
            → Sigmoid
        Output: (B, 1) probability tensor.

        Attributes:
            backbone (nn.Sequential): ResNet-18 feature extractor (all layers
                up to and including the adaptive average pool).
            classifier_head (nn.Sequential): Dropout → Linear(512, 1) → Sigmoid.
            num_classes (int): Always 2 (binary: cracked / non-cracked).
        """

        def __init__(
            self,
            pretrained: bool = True,
            dropout_rate: float = 0.3,
            freeze_backbone: bool = False,
        ):
            """Initialize the CrackClassifier.

            Args:
                pretrained: If True, loads ImageNet-pretrained ResNet-18 weights.
                    Strongly recommended for transfer learning with limited
                    railway track datasets.
                dropout_rate: Dropout probability before the final linear layer.
                    0.3 provides good regularization without starving the
                    classifier of information.
                freeze_backbone: If True, freezes all backbone parameters so only
                    the classification head is trainable. Useful for small
                    datasets (< 1000 images) to prevent overfitting the backbone.
            """
            super().__init__()
            self.num_classes = 2  # binary: cracked vs non-cracked

            # Load ResNet-18 backbone
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                resnet = models.resnet18(weights=weights)
            else:
                resnet = models.resnet18(weights=None)

            # Extract all layers up to (and including) avgpool
            # ResNet-18 structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
            self.backbone = nn.Sequential(
                resnet.conv1,    # (B, 64, 112, 112)
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,  # (B, 64, 56, 56)
                resnet.layer1,   # (B, 64, 56, 56)
                resnet.layer2,   # (B, 128, 28, 28)
                resnet.layer3,   # (B, 256, 14, 14)
                resnet.layer4,   # (B, 512, 7, 7)
                resnet.avgpool,  # (B, 512, 1, 1)
            )

            # Freeze backbone if requested (head-only fine-tuning)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # Custom binary classification head
            backbone_out_features = 512  # ResNet-18 layer4 output channels
            self.classifier_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(backbone_out_features, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass: extract features then classify.

            Args:
                x: Input tensor of shape (B, 3, H, W). Expected to be
                   ImageNet-normalized (the preprocessing pipeline handles this).

            Returns:
                torch.Tensor: Shape (B, 1), probability of class "cracked".
                    Values in [0.0, 1.0].
            """
            features = self.backbone(x)
            return self.classifier_head(features)

        def get_trainable_params(self) -> int:
            """Count trainable parameters (useful for logging before training)."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def get_total_params(self) -> int:
            """Count total parameters."""
            return sum(p.numel() for p in self.parameters())

else:
    # Stub when PyTorch is not installed
    CrackClassifier = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Inference Utilities
# ──────────────────────────────────────────────────────────────────────────────

def classify_crack(
    preprocessed: Dict[str, Any],
    model: Any = None,
    threshold: float = DEFAULT_THRESHOLD,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a preprocessed image as cracked or non-cracked with confidence.

    This function takes the output dictionary from
    ``preprocess_for_classification()`` (or ``preprocess_for_inference()``)
    and runs it through the CrackClassifier model.

    Args:
        preprocessed: Dictionary returned by ``preprocess_for_classification()``
            or ``preprocess_for_inference()``. Must contain key ``'tensor'``
            with shape (1, 3, H, W) float32 numpy array.
        model: An instance of ``CrackClassifier``. If None, a new model with
            pretrained ImageNet weights is instantiated (useful for testing
            the pipeline without trained crack weights).
        threshold: Decision boundary for binary classification.
            probability ≥ threshold → "cracked", else "non-cracked".
        device: PyTorch device string ('cpu', 'cuda', 'cuda:0').
            If None, automatically selects CUDA if available.

    Returns:
        Dict containing:
            - 'prediction': str — "cracked" or "non-cracked".
            - 'confidence': float — Confidence level for the predicted class
                (in [0.0, 1.0]). For "cracked" this is the raw probability;
                for "non-cracked" it's (1 - probability).
            - 'crack_probability': float — Raw model output probability of crack.
            - 'threshold': float — Decision threshold used.
            - 'class_id': int — 1 for cracked, 0 for non-cracked.
            - 'original_shape': Tuple[int, int] — From preprocessing metadata.

    Raises:
        ImportError: If PyTorch is not installed.
        KeyError: If ``preprocessed`` dict is missing the ``'tensor'`` key.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for crack classification. "
            "Install it with: pip install torch torchvision"
        )

    if "tensor" not in preprocessed:
        raise KeyError(
            "Preprocessed dict must contain 'tensor' key. "
            "Use preprocess_for_classification() or preprocess_for_inference()."
        )

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model if not provided
    if model is None:
        if DEFAULT_MODEL_PATH.is_file():
            model = load_model(DEFAULT_MODEL_PATH, device=device)
        else:
            model = CrackClassifier(pretrained=True, freeze_backbone=False)

    model = model.to(device)
    model.eval()

    # Prepare input tensor
    tensor_np = preprocessed["tensor"]
    if isinstance(tensor_np, np.ndarray):
        input_tensor = torch.from_numpy(tensor_np).float()
    else:
        input_tensor = tensor_np.float()

    # Ensure batch dimension: (1, 3, H, W)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # Forward pass (no gradient computation during inference)
    with torch.no_grad():
        crack_probability = model(input_tensor).item()

    # Decision logic
    is_cracked = crack_probability >= threshold
    class_id = 1 if is_cracked else 0
    prediction = CLASS_NAMES[class_id]

    # Confidence: how certain the model is about its chosen class
    confidence = crack_probability if is_cracked else (1.0 - crack_probability)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "crack_probability": round(crack_probability, 4),
        "threshold": threshold,
        "class_id": class_id,
        "original_shape": preprocessed.get("original_shape"),
    }


def classify_image(
    image_input: Any,
    model: Any = None,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    device: Optional[str] = None,
    use_clahe: bool = True,
    use_denoise: bool = True,
) -> Dict[str, Any]:
    """End-to-end convenience: preprocess + classify in a single call.

    Chains ``preprocess_for_classification()`` → ``classify_crack()`` so the
    caller only needs a raw image path/array and gets back a prediction.

    Args:
        image_input: File path (str/Path), numpy array, PIL Image, or bytes.
        model: CrackClassifier instance (if None, uses pretrained ImageNet init).
        target_size: Model input resolution (default 224×224).
        threshold: Decision boundary (default 0.5).
        device: PyTorch device string.
        use_clahe: Enable CLAHE contrast enhancement.
        use_denoise: Enable bilateral denoising.

    Returns:
        Dict with all keys from ``classify_crack()`` plus the full
        preprocessing metadata (original_rgb, enhanced_rgb, letterboxed_rgb,
        tensor, ratio, pad, original_shape, edge_density).
    """
    # Import here to avoid circular dependency
    from detection.preprocessing import preprocess_for_classification

    preprocessed = preprocess_for_classification(
        image_input,
        target_size=target_size,
        use_clahe=use_clahe,
        use_denoise=use_denoise,
    )

    result = classify_crack(
        preprocessed,
        model=model,
        threshold=threshold,
        device=device,
    )

    # Merge preprocessing metadata into the result for downstream consumers
    # (e.g., visualizer needs original_rgb, letterboxed_rgb, etc.)
    result["preprocessing"] = {
        k: v for k, v in preprocessed.items() if k != "tensor"
    }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Model Persistence Helpers (for future training workflow)
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model: Any, path: Union[str, Path], metadata: Optional[Dict] = None) -> str:
    """Save model weights and optional metadata to disk.

    Args:
        model: CrackClassifier instance.
        path: Destination file path (e.g., 'models/crack_classifier_v1.pth').
        metadata: Optional dict of training metadata (epoch, accuracy, etc.).

    Returns:
        str: Absolute path of the saved checkpoint.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to save model weights.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "CrackClassifier",
        "num_classes": 2,
        "class_names": CLASS_NAMES,
        "input_size": MODEL_INPUT_SIZE,
    }
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, str(path))
    return str(path.resolve())


def load_model(
    path: Union[str, Path],
    device: Optional[str] = None,
) -> Any:
    """Load a saved CrackClassifier checkpoint from disk.

    Args:
        path: Path to the saved .pth checkpoint file.
        device: Target device ('cpu' or 'cuda'). Auto-detects if None.

    Returns:
        CrackClassifier: Model instance with loaded weights in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ImportError: If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load model weights.")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(str(path), map_location=device)

    model = CrackClassifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Class Activation Mapping (CAM) & Crack Localization
# ──────────────────────────────────────────────────────────────────────────────

def compute_cam(
    model: Any,
    tensor: Any,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute 7x7 Class Activation Map (CAM) from the final conv layer.

    Args:
        model: CrackClassifier instance.
        tensor: Normalized PyTorch tensor of shape (1, 3, 224, 224) or (3, 224, 224).
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        np.ndarray: 2D array of shape (7, 7) normalized in [0.0, 1.0].
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to compute CAM.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        x = tensor
        for layer in model.backbone[:8]:
            x = layer(x)
        # x is (1, 512, 7, 7)
        weights = model.classifier_head[2].weight.data  # (1, 512)
        feat = x[0]  # (512, 7, 7)
        cam = torch.zeros(feat.shape[1:], dtype=torch.float32, device=device)
        for k in range(512):
            cam += weights[0, k] * feat[k]

        cam = torch.relu(cam).detach().cpu().numpy()
        c_min, c_max = float(cam.min()), float(cam.max())
        cam = (cam - c_min) / (c_max - c_min + 1e-8)
        return cam


def detect_and_visualize_cracks(
    image_input: Any,
    model: Any = None,
    threshold: float = DEFAULT_THRESHOLD,
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    overlay_heatmap: bool = True,
    generate_comparison: bool = True,
) -> Dict[str, Any]:
    """End-to-end pipeline: raw image -> preprocess -> classify -> localize cracks with confidence badges.

    Args:
        image_input: File path (str/Path), numpy array, PIL Image, or bytes.
        model: CrackClassifier instance (loads trained weights from disk if None).
        threshold: Decision boundary for crack classification (default 0.5).
        device: Target device string.
        output_dir: Optional directory to save annotated images and comparisons.
        overlay_heatmap: Whether to overlay CAM activation heatmap on crack regions.
        generate_comparison: Whether to build a side-by-side comparison canvas.

    Returns:
        Dict containing:
            - 'prediction': str ('cracked' or 'non-cracked')
            - 'confidence': float in [0.0, 1.0]
            - 'confidence_pct': str (e.g. '100.0%')
            - 'crack_probability': float
            - 'detections': List of bounding box dicts
            - 'annotated_rgb': np.ndarray of the annotated image
            - 'comparison_canvas': np.ndarray (if generate_comparison=True)
            - 'saved_files': Dict of saved paths (if output_dir provided)
    """
    from detection.preprocessing import preprocess_for_classification
    from detection.visualizer import (
        draw_detections,
        create_side_by_side_comparison,
        save_annotated_image,
    )

    if device is None:
        device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    if model is None:
        if DEFAULT_MODEL_PATH.is_file():
            model = load_model(DEFAULT_MODEL_PATH, device=device)
        else:
            model = CrackClassifier(pretrained=True, freeze_backbone=False)

    # 1. Preprocess raw input
    pre = preprocess_for_classification(image_input)
    tensor = torch.from_numpy(pre["tensor"]).float()
    orig_rgb = pre["original_rgb"]
    h_orig, w_orig = pre["original_shape"]

    # 2. Classify
    res = classify_crack(pre, model=model, threshold=threshold, device=device)
    pred = res["prediction"]
    conf = res["confidence"]
    prob = res["crack_probability"]

    annotated = orig_rgb.copy()
    detections: List[Dict[str, Any]] = []
    cam_orig: Optional[np.ndarray] = None

    if pred == "cracked":
        # 3. Compute CAM & map back to original dimensions
        cam_7x7 = compute_cam(model, tensor, device=device)
        cam_224 = cv2.resize(cam_7x7, (224, 224), interpolation=cv2.INTER_CUBIC)

        unpad_w = max(1, int(round(w_orig * pre["ratio"])))
        unpad_h = max(1, int(round(h_orig * pre["ratio"])))
        pad_x = int(round(pre["pad"][0]))
        pad_y = int(round(pre["pad"][1]))

        cam_active = cam_224[pad_y : pad_y + unpad_h, pad_x : pad_x + unpad_w]
        cam_orig = cv2.resize(cam_active, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        cam_orig = np.clip(cam_orig, 0.0, 1.0)

        # 4. Extract bounding boxes from activation contours
        binary_mask = (cam_orig >= 0.45).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = w_orig * h_orig * 0.005
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                pad_b = 10
                detections.append({
                    "class_name": "crack",
                    "confidence": round(float(conf), 4),
                    "bbox": [
                        max(0, bx - pad_b),
                        max(0, by - pad_b),
                        min(w_orig, bx + bw + pad_b),
                        min(h_orig, by + bh + pad_b),
                    ],
                })

        # Fallback to peak region if no large contour
        if not detections and cam_orig.max() > 0.6:
            py, px = np.unravel_index(np.argmax(cam_orig), cam_orig.shape)
            bw, bh = max(60, int(w_orig * 0.25)), max(60, int(h_orig * 0.25))
            detections.append({
                "class_name": "crack",
                "confidence": round(float(conf), 4),
                "bbox": [
                    max(0, int(px - bw // 2)),
                    max(0, int(py - bh // 2)),
                    min(w_orig, int(px + bw // 2)),
                    min(h_orig, int(py + bh // 2)),
                ],
            })

        # 5. Overlay heat glow
        if overlay_heatmap:
            heatmap_color = cv2.applyColorMap((cam_orig * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            mask_heat = (cam_orig[:, :, np.newaxis] > 0.3).astype(np.float32)
            annotated = (annotated * (1 - mask_heat * 0.45) + heatmap_rgb * (mask_heat * 0.45)).astype(np.uint8)

        # 6. Draw bounding boxes & badges
        annotated = draw_detections(annotated, detections, show_confidence=True)
    else:
        # Non-cracked verification badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        badge_text = f"INTACT RAIL STEEL - CONFIDENCE: {conf * 100:.1f}%"
        cv2.putText(annotated, badge_text, (20, 45), font, 0.75, (0, 255, 128), 2, cv2.LINE_AA)

    # 7. Comparison canvas
    comparison_canvas = None
    if generate_comparison:
        comparison_canvas = create_side_by_side_comparison(orig_rgb, annotated, header_height=45)

    # 8. Save if output directory specified
    saved_files = {}
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        stem = Path(image_input).stem if isinstance(image_input, (str, Path)) else "inspection_result"
        det_path = out_p / f"annotated_{stem}.jpg"
        save_annotated_image(annotated, det_path)
        saved_files["annotated"] = str(det_path)

        if comparison_canvas is not None:
            cmp_path = out_p / f"comparison_{stem}.jpg"
            save_annotated_image(comparison_canvas, cmp_path)
            saved_files["comparison"] = str(cmp_path)

    return {
        "prediction": pred,
        "confidence": conf,
        "confidence_pct": f"{conf * 100:.2f}%",
        "crack_probability": prob,
        "detections": detections,
        "annotated_rgb": annotated,
        "comparison_canvas": comparison_canvas,
        "cam_heatmap": cam_orig,
        "saved_files": saved_files,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model Summary
# ──────────────────────────────────────────────────────────────────────────────

def get_model_summary() -> Dict[str, Any]:
    """Return a human-readable summary of the CrackClassifier architecture.

    Returns:
        Dict with architecture details, parameter counts, and expected I/O shapes.
    """
    summary = {
        "model_name": "CrackClassifier",
        "backbone": "ResNet-18 (torchvision)",
        "pretrained_on": "ImageNet-1K (IMAGENET1K_V1)",
        "task": "Binary Classification (cracked vs non-cracked)",
        "input_shape": "(B, 3, 224, 224)",
        "input_normalization": "ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]",
        "output_shape": "(B, 1)",
        "output_activation": "Sigmoid -> probability in [0, 1]",
        "classification_head": [
            "AdaptiveAvgPool2d(1, 1)",
            "Flatten",
            "Dropout(p=0.3)",
            "Linear(512, 1)",
            "Sigmoid",
        ],
        "class_names": CLASS_NAMES,
        "default_threshold": DEFAULT_THRESHOLD,
    }

    if TORCH_AVAILABLE:
        model = CrackClassifier(pretrained=False)
        summary["total_parameters"] = model.get_total_params()
        summary["trainable_parameters"] = model.get_trainable_params()

        # With frozen backbone
        model_frozen = CrackClassifier(pretrained=False, freeze_backbone=True)
        summary["trainable_parameters_frozen_backbone"] = model_frozen.get_trainable_params()

    return summary
