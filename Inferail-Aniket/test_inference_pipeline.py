"""End-to-End Test for InfeRail Crack Detection and Localization Pipeline.

Tests that a raw image is properly loaded, preprocessed, classified,
and has cracks localized with bounding boxes and confidence percentages.
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch

from detection.crack_classifier import (
    CrackClassifier,
    load_model,
    classify_crack,
    classify_image,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    CLASS_NAMES,
)
from detection.preprocessing import (
    validate_image,
    preprocess_for_classification,
    inverse_letterbox_coords,
)
from detection.visualizer import (
    draw_detections,
    create_side_by_side_comparison,
    save_annotated_image,
    CLASS_COLORS,
)

REPO_ROOT = Path(__file__).resolve().parent


def compute_cam(model: CrackClassifier, tensor: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Compute Class Activation Map (CAM) from the final conv layer."""
    model.eval()
    tensor = tensor.to(device)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

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
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def map_cam_to_original(
    cam_7x7: np.ndarray,
    original_shape: tuple,
    ratio: float,
    pad: tuple,
) -> np.ndarray:
    """Map 7x7 CAM back to the original image dimensions, removing letterbox padding."""
    h_orig, w_orig = original_shape
    pad_w, pad_h = pad

    # Resize CAM to 224x224
    cam_224 = cv2.resize(cam_7x7, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Active area inside 224x224
    unpad_w = max(1, int(round(w_orig * ratio)))
    unpad_h = max(1, int(round(h_orig * ratio)))
    pad_x = int(round(pad_w))
    pad_y = int(round(pad_h))

    cam_active = cam_224[pad_y : pad_y + unpad_h, pad_x : pad_x + unpad_w]
    cam_orig = cv2.resize(cam_active, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    return np.clip(cam_orig, 0.0, 1.0)


def extract_crack_bounding_boxes(
    cam_orig: np.ndarray,
    confidence: float,
    min_area_ratio: float = 0.005,
    threshold: float = 0.45,
) -> list:
    """Extract bounding boxes from activation peaks."""
    h, w = cam_orig.shape[:2]
    binary_mask = (cam_orig >= threshold).astype(np.uint8) * 255

    # Clean mask with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    min_area = w * h * min_area_ratio

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Add padding around the crack
            pad = 10
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, by + bh + pad)
            detections.append({
                "class_name": "crack",
                "confidence": round(float(confidence), 4),
                "bbox": [x1, y1, x2, y2],
            })

    # If no contour passed area filter but CAM has strong peak, use the peak region
    if not detections and cam_orig.max() > 0.6:
        peak_y, peak_x = np.unravel_index(np.argmax(cam_orig), cam_orig.shape)
        box_w = max(50, int(w * 0.25))
        box_h = max(50, int(h * 0.25))
        x1 = max(0, peak_x - box_w // 2)
        y1 = max(0, peak_y - box_h // 2)
        x2 = min(w, peak_x + box_w // 2)
        y2 = min(h, peak_y + box_h // 2)
        detections.append({
            "class_name": "crack",
            "confidence": round(float(confidence), 4),
            "bbox": [x1, y1, x2, y2],
        })

    return detections


def inspect_raw_image(
    image_path: Path,
    model: CrackClassifier,
    device: str = "cpu",
    output_dir: Path = None,
) -> dict:
    """End-to-end processing: raw image -> preprocessing -> prediction -> CAM -> highlight boxes."""
    # 1. Preprocess
    pre = preprocess_for_classification(image_path)
    tensor = torch.from_numpy(pre["tensor"]).float()
    orig_rgb = pre["original_rgb"]
    h_orig, w_orig = pre["original_shape"]

    # 2. Predict with confidence
    res = classify_crack(pre, model=model, device=device)
    pred = res["prediction"]
    conf = res["confidence"]
    prob = res["crack_probability"]

    annotated = orig_rgb.copy()
    detections = []

    if pred == "cracked":
        # 3. Compute CAM
        cam_7x7 = compute_cam(model, tensor, device=device)
        cam_orig = map_cam_to_original(cam_7x7, (h_orig, w_orig), pre["ratio"], pre["pad"])

        # 4. Extract bounding boxes
        detections = extract_crack_bounding_boxes(cam_orig, confidence=conf)

        # 5. Overlay subtle heat glow on crack regions
        heatmap_color = cv2.applyColorMap((cam_orig * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        # Blend only where CAM is active
        mask = (cam_orig[:, :, np.newaxis] > 0.3).astype(np.float32)
        annotated = (annotated * (1 - mask * 0.45) + heatmap_rgb * (mask * 0.45)).astype(np.uint8)

        # 6. Draw bounding boxes & badges
        annotated = draw_detections(annotated, detections, show_confidence=True)
    else:
        # Non-cracked: add a clean verification badge at top
        banner_text = f"INTACT RAIL STEEL - CONFIDENCE: {conf * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, banner_text, (20, 40), font, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

    # 7. Side-by-side comparison
    comparison = create_side_by_side_comparison(orig_rgb, annotated, header_height=45)

    # 8. Save if output_dir provided
    saved = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        det_path = output_dir / f"annotated_{stem}.jpg"
        cmp_path = output_dir / f"comparison_{stem}.jpg"
        save_annotated_image(annotated, det_path)
        save_annotated_image(comparison, cmp_path)
        saved = {"annotated": str(det_path), "comparison": str(cmp_path)}

    return {
        "file": Path(image_path).name,
        "prediction": pred,
        "confidence": conf,
        "confidence_pct": f"{conf * 100:.2f}%",
        "crack_probability": prob,
        "detections": detections,
        "saved_files": saved,
    }


def main():
    print("=" * 80)
    print("INFERAIL END-TO-END CRACK DETECTION & VISUALIZATION BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading trained model from: {DEFAULT_MODEL_PATH}")
    model = load_model(DEFAULT_MODEL_PATH, device=device)

    output_dir = REPO_ROOT / "outputs" / "inspection_results"

    # Test cases: raw cracked images, raw uncracked images, and sample test track images
    test_images = [
        # New cracked images
        (REPO_ROOT.parent / "cracked" / "cracked4.jpeg", "cracked"),
        (REPO_ROOT.parent / "cracked" / "cracked5.jpeg", "cracked"),
        (REPO_ROOT.parent / "cracked" / "cracked 1.png", "cracked"),
        # Clean images
        (REPO_ROOT.parent / "uncracked" / "uncrack1.jpeg", "non-cracked"),
        (REPO_ROOT.parent / "uncracked" / "uncrack2.jpeg", "non-cracked"),
        # Sample test images
        (REPO_ROOT / "dataset" / "sample_test_images" / "track_sample_clean.jpg", "non-cracked"),
        (REPO_ROOT / "dataset" / "sample_test_images" / "track_sample_crack_shadow.jpg", "cracked"),
    ]

    results = []
    print(f"{'Image File':<32} | {'Expected':<12} | {'Prediction':<12} | {'Confidence':<10} | {'Boxes':<6} | {'Status'}")
    print("-" * 80)

    for img_path, expected in test_images:
        if not img_path.is_file():
            print(f"Skipping missing image: {img_path}")
            continue
        res = inspect_raw_image(img_path, model=model, device=device, output_dir=output_dir)
        is_correct = res["prediction"] == expected
        match_symbol = "✓ PASS" if is_correct else "✗ FAIL"
        num_boxes = len(res["detections"])
        print(f"{res['file'][:32]:<32} | {expected:<12} | {res['prediction']:<12} | {res['confidence_pct']:<10} | {num_boxes:<6} | {match_symbol}")
        results.append(res)

    print("=" * 80)
    print(f"Annotated images and side-by-side comparisons saved to: {output_dir}")


if __name__ == "__main__":
    main()

