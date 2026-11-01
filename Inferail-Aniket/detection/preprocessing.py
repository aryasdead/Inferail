"""InfeRail Image Preprocessing Pipeline.

Owned by: Team 1 (Computer Vision & AI Pipeline Lead)
Description:
    Provides production-grade image validation, illumination enhancement (CLAHE),
    edge-preserving denoising, aspect-ratio preserving letterboxing, normalization,
    and coordinate mapping for railway track defect detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import cv2
import numpy as np
from PIL import Image


def validate_image(image_input: Union[str, Path, np.ndarray, Image.Image, bytes]) -> np.ndarray:
    """Validate and load an input image into a 3-channel RGB uint8 numpy array.

    Args:
        image_input: File path (str or Path), numpy ndarray, PIL Image, or raw image bytes.

    Returns:
        np.ndarray: Loaded image in RGB color space with shape (H, W, 3) and dtype uint8.

    Raises:
        ValueError: If the input cannot be decoded, has invalid dimensions, or is empty.
    """
    if image_input is None:
        raise ValueError("Image input cannot be None.")

    image_rgb: Optional[np.ndarray] = None

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise ValueError(f"Image file does not exist: {path}")
        # Use cv2.imread for fast decoding, fallback to PIL if needed
        bgr = cv2.imread(str(path))
        if bgr is None:
            # Fallback for unicode paths or unusual formats
            try:
                pil_img = Image.open(path)
                image_rgb = np.array(pil_img.convert("RGB"))
            except Exception as e:
                raise ValueError(f"Failed to read image at '{path}': {e}") from e
        else:
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            # Try PIL for bytes
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                image_rgb = np.array(pil_img.convert("RGB"))
            except Exception as e:
                raise ValueError(f"Failed to decode image from raw bytes: {e}") from e
        else:
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"))

    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Provided numpy array is empty.")

        if image_input.ndim == 2:
            # Grayscale -> RGB
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            channels = image_input.shape[2]
            if channels == 1:
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                # Assume already RGB if passed from PIL or pipeline; ensure uint8
                image_rgb = image_input.copy()
            elif channels == 4:
                # RGBA -> RGB
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError(f"Unsupported number of image channels: {channels}")
        else:
            raise ValueError(f"Invalid image array shape: {image_input.shape}")

        if image_rgb.dtype != np.uint8:
            if np.max(image_rgb) <= 1.0:
                image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unsupported input type for image: {type(image_input)}")

    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Loaded image array is empty.")

    h, w = image_rgb.shape[:2]
    if h < 10 or w < 10:
        raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum supported is 10x10.")

    return image_rgb


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance track visibility using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Applies CLAHE exclusively to the Lightness (L) channel in LAB color space to reveal
    cracks in deep shadows without distorting colors or blowing out shiny rail crowns.

    Args:
        image_rgb: 3-channel RGB image (uint8).
        clip_limit: Threshold for contrast limiting (higher = more contrast, 2.0-3.0 recommended).
        tile_grid_size: Size of grid for histogram equalization (default: 8x8).

    Returns:
        np.ndarray: Contrast-enhanced RGB image.
    """
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def denoise_image(
    image_rgb: np.ndarray,
    method: str = "bilateral",
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
    ksize: int = 3,
) -> np.ndarray:
    """Reduce high-frequency texture noise (e.g., ballast gravel) while preserving sharp crack edges.

    Args:
        image_rgb: 3-channel RGB image (uint8).
        method: Denoising technique - 'bilateral' (edge-preserving) or 'gaussian'.
        d: Diameter of each pixel neighborhood (for bilateral filter).
        sigma_color: Filter sigma in the color space (for bilateral filter).
        sigma_space: Filter sigma in the coordinate space (for bilateral filter).
        ksize: Kernel size for gaussian filter (must be odd).

    Returns:
        np.ndarray: Denoised RGB image.
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif method == "gaussian":
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(image_rgb, (k, k), 0)
    else:
        raise ValueError(f"Unknown denoising method '{method}'. Choose 'bilateral' or 'gaussian'.")


DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
HAIRLINE_CRACK_TARGET_SIZE: Tuple[int, int] = (384, 384)
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(
    image_rgb: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = (224, 224),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while preserving aspect ratio for model ingestion.

    Args:
        image_rgb: 3-channel RGB input image (H, W, 3).
        target_size: Desired output shape (target_height, target_width) or int for square.
            Default (224, 224) to start, or (384, 384) if hairline cracks get lost.
        pad_color: RGB fill color for padding borders (default gray 114, 114, 114).
        auto: If True, minimum rectangle padding aligned to stride.
        scale_fill: If True, stretch directly without padding (distorts aspect ratio).
        scale_up: If False, only scale down, do not scale up.
        stride: Stride divisor for padding alignment.

    Returns:
        Tuple containing:
            - letterboxed_image (np.ndarray): Padded image of shape (target_height, target_width, 3).
            - ratio (float): Scaling factor applied to original dimensions.
            - (dw, dh) (Tuple[float, float]): Width and height padding added on each side.
    """
    shape = image_rgb.shape[:2]  # [current_h, current_w]
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_h, target_w = target_size

    # Scale ratio (new / old)
    r = min(target_h / shape[0], target_w / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    # Compute unpadded new dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]  # padding wh

    if auto:
        # minimum rectangle padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (target_w, target_h)
        r = target_w / shape[1]

    dw /= 2  # divide padding into 2 sides (left/right)
    dh /= 2  # divide padding into 2 sides (top/bottom)

    # Resize image to new unpadded dimensions
    if shape[::-1] != new_unpad:
        interp = cv2.INTER_LINEAR if r > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image_rgb, new_unpad, interpolation=interp)
    else:
        resized = image_rgb

    # Compute integer padding borders
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Pad with constant border
    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )

    return letterboxed, r, (dw, dh)


def normalize_and_to_tensor(
    image_rgb: np.ndarray,
    normalize_imagenet: bool = True,
    add_batch_dim: bool = False,
) -> np.ndarray:
    """Normalize pixel intensities (÷255, then ImageNet mean/std) and rearrange to PyTorch shape.

    Batch dimension is added by the DataLoader in training (add_batch_dim=False),
    and only at inference (add_batch_dim=True).

    Args:
        image_rgb: Padded image array (H, W, 3) in uint8 [0, 255] or float.
        normalize_imagenet: If True, applies ImageNet mean/std standardization after dividing by 255.0.
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
            If False, normalizes to [0.0, 1.0] only.
        add_batch_dim: If True, prepends batch dimension: (1, C, H, W).
            Should be False for DataLoader in training, True for single-image inference.

    Returns:
        np.ndarray: Float32 array with shape (3, H, W) or (1, 3, H, W) if add_batch_dim is True.
    """
    img = np.ascontiguousarray(image_rgb, dtype=np.float32)
    # Scale to [0.0, 1.0] if pixel values are uint8 or in [0, 255]
    if img.max() > 1.0 or image_rgb.dtype == np.uint8:
        img = img / 255.0

    if normalize_imagenet:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # Convert HWC to CHW: (3, H, W)
    tensor = img.transpose((2, 0, 1))

    # Add batch dimension only when requested (at inference; DataLoader handles batching in training)
    if add_batch_dim:
        tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor, dtype=np.float32)


def inverse_letterbox_coords(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Map bounding boxes predicted on letterboxed image back to original image coordinates.

    Args:
        boxes: Array of bounding boxes with shape (N, 4) or 1D (4,) in [x_min, y_min, x_max, y_max].
        ratio: Scale ratio applied during letterbox.
        pad: (dw, dh) padding offsets applied during letterbox.
        original_shape: Original image shape (H, W).

    Returns:
        np.ndarray: Bounding boxes mapped to original image dimensions, clipped to boundaries.
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes_arr = np.array(boxes, dtype=np.float32)
    is_1d = (boxes_arr.ndim == 1)
    if is_1d:
        if len(boxes_arr) != 4:
            raise ValueError(f"1D bounding box must have length 4 [x1, y1, x2, y2], got {len(boxes_arr)}")
        boxes_arr = np.expand_dims(boxes_arr, axis=0)

    dw, dh = pad

    # Subtract padding
    boxes_arr[:, [0, 2]] -= dw
    boxes_arr[:, [1, 3]] -= dh

    # Divide by scale ratio
    boxes_arr[:, :4] /= ratio

    # Clip to original image boundaries
    orig_h, orig_w = original_shape[:2]
    boxes_arr[:, [0, 2]] = np.clip(boxes_arr[:, [0, 2]], 0, orig_w)
    boxes_arr[:, [1, 3]] = np.clip(boxes_arr[:, [1, 3]], 0, orig_h)

    return boxes_arr[0] if is_1d else boxes_arr


def preprocess_for_inference(
    image_input: Union[str, Path, np.ndarray, Image.Image, bytes],
    target_size: Union[int, Tuple[int, int]] = (224, 224),
    use_clahe: bool = True,
    use_denoise: bool = True,
    clahe_clip_limit: float = 2.5,
    normalize_imagenet: bool = True,
) -> Dict[str, Union[np.ndarray, float, Tuple[float, float], Tuple[int, int]]]:
    """Complete end-to-end preprocessing pipeline for InfeRail model inference.

    Steps:
        1. Validate & load input into RGB uint8.
        2. Apply CLAHE illumination enhancement for track shadow compensation (kept in pipeline only).
        3. Optionally apply edge-preserving bilateral denoising.
        4. Apply aspect-ratio preserving letterbox padding (224x224 to start, or 384x384 if hairline cracks get lost).
        5. Convert to normalized float32 tensor (÷255, then ImageNet mean/std).
           Batch dimension (1, 3, H, W) is added only at inference.

    Args:
        image_input: File path, numpy array, PIL Image, or bytes.
        target_size: Model input dimensions (default: 224x224 to start, or 384x384 if hairline cracks get lost).
        use_clahe: Enable CLAHE contrast enhancement.
        use_denoise: Enable bilateral denoising.
        clahe_clip_limit: Contrast clip limit for CLAHE.
        normalize_imagenet: If True, standardizes using ImageNet mean/std.

    Returns:
        Dict containing:
            - 'original_rgb': Original loaded RGB image (uint8, H, W, 3).
            - 'enhanced_rgb': Contrast enhanced and denoised image (uint8, H, W, 3).
            - 'letterboxed_rgb': Padded image at target size (uint8, target_h, target_w, 3).
            - 'tensor': Normalized float32 model input with batch dimension added at inference (1, 3, target_h, target_w).
            - 'ratio': Scale factor applied.
            - 'pad': (dw, dh) padding offsets.
            - 'original_shape': Original (H, W).
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # 1. Validate
    original_rgb = validate_image(image_input)
    orig_h, orig_w = original_rgb.shape[:2]

    # 2. Enhance (kept in pipeline only)
    processed = original_rgb.copy()
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=clahe_clip_limit)

    # 3. Denoise
    if use_denoise:
        processed = denoise_image(processed, method="bilateral")

    # 4. Letterbox (default 224x224, or 384x384)
    letterboxed, ratio, pad = letterbox(processed, target_size=target_size)

    # 5. Tensor (÷255, then ImageNet mean/std; batch dimension added only at inference)
    tensor = normalize_and_to_tensor(
        letterboxed,
        normalize_imagenet=normalize_imagenet,
        add_batch_dim=True,
    )

    return {
        "original_rgb": original_rgb,
        "enhanced_rgb": processed,
        "letterboxed_rgb": letterboxed,
        "tensor": tensor,
        "ratio": ratio,
        "pad": pad,
        "original_shape": (orig_h, orig_w),
    }


def compute_edge_density(
    image_rgb: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> float:
    """Compute the edge density of an image using Canny edge detection.

    Edge density is the fraction of pixels that are edges. Cracked surfaces
    exhibit significantly higher edge density than smooth rail surfaces,
    making this a useful supplementary feature for the classifier.

    Args:
        image_rgb: 3-channel RGB image (uint8).
        low_threshold: Canny lower hysteresis threshold.
        high_threshold: Canny upper hysteresis threshold.

    Returns:
        float: Edge density ratio in [0.0, 1.0]. Higher values suggest
            more texture / potential cracks.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return float(np.count_nonzero(edges) / edges.size)


def preprocess_for_classification(
    image_input: Union[str, Path, np.ndarray, Image.Image, bytes],
    target_size: Union[int, Tuple[int, int]] = (224, 224),
    use_clahe: bool = True,
    use_denoise: bool = True,
    clahe_clip_limit: float = 2.5,
    normalize_imagenet: bool = True,
) -> Dict[str, Any]:
    """Preprocess a single image for crack vs non-crack binary classification.

    This is the **primary entry point** for the CrackClassifier model. It wraps
    ``preprocess_for_inference()`` and adds supplementary features that aid
    binary classification:

        - ``edge_density``: Canny edge ratio on the enhanced image — cracked
          surfaces have noticeably higher edge density (~0.08–0.15) than clean
          rail (~0.02–0.05).
        - ``mean_intensity``: Average pixel intensity of the enhanced image,
          useful for detecting overexposed / underexposed frames.

    The output dict contains a ready-to-feed ``'tensor'`` key with shape
    (1, 3, target_h, target_w) — the exact input the CrackClassifier expects.

    Args:
        image_input: File path (str/Path), numpy array, PIL Image, or bytes.
        target_size: Model input resolution (default 224×224).
        use_clahe: Enable CLAHE contrast enhancement (recommended for
            shadow-heavy track images).
        use_denoise: Enable bilateral edge-preserving denoising.
        clahe_clip_limit: Contrast clip limit for CLAHE.
        normalize_imagenet: If True, applies ImageNet standardization.

    Returns:
        Dict containing all keys from ``preprocess_for_inference()`` plus:
            - 'edge_density': float — Canny edge ratio of the enhanced image.
            - 'mean_intensity': float — Mean grayscale intensity of enhanced image.
    """
    result = preprocess_for_inference(
        image_input,
        target_size=target_size,
        use_clahe=use_clahe,
        use_denoise=use_denoise,
        clahe_clip_limit=clahe_clip_limit,
        normalize_imagenet=normalize_imagenet,
    )

    # Compute supplementary features on the enhanced (pre-letterbox) image
    enhanced = result["enhanced_rgb"]
    result["edge_density"] = compute_edge_density(enhanced)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    result["mean_intensity"] = float(gray.mean())

    return result


def get_training_augmentations(img_size: int = 224, is_classifier: bool = True):
    """Build Albumentations augmentation pipeline tailored for railway track defect training.

    Includes flip, brightness/contrast, light blur, and small shift/scale/rotate.
    Duplicate CLAHE has been removed from augmentations and kept strictly in the preprocessing pipeline.

    Args:
        img_size: Input square resolution for training (default 224, or 384 if hairline cracks get lost).
        is_classifier: If True, configures pipeline for image classification (no bbox_params).
            If False, configures bounding box parameters for YOLO defect detection.

    Returns:
        albumentations.Compose: Configured augmentation pipeline.
    """
    import albumentations as A

    transforms = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ],
            p=0.3,
        ),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            fill=(114, 114, 114),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4,
        ),
    ]

    bbox_params = None if is_classifier else A.BboxParams(format="yolo", label_fields=["class_labels"])
    return A.Compose(transforms, bbox_params=bbox_params)


def preprocess_for_training(
    image_input: Union[str, Path, np.ndarray, Image.Image, bytes],
    target_size: Union[int, Tuple[int, int]] = (224, 224),
    use_clahe: bool = True,
    use_denoise: bool = True,
    clahe_clip_limit: float = 2.5,
    normalize_imagenet: bool = True,
    augment: bool = True,
    augmentation_pipeline: Optional[Any] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int]]]:
    """Single-sample preprocessing for PyTorch Dataset / DataLoader ingestion.

    Applies pipeline CLAHE, optional bilateral denoising, training augmentations,
    letterboxing, and ImageNet standardization. Crucially, it does NOT prepend
    a batch dimension (returns shape (3, H, W)), allowing the PyTorch DataLoader
    to batch samples into (B, 3, H, W).

    Args:
        image_input: File path, numpy array, PIL Image, or bytes.
        target_size: Model input resolution (default: 224x224, or 384x384).
        use_clahe: Enable CLAHE contrast enhancement in the pipeline.
        use_denoise: Enable bilateral denoising.
        clahe_clip_limit: Contrast clip limit for CLAHE.
        normalize_imagenet: If True, applies ImageNet mean/std standardization.
        augment: If True, applies training augmentations.
        augmentation_pipeline: Optional custom Albumentations pipeline.

    Returns:
        Dict containing:
            - 'tensor': Normalized float32 array with shape (3, target_h, target_w).
            - 'processed_rgb': Preprocessed and letterboxed RGB image (uint8, target_h, target_w, 3).
            - 'letterboxed_rgb': Alias for processed_rgb (uint8, target_h, target_w, 3).
            - 'enhanced_rgb': Enhanced image before letterboxing (uint8, H, W, 3).
            - 'original_rgb': Original loaded RGB image (uint8, H, W, 3).
            - 'original_shape': Original (H, W).
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # 1. Validate
    original_rgb = validate_image(image_input)
    orig_h, orig_w = original_rgb.shape[:2]

    # 2. Enhance (kept in pipeline only)
    processed = original_rgb.copy()
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=clahe_clip_limit)

    # 3. Denoise
    if use_denoise:
        processed = denoise_image(processed, method="bilateral")

    enhanced_rgb = processed.copy()

    # 4. Augmentation (if training)
    if augment:
        if augmentation_pipeline is None:
            augmentation_pipeline = get_training_augmentations(img_size=target_size[0], is_classifier=True)
        augmented = augmentation_pipeline(image=processed)
        processed = augmented["image"]

    # 5. Letterbox to exact target size (if not already matching)
    if processed.shape[:2] != target_size:
        letterboxed, _, _ = letterbox(processed, target_size=target_size)
    else:
        letterboxed = processed

    # 6. Normalize without batch dimension (added by DataLoader)
    tensor = normalize_and_to_tensor(
        letterboxed,
        normalize_imagenet=normalize_imagenet,
        add_batch_dim=False,
    )

    return {
        "tensor": tensor,
        "processed_rgb": letterboxed,
        "letterboxed_rgb": letterboxed,
        "enhanced_rgb": enhanced_rgb,
        "original_rgb": original_rgb,
        "original_shape": (orig_h, orig_w),
    }


try:
    import torch
    from torch.utils.data import Dataset as PyTorchDataset

    class TrackClassifierDataset(PyTorchDataset):
        """Production PyTorch Dataset for InfeRail Track Defect Classification.

        Integrates with PyTorch DataLoader to batch single-image tensors (3, H, W) into (B, 3, H, W).
        """

        def __init__(
            self,
            image_paths: List[Union[str, Path]],
            labels: List[int],
            target_size: Tuple[int, int] = (224, 224),
            is_training: bool = True,
            use_clahe: bool = True,
            use_denoise: bool = True,
            normalize_imagenet: bool = True,
        ):
            self.image_paths = list(image_paths)
            self.labels = list(labels)
            self.target_size = target_size
            self.is_training = is_training
            self.use_clahe = use_clahe
            self.use_denoise = use_denoise
            self.normalize_imagenet = normalize_imagenet
            self.aug_pipeline = get_training_augmentations(img_size=target_size[0], is_classifier=True) if is_training else None

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            img_path = self.image_paths[idx]
            label = self.labels[idx]

            result = preprocess_for_training(
                img_path,
                target_size=self.target_size,
                use_clahe=self.use_clahe,
                use_denoise=self.use_denoise,
                normalize_imagenet=self.normalize_imagenet,
                augment=self.is_training,
                augmentation_pipeline=self.aug_pipeline,
            )

            # Return (3, H, W) PyTorch Tensor; DataLoader will batch into (B, 3, H, W)
            tensor_torch = torch.from_numpy(result["tensor"]).float()
            return tensor_torch, label

except ImportError:
    TrackClassifierDataset = None  # type: ignore


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="InfeRail Image Preprocessing Pipeline CLI")
    parser.add_argument(
        "--image",
        type=str,
        default="dataset/sample_test_images/user_track_input.jpg",
        help="Path to input railway track image",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs="+",
        default=[224, 224],
        help="Target size as H W (default: 224 224)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save preprocessed output images",
    )
    args = parser.parse_args()

    target = tuple(args.target_size) if len(args.target_size) == 2 else (args.target_size[0], args.target_size[0])
    img_path = Path(args.image)

    if not img_path.is_file():
        # Fallback to any existing sample
        sample_fallback = Path("dataset/sample_test_images/track_sample_clean.jpg")
        if sample_fallback.is_file():
            img_path = sample_fallback

    print(f"[InfeRail Preprocessing] Ingesting: {img_path}")
    print(f"[InfeRail Preprocessing] Target dimensions: {target}")

    res = preprocess_for_inference(img_path, target_size=target)

    print("\n--- Pipeline Result Data Contract ---")
    print(f"Original shape: {res['original_shape']}")
    print(f"Scale ratio: {res['ratio']:.5f}")
    print(f"Padding (dw, dh): {res['pad']}")
    print(f"Enhanced RGB shape: {res['enhanced_rgb'].shape}, dtype: {res['enhanced_rgb'].dtype}")
    print(f"Letterboxed RGB shape: {res['letterboxed_rgb'].shape}, dtype: {res['letterboxed_rgb'].dtype}")
    print(f"Tensor shape: {res['tensor'].shape}, dtype: {res['tensor'].dtype}")
    print(f"Tensor stats: min={res['tensor'].min():.3f}, max={res['tensor'].max():.3f}, mean={res['tensor'].mean():.3f}, std={res['tensor'].std():.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "enhanced.jpg"), cv2.cvtColor(res["enhanced_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "letterboxed.jpg"), cv2.cvtColor(res["letterboxed_rgb"], cv2.COLOR_RGB2BGR))
    print(f"\n[Saved] Enhanced & letterboxed images written to: {out_dir.resolve()}")

