"""InfeRail Image Preprocessing Pipeline.

Owned by: Team 1 (Computer Vision & AI Pipeline Lead)
Description:
    Provides production-grade image validation, illumination enhancement (CLAHE),
    edge-preserving denoising, aspect-ratio preserving letterboxing, normalization,
    and coordinate mapping for railway track defect detection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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
                image_rgb = (image_rgb * 255.0).astype(np.uint8)
            else:
                image_rgb = image_rgb.astype(np.uint8)

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


def letterbox(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while preserving aspect ratio for YOLO model ingestion.

    Args:
        image_rgb: 3-channel RGB input image (H, W, 3).
        target_size: Desired output shape (target_height, target_width). Default (640, 640).
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


def normalize_and_to_tensor(image_rgb: np.ndarray) -> np.ndarray:
    """Normalize pixel intensities to [0.0, 1.0] and rearrange to PyTorch shape (1, C, H, W).

    Args:
        image_rgb: Padded image array (H, W, 3) in uint8 [0, 255].

    Returns:
        np.ndarray: Float32 array with shape (1, 3, H, W) in range [0.0, 1.0].
    """
    # Convert HWC to CHW
    img = image_rgb.transpose((2, 0, 1))
    # Add batch dimension: (1, C, H, W)
    img = np.expand_dims(img, axis=0)
    # Normalize to [0.0, 1.0] float32
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img


def inverse_letterbox_coords(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Map bounding boxes predicted on letterboxed 640x640 back to original image coordinates.

    Args:
        boxes: Array of bounding boxes with shape (N, 4) in [x_min, y_min, x_max, y_max].
        ratio: Scale ratio applied during letterbox.
        pad: (dw, dh) padding offsets applied during letterbox.
        original_shape: Original image shape (H, W).

    Returns:
        np.ndarray: Bounding boxes mapped to original image dimensions, clipped to boundaries.
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = np.array(boxes, dtype=np.float32)
    dw, dh = pad

    # Subtract padding
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh

    # Divide by scale ratio
    boxes[:, :4] /= ratio

    # Clip to original image boundaries
    orig_h, orig_w = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes


def preprocess_for_inference(
    image_input: Union[str, Path, np.ndarray, Image.Image, bytes],
    target_size: Tuple[int, int] = (640, 640),
    use_clahe: bool = True,
    use_denoise: bool = True,
    clahe_clip_limit: float = 2.5,
) -> Dict[str, Union[np.ndarray, float, Tuple[float, float], Tuple[int, int]]]:
    """Complete end-to-end preprocessing pipeline for InfeRail model inference.

    Steps:
        1. Validate & load input into RGB uint8.
        2. Optionally apply CLAHE contrast enhancement for track shadow compensation.
        3. Optionally apply edge-preserving bilateral denoising.
        4. Apply aspect-ratio preserving letterbox padding to 640x640.
        5. Convert to normalized float32 tensor (1, 3, H, W).

    Args:
        image_input: File path, numpy array, PIL Image, or bytes.
        target_size: Model input dimensions (default: 640x640).
        use_clahe: Enable CLAHE contrast enhancement.
        use_denoise: Enable bilateral denoising.
        clahe_clip_limit: Contrast clip limit for CLAHE.

    Returns:
        Dict containing:
            - 'original_rgb': Original loaded RGB image (uint8, H, W, 3).
            - 'enhanced_rgb': Contrast enhanced and denoised image (uint8, H, W, 3).
            - 'letterboxed_rgb': Padded image at target size (uint8, target_h, target_w, 3).
            - 'tensor': Normalized float32 model input (1, 3, target_h, target_w).
            - 'ratio': Scale factor applied.
            - 'pad': (dw, dh) padding offsets.
            - 'original_shape': Original (H, W).
    """
    # 1. Validate
    original_rgb = validate_image(image_input)
    orig_h, orig_w = original_rgb.shape[:2]

    # 2. Enhance
    processed = original_rgb.copy()
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=clahe_clip_limit)

    # 3. Denoise
    if use_denoise:
        processed = denoise_image(processed, method="bilateral")

    # 4. Letterbox
    letterboxed, ratio, pad = letterbox(processed, target_size=target_size)

    # 5. Tensor
    tensor = normalize_and_to_tensor(letterboxed)

    return {
        "original_rgb": original_rgb,
        "enhanced_rgb": processed,
        "letterboxed_rgb": letterboxed,
        "tensor": tensor,
        "ratio": ratio,
        "pad": pad,
        "original_shape": (orig_h, orig_w),
    }


def get_training_augmentations(img_size: int = 640):
    """Build Albumentations augmentation pipeline tailored for railway track defect training.

    Simulates track vibrations, directional shadows, sunlight reflections, and weather variations.

    Args:
        img_size: Input square resolution for training.

    Returns:
        albumentations.Compose: Configured augmentation pipeline.
    """
    import albumentations as A

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
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
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
