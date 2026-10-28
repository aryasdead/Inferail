"""InfeRail Defect Visualizer and Bounding Box Rendering Utilities.

Owned by: Team 1 (Computer Vision & AI Pipeline Lead)
Description:
    Renders high-visibility, color-coded bounding boxes, confidence badges,
    and side-by-side inspection comparisons on railway track images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# High-visibility RGB color palette mapped to standard InfeRail defect classes
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "crack": (255, 42, 77),            # High-visibility Crimson
    "missing_fastener": (255, 170, 0),  # Bright Amber Orange
    "broken_rail": (214, 0, 255),       # Critical Magenta Purple
    "joint_defect": (0, 212, 255),      # Bright Cyan
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 128)  # Bright Spring Green


def draw_detections(
    image_rgb: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    box_thickness: Optional[int] = None,
    font_scale: Optional[float] = None,
) -> np.ndarray:
    """Draw defect bounding boxes and readable badge labels on an RGB track image.

    Args:
        image_rgb: Input image array (H, W, 3) in RGB format and uint8 dtype.
        detections: List of detection dicts conforming to the shared data contract:
            [{"class_name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]
        show_confidence: Whether to display confidence percentage alongside class label.
        box_thickness: Line thickness for boxes. Automatically scaled if None.
        font_scale: Font scale for text. Automatically scaled if None.

    Returns:
        np.ndarray: Annotated RGB image copy.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    # Auto-scale thickness based on image diagonal
    scale_factor = max(w, h) / 1000.0
    thickness = box_thickness if box_thickness is not None else max(2, int(round(2.0 * scale_factor)))
    text_scale = font_scale if font_scale is not None else max(0.45, 0.55 * scale_factor)

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        # Clamp to image boundaries
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = str(det.get("class_name", "defect")).lower()
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        conf = det.get("confidence")

        # 1. Draw outer bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        # 2. Draw corner accent brackets (modern UI accent)
        corner_len = max(8, int(min(x2 - x1, y2 - y1) * 0.2))
        corner_thick = thickness + 1
        # Top-left corner
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
        # Top-right corner
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
        # Bottom-left corner
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
        # Bottom-right corner
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, corner_thick)

        # 3. Format badge text
        if show_confidence and conf is not None:
            label_text = f"{class_name.upper()} {conf * 100:.1f}%"
        else:
            label_text = class_name.upper()

        # Compute text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        (t_w, t_h), baseline = cv2.getTextSize(label_text, font, text_scale, 1)

        # Badge placement: Above box if room, otherwise inside top
        badge_y1 = y1 - t_h - 8 if (y1 - t_h - 8) > 0 else y1
        badge_y2 = badge_y1 + t_h + 8
        badge_x2 = min(w, x1 + t_w + 10)

        # Draw solid badge background with defect color
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # White or black text for optimal readability
        cv2.putText(
            annotated,
            label_text,
            (x1 + 5, badge_y2 - 4),
            font,
            text_scale,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def create_side_by_side_comparison(
    original_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    header_height: int = 40,
) -> np.ndarray:
    """Combine original image and annotated image side-by-side with header banners.

    Args:
        original_rgb: Unmodified input image array (H, W, 3).
        annotated_rgb: Annotated output image array (H, W, 3).
        header_height: Height of the top label banner in pixels.

    Returns:
        np.ndarray: Combined horizontal comparison canvas.
    """
    if original_rgb.shape != annotated_rgb.shape:
        # Resize annotated to match original dimensions if needed
        h, w = original_rgb.shape[:2]
        annotated_rgb = cv2.resize(annotated_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    h, w = original_rgb.shape[:2]

    # Create top banner
    banner = np.full((header_height, w * 2, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(banner, "ORIGINAL TRACK CAPTURE", (int(w * 0.25), int(header_height * 0.65)), font, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(banner, "AI DEFECT DETECTION (INFERAIL)", (w + int(w * 0.2), int(header_height * 0.65)), font, 0.65, (0, 212, 255), 2, cv2.LINE_AA)

    # Concatenate horizontally
    side_by_side = np.hstack((original_rgb, annotated_rgb))
    canvas = np.vstack((banner, side_by_side))
    return canvas


def save_annotated_image(
    image_rgb: np.ndarray,
    output_path: Union[str, Path],
) -> str:
    """Save an RGB annotated image to disk, ensuring directory existence.

    Args:
        image_rgb: Annotated RGB image array.
        output_path: Target file destination (e.g., 'outputs/result.jpg').

    Returns:
        str: Absolute path of saved image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return str(path.resolve())
