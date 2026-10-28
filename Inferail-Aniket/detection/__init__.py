"""InfeRail Detection Package (Team 1: Computer Vision & AI Pipeline)."""

from .preprocessing import (
    validate_image,
    apply_clahe,
    denoise_image,
    letterbox,
    normalize_and_to_tensor,
    preprocess_for_inference,
    inverse_letterbox_coords,
    get_training_augmentations,
)
from .visualizer import (
    CLASS_COLORS,
    draw_detections,
    create_side_by_side_comparison,
    save_annotated_image,
)

__all__ = [
    "validate_image",
    "apply_clahe",
    "denoise_image",
    "letterbox",
    "normalize_and_to_tensor",
    "preprocess_for_inference",
    "inverse_letterbox_coords",
    "get_training_augmentations",
    "CLASS_COLORS",
    "draw_detections",
    "create_side_by_side_comparison",
    "save_annotated_image",
]
