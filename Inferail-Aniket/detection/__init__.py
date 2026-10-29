"""InfeRail Detection Package (Team 1: Computer Vision & AI Pipeline)."""

from .preprocessing import (
    DEFAULT_TARGET_SIZE,
    HAIRLINE_CRACK_TARGET_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    validate_image,
    apply_clahe,
    denoise_image,
    letterbox,
    normalize_and_to_tensor,
    preprocess_for_inference,
    preprocess_for_training,
    TrackClassifierDataset,
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
    "DEFAULT_TARGET_SIZE",
    "HAIRLINE_CRACK_TARGET_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "validate_image",
    "apply_clahe",
    "denoise_image",
    "letterbox",
    "normalize_and_to_tensor",
    "preprocess_for_inference",
    "preprocess_for_training",
    "TrackClassifierDataset",
    "inverse_letterbox_coords",
    "get_training_augmentations",
    "CLASS_COLORS",
    "draw_detections",
    "create_side_by_side_comparison",
    "save_annotated_image",
]
