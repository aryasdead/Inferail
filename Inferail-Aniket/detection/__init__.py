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
    compute_edge_density,
    preprocess_for_inference,
    preprocess_for_classification,
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
from .crack_classifier import (
    CrackClassifier,
    classify_crack,
    classify_image,
    compute_cam,
    detect_and_visualize_cracks,
    save_model,
    load_model,
    get_model_summary,
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
    MODEL_INPUT_SIZE,
)

__all__ = [
    # Preprocessing constants
    "DEFAULT_TARGET_SIZE",
    "HAIRLINE_CRACK_TARGET_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Preprocessing functions
    "validate_image",
    "apply_clahe",
    "denoise_image",
    "letterbox",
    "normalize_and_to_tensor",
    "compute_edge_density",
    "preprocess_for_inference",
    "preprocess_for_classification",
    "preprocess_for_training",
    "TrackClassifierDataset",
    "inverse_letterbox_coords",
    "get_training_augmentations",
    # Visualizer
    "CLASS_COLORS",
    "draw_detections",
    "create_side_by_side_comparison",
    "save_annotated_image",
    # Crack Classifier
    "CrackClassifier",
    "classify_crack",
    "classify_image",
    "compute_cam",
    "detect_and_visualize_cracks",
    "save_model",
    "load_model",
    "get_model_summary",
    "CLASS_NAMES",
    "DEFAULT_THRESHOLD",
    "MODEL_INPUT_SIZE",
]
