"""Unit tests for the InfeRail image preprocessing pipeline."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from detection.preprocessing import (
    validate_image,
    apply_clahe,
    denoise_image,
    letterbox,
    normalize_and_to_tensor,
    inverse_letterbox_coords,
    preprocess_for_inference,
    preprocess_for_training,
    TrackClassifierDataset,
    get_training_augmentations,
    DEFAULT_TARGET_SIZE,
    HAIRLINE_CRACK_TARGET_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@pytest.fixture
def dummy_rgb_image():
    """Create a synthetic 720x1280 RGB image."""
    np.random.seed(0)
    return np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)


def test_validate_image_with_numpy(dummy_rgb_image):
    result = validate_image(dummy_rgb_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == (720, 1280, 3)
    assert result.dtype == np.uint8


def test_validate_image_with_pil(dummy_rgb_image):
    pil_img = Image.fromarray(dummy_rgb_image)
    result = validate_image(pil_img)
    assert result.shape == (720, 1280, 3)
    assert result.dtype == np.uint8


def test_validate_image_invalid():
    with pytest.raises(ValueError):
        validate_image(None)

    with pytest.raises(ValueError):
        validate_image(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        validate_image("non_existent_image_path.jpg")


def test_apply_clahe(dummy_rgb_image):
    enhanced = apply_clahe(dummy_rgb_image, clip_limit=2.5)
    assert enhanced.shape == dummy_rgb_image.shape
    assert enhanced.dtype == np.uint8


def test_denoise_image(dummy_rgb_image):
    denoised = denoise_image(dummy_rgb_image, method="bilateral")
    assert denoised.shape == dummy_rgb_image.shape
    assert denoised.dtype == np.uint8

    denoised_gauss = denoise_image(dummy_rgb_image, method="gaussian")
    assert denoised_gauss.shape == dummy_rgb_image.shape


def test_letterbox_defaults(dummy_rgb_image):
    # Default target size is (224, 224) for classifier
    letterboxed, ratio, (dw, dh) = letterbox(dummy_rgb_image)

    assert letterboxed.shape == (224, 224, 3)
    assert ratio == pytest.approx(224 / 1280)  # width is limiting factor
    assert dw == 0.0
    assert dh == pytest.approx((224 - 720 * ratio) / 2)


def test_letterbox_hairline_cracks(dummy_rgb_image):
    # High-resolution target size (384, 384) when hairline cracks get lost
    letterboxed, ratio, (dw, dh) = letterbox(dummy_rgb_image, target_size=HAIRLINE_CRACK_TARGET_SIZE)

    assert letterboxed.shape == (384, 384, 3)
    assert ratio == pytest.approx(384 / 1280)


def test_normalize_and_to_tensor_imagenet():
    # Test ImageNet normalization: ÷255, then ImageNet mean/std
    img = np.full((224, 224, 3), 255, dtype=np.uint8)

    # In training (added by DataLoader): no batch dimension -> (3, H, W)
    tensor = normalize_and_to_tensor(img, normalize_imagenet=True, add_batch_dim=False)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32

    # Expected standardized values: (1.0 - mean) / std
    expected_c0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_c1 = (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_c2 = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    assert tensor[0, 0, 0] == pytest.approx(expected_c0, rel=1e-4)
    assert tensor[1, 0, 0] == pytest.approx(expected_c1, rel=1e-4)
    assert tensor[2, 0, 0] == pytest.approx(expected_c2, rel=1e-4)

    # At inference: batch dimension added -> (1, 3, H, W)
    tensor_infer = normalize_and_to_tensor(img, normalize_imagenet=True, add_batch_dim=True)
    assert tensor_infer.shape == (1, 3, 224, 224)


def test_normalize_and_to_tensor_div255_only():
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    tensor = normalize_and_to_tensor(img, normalize_imagenet=False, add_batch_dim=False)

    assert tensor.shape == (3, 224, 224)
    assert tensor.max() == pytest.approx(1.0)
    assert tensor.min() == pytest.approx(1.0)


def test_inverse_letterbox_coords():
    orig_shape = (720, 1280)
    target_size = (224, 224)
    ratio = 224 / 1280
    dh = (224 - 720 * ratio) / 2
    dw = 0.0

    boxes_letterboxed = np.array([[50, 60, 100, 120]], dtype=np.float32)
    orig_boxes = inverse_letterbox_coords(boxes_letterboxed, ratio=ratio, pad=(dw, dh), original_shape=orig_shape)

    assert orig_boxes.shape == (1, 4)
    assert orig_boxes[0, 0] == pytest.approx(50 / ratio)
    assert orig_boxes[0, 2] == pytest.approx(100 / ratio)
    assert orig_boxes[0, 1] == pytest.approx((60 - dh) / ratio)
    assert orig_boxes[0, 3] == pytest.approx((120 - dh) / ratio)


def test_preprocess_for_inference(dummy_rgb_image):
    result = preprocess_for_inference(dummy_rgb_image)

    assert "original_rgb" in result
    assert "enhanced_rgb" in result
    assert "letterboxed_rgb" in result
    assert "tensor" in result
    assert "ratio" in result
    assert "pad" in result
    assert "original_shape" in result

    # Batch dimension added only at inference: (1, 3, 224, 224)
    assert result["tensor"].shape == (1, 3, 224, 224)
    assert result["letterboxed_rgb"].shape == (224, 224, 3)

    # Test with 384x384 resolution
    result_384 = preprocess_for_inference(dummy_rgb_image, target_size=(384, 384))
    assert result_384["tensor"].shape == (1, 3, 384, 384)
    assert result_384["letterboxed_rgb"].shape == (384, 384, 3)


def test_get_training_augmentations_no_duplicate_clahe():
    pipeline = get_training_augmentations(img_size=224)

    # Extract transformation class names
    transform_names = [t.__class__.__name__ for t in pipeline.transforms]

    # Verify duplicate CLAHE is removed from augmentations (kept in pipeline only)
    assert "CLAHE" not in transform_names

    # Verify standard classifier augmentations exist (flip, brightness/contrast, light blur, small shift/scale/rotate)
    assert any("Flip" in name for name in transform_names)
    assert any("BrightnessContrast" in name for name in transform_names)
    assert any("Affine" in name or "ShiftScaleRotate" in name for name in transform_names)
    assert any("OneOf" in name for name in transform_names)


def test_inverse_letterbox_coords_1d():
    orig_shape = (720, 1280)
    ratio = 224 / 1280
    dh = (224 - 720 * ratio) / 2
    dw = 0.0

    # 1D single box input: [x1, y1, x2, y2]
    box_1d = [50, 60, 100, 120]
    orig_box = inverse_letterbox_coords(box_1d, ratio=ratio, pad=(dw, dh), original_shape=orig_shape)

    assert orig_box.shape == (4,)
    assert orig_box[0] == pytest.approx(50 / ratio)
    assert orig_box[2] == pytest.approx(100 / ratio)


def test_letterbox_int_target_size(dummy_rgb_image):
    # Support integer square size
    letterboxed, ratio, _ = letterbox(dummy_rgb_image, target_size=224)
    assert letterboxed.shape == (224, 224, 3)


def test_normalize_and_to_tensor_float_input():
    # Pass pre-scaled float [0.0, 1.0] array; must not divide by 255 twice
    img_float = np.ones((224, 224, 3), dtype=np.float32)
    tensor = normalize_and_to_tensor(img_float, normalize_imagenet=True, add_batch_dim=False)

    expected_c0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert tensor[0, 0, 0] == pytest.approx(expected_c0, rel=1e-4)


def test_validate_image_float_formats():
    # Float [0.0, 1.0] -> uint8 [0, 255]
    img_float_01 = np.ones((50, 50, 3), dtype=np.float32)
    validated = validate_image(img_float_01)
    assert validated.dtype == np.uint8
    assert validated.max() == 255

    # Float [0.0, 255.0] -> uint8 [0, 255]
    img_float_255 = np.full((50, 50, 3), 128.0, dtype=np.float32)
    validated_255 = validate_image(img_float_255)
    assert validated_255.dtype == np.uint8
    assert validated_255.max() == 128


def test_preprocess_for_training(dummy_rgb_image):
    result = preprocess_for_training(dummy_rgb_image, target_size=(224, 224), augment=True)

    assert "tensor" in result
    assert "processed_rgb" in result
    assert "original_shape" in result

    # CRITICAL: Batch dimension must NOT be added here (shape: 3, 224, 224)
    # The DataLoader in training adds the batch dimension
    assert result["tensor"].shape == (3, 224, 224)
    assert result["processed_rgb"].shape == (224, 224, 3)


def test_track_classifier_dataset_with_dataloader(tmp_path, dummy_rgb_image):
    if TrackClassifierDataset is None:
        pytest.skip("PyTorch not installed")

    import cv2
    import torch
    from torch.utils.data import DataLoader

    # Save 4 dummy image files
    img_paths = []
    labels = [0, 1, 2, 3]
    for idx in range(4):
        file_path = tmp_path / f"track_{idx}.jpg"
        cv2.imwrite(str(file_path), cv2.cvtColor(dummy_rgb_image, cv2.COLOR_RGB2BGR))
        img_paths.append(str(file_path))

    dataset = TrackClassifierDataset(
        image_paths=img_paths,
        labels=labels,
        target_size=(224, 224),
        is_training=True,
    )
    assert len(dataset) == 4

    # Single item from Dataset: shape (3, 224, 224)
    sample_tensor, sample_label = dataset[0]
    assert sample_tensor.shape == (3, 224, 224)
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_label == 0

    # Batching via DataLoader: adds batch dimension -> (2, 3, 224, 224)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_tensors, batch_labels = next(iter(loader))

    assert batch_tensors.shape == (2, 3, 224, 224)
    assert batch_labels.shape == (2,)

