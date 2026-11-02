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


def test_letterbox(dummy_rgb_image):
    target_size = (640, 640)
    letterboxed, ratio, (dw, dh) = letterbox(dummy_rgb_image, target_size=target_size)

    assert letterboxed.shape == (640, 640, 3)
    assert ratio == pytest.approx(640 / 1280)  # width is limiting factor
    assert dw == 0.0
    assert dh == pytest.approx((640 - 720 * ratio) / 2)


def test_normalize_and_to_tensor():
    img = np.full((640, 640, 3), 255, dtype=np.uint8)
    tensor = normalize_and_to_tensor(img)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() == pytest.approx(1.0)
    assert tensor.min() == pytest.approx(1.0)


def test_inverse_letterbox_coords():
    orig_shape = (720, 1280)
    target_size = (640, 640)
    ratio = 640 / 1280  # 0.5
    dh = (640 - 720 * 0.5) / 2  # 140.0
    dw = 0.0

    # Box located in the center of letterboxed 640x640: [x1, y1, x2, y2]
    # y ranges from 140 to 500 on letterboxed
    boxes_letterboxed = np.array([[100, 140, 200, 240]], dtype=np.float32)
    orig_boxes = inverse_letterbox_coords(boxes_letterboxed, ratio=ratio, pad=(dw, dh), original_shape=orig_shape)

    assert orig_boxes.shape == (1, 4)
    # x should be 100 / 0.5 = 200, 200 / 0.5 = 400
    assert orig_boxes[0, 0] == pytest.approx(200.0)
    assert orig_boxes[0, 2] == pytest.approx(400.0)
    # y should be (140 - 140) / 0.5 = 0, (240 - 140) / 0.5 = 200
    assert orig_boxes[0, 1] == pytest.approx(0.0)
    assert orig_boxes[0, 3] == pytest.approx(200.0)


def test_preprocess_for_inference(dummy_rgb_image):
    result = preprocess_for_inference(dummy_rgb_image, target_size=(640, 640))

    assert "original_rgb" in result
    assert "enhanced_rgb" in result
    assert "letterboxed_rgb" in result
    assert "tensor" in result
    assert "ratio" in result
    assert "pad" in result
    assert "original_shape" in result

    assert result["tensor"].shape == (1, 3, 640, 640)
    assert result["letterboxed_rgb"].shape == (640, 640, 3)
