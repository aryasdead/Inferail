"""Unit tests for detection/visualizer.py."""

from pathlib import Path
import numpy as np
import pytest

from detection.visualizer import (
    CLASS_COLORS,
    draw_detections,
    create_side_by_side_comparison,
    save_annotated_image,
)


@pytest.fixture
def dummy_track_image():
    """Create a synthetic 720x1280 RGB image."""
    return np.full((720, 1280, 3), 120, dtype=np.uint8)


def test_draw_detections(dummy_track_image):
    detections = [
        {"class_name": "crack", "confidence": 0.94, "bbox": [100, 150, 300, 350]},
        {"class_name": "missing_fastener", "confidence": 0.88, "bbox": [500, 200, 560, 260]},
    ]
    annotated = draw_detections(dummy_track_image, detections)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == dummy_track_image.shape
    assert annotated.dtype == np.uint8
    # Ensure drawing modified pixels
    assert not np.array_equal(annotated, dummy_track_image)


def test_draw_detections_empty(dummy_track_image):
    annotated = draw_detections(dummy_track_image, [])
    assert np.array_equal(annotated, dummy_track_image)


def test_create_side_by_side_comparison(dummy_track_image):
    annotated = dummy_track_image.copy()
    canvas = create_side_by_side_comparison(dummy_track_image, annotated, header_height=40)

    h, w = dummy_track_image.shape[:2]
    assert canvas.shape == (h + 40, w * 2, 3)
    assert canvas.dtype == np.uint8


def test_save_annotated_image(dummy_track_image, tmp_path):
    out_file = tmp_path / "test_annotated.jpg"
    saved_path = save_annotated_image(dummy_track_image, out_file)

    assert Path(saved_path).is_file()
    assert Path(saved_path).stat().st_size > 0
