"""Unit tests for InfeRail Crack Classifier & Localization utilities."""

import pytest
import numpy as np
import torch

from detection.crack_classifier import (
    CrackClassifier,
    classify_crack,
    classify_image,
    compute_cam,
    detect_and_visualize_cracks,
    get_model_summary,
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
    MODEL_INPUT_SIZE,
)
from detection.preprocessing import preprocess_for_classification


@pytest.fixture
def dummy_model():
    """Create a lightweight CrackClassifier for testing."""
    return CrackClassifier(pretrained=False, freeze_backbone=False)


@pytest.fixture
def dummy_rgb_image():
    """Create a synthetic 100x150 RGB railway track image."""
    np.random.seed(42)
    return np.random.randint(50, 200, size=(100, 150, 3), dtype=np.uint8)


class TestCrackClassifierModel:
    def test_model_instantiation(self, dummy_model):
        assert dummy_model is not None
        assert dummy_model.get_total_params() > 0
        assert dummy_model.get_trainable_params() > 0

    def test_model_forward_shape(self, dummy_model):
        dummy_tensor = torch.randn(2, 3, 224, 224)
        out = dummy_model(dummy_tensor)
        assert out.shape == (2, 1)
        # Sigmoid output in [0, 1]
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_model_summary(self):
        summary = get_model_summary()
        assert summary["model_name"] == "CrackClassifier"
        assert summary["input_shape"] == "(B, 3, 224, 224)"
        assert summary["class_names"] == {0: "non-cracked", 1: "cracked"}


class TestClassifyCrack:
    def test_classify_crack_contract(self, dummy_model, dummy_rgb_image):
        pre = preprocess_for_classification(dummy_rgb_image)
        res = classify_crack(pre, model=dummy_model, threshold=0.5)

        assert "prediction" in res
        assert res["prediction"] in ["cracked", "non-cracked"]
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0
        assert "crack_probability" in res
        assert 0.0 <= res["crack_probability"] <= 1.0
        assert res["class_id"] in [0, 1]


class TestComputeCam:
    def test_compute_cam_shape_and_range(self, dummy_model):
        dummy_tensor = torch.randn(1, 3, 224, 224)
        cam = compute_cam(dummy_model, dummy_tensor, device="cpu")

        assert isinstance(cam, np.ndarray)
        assert cam.shape == (7, 7)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0


class TestDetectAndVisualizeCracks:
    def test_detect_and_visualize_output_structure(self, dummy_model, dummy_rgb_image):
        res = detect_and_visualize_cracks(
            dummy_rgb_image,
            model=dummy_model,
            generate_comparison=True,
        )

        assert "prediction" in res
        assert "confidence" in res
        assert "confidence_pct" in res
        assert "detections" in res
        assert isinstance(res["detections"], list)
        assert "annotated_rgb" in res
        assert res["annotated_rgb"].shape == dummy_rgb_image.shape
        assert "comparison_canvas" in res
        assert res["comparison_canvas"] is not None

