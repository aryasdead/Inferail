# InfeRail — Railway Defect Dataset Curation & Specification Guide

> **Module Owner:** Team 1 (Member 2: Dataset Curation, Preprocessing & Evaluation)  
> **Target Directory:** [`dataset/`](file:///e:/Inferail-Aniket/dataset/)  
> **Reference Specifications:** [`README.md`](file:///e:/Inferail-Aniket/README.md) | [`docs/preprocessing/README.md`](file:///e:/Inferail-Aniket/docs/preprocessing/README.md)

---

## 1. Overview & Objectives

High-fidelity computer vision defect detection on railway tracks requires disciplined data management. Outdoor track inspection environments present extreme real-world challenges:
- High contrast variations between polished steel heads and deep ballast shadows.
- Surface rust, oil stains, and track markings that mimic cracks.
- Small object scales (pandrol clips, hairline fissures) in high-resolution track images.

This document establishes the data dictionary, annotation taxonomy, dataset partition standards, external data sources, and synthetic verification assets for **InfeRail**.

---

## 2. Defect Taxonomy & Data Dictionary

InfeRail detects and evaluates four primary track anomalies as defined in the system data contract:

| Class ID | Class Name | Target Category | Visual Characteristics | Safety & Physical Significance |
|:---:|:---|:---|:---|:---|
| `0` | `crack` | Rail Head Surface Flaw | Dark, irregular, jagged linear or branching fissures along the polished running surface of the rail crown. | Critical precursors to catastrophic rail breaks caused by rolling contact fatigue (RCF). |
| `1` | `missing_fastener` | Component Dislodgement | Absence, breakage, or misalignment of Pandrol clips, bolts, or tie plates securing rail to sleepers. | Causes track gauge widening, tie instability, and elevated derailment risk under dynamic lateral train loads. |
| `2` | `broken_rail` | Complete Structural Break | Full transverse fracture through the rail head, web, or base with visible separation or gap. | Immediate derailment hazard; requires immediate line closure and emergency repair. |
| `3` | `joint_defect` | Fishplate / Joint Failure | Cracked or missing fishplates, sheared joint bolts, or excessive longitudinal rail expansion gap. | High mechanical impact site causing accelerated wheel-rail impact fatigue and track bed degradation. |

---

## 3. Annotation Standard & Shared Data Contract

### 3.1 YOLO Annotation Format (Training & Evaluation)
All bounding box annotations for object detection training must strictly adhere to the standardized **YOLO format**:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- Coordinates must be normalized floats within the range `[0.0, 1.0]` relative to image width and height:
  - `x_center = box_center_x / image_width`
  - `y_center = box_center_y / image_height`
  - `width = box_width / image_width`
  - `height = box_height / image_height`

Example annotation file (`track_milemarker_42_km12.txt`):
```text
0 0.325000 0.485000 0.045000 0.082000
1 0.285000 0.520000 0.035000 0.040000
```

### 3.2 System Interface Contract (Team 1 Output -> Teams 2 & 3 Input)
InfeRail's detection pipeline converts normalized bounding boxes back to original pixel coordinates via `inverse_letterbox_coords()` and returns Python dictionaries conforming to the project data contract:

```python
{
    "defect_id": 1,
    "class_name": "crack",               # "crack", "missing_fastener", "broken_rail", "joint_defect"
    "confidence": 0.89,                   # float between 0.0 and 1.0
    "bbox": [120, 245, 380, 410],        # [x_min, y_min, x_max, y_max] in original pixel space
    "area_pixels": 42900                 # estimated area (optional)
}
```

---

## 4. Public Railway Datasets & External Benchmarks

For model pre-training, fine-tuning, and performance benchmarking, the following open datasets are recommended:

### 4.1 Rail Surface Defect Datasets (RSDDs)
- **Type-I (Express Track):** Captured from high-speed passenger lines. Features uniform ballast, continuous welded rail, and clean running surfaces.
- **Type-II (Heavy Haul / Freight Track):** Captured from heavy freight lines. Features significant surface shelling, squats, abrasive wear, and oily rail crowns.
- **Access / Reference:** Widely available on Kaggle and academic GitHub repositories under research use terms.

### 4.2 NEU Surface Defect Database (NEU-DET)
- High-resolution industrial steel surface defect database containing six defect categories including cracks, inclusions, patches, pitted surfaces, rolled-in scale, and scratches.
- Ideal for transfer learning baseline weights for small metallic fracture detection.

### 4.3 Roboflow Universe Railway Datasets
- Open community datasets covering:
  - *Railway Track Fasteners Detection* (Pandrol clips, missing bolts).
  - *Railway Crack & Track Defect Segmentation/Detection*.
- Provides direct export in YOLOv8 PyTorch format with pre-configured train/val/test splits.

> **Attribution & Licensing Note:** Ensure all academic citations, CC-BY, or research licenses are properly acknowledged when incorporating external imagery into the training sets.

---

## 5. Dataset Partitioning & Splitting Strategy

To ensure robust generalization without data leakage across sequential track inspection videos, use the following split:

| Partition | Proportion | Target Role | Key Validation Criteria |
|:---|:---:|:---|:---|
| **Train** | `70%` | Model optimization & weight updates | Augmented with brightness shifts, Gaussian noise, and motion blurs. |
| **Validation** | `15%` | Hyperparameter tuning & early stopping | Evaluated during training epochs to prevent overfitting. |
| **Test** | `15%` | Unbiased benchmark evaluation | Strictly isolated; no augmentations applied; contains all edge cases. |

### Splitting Guidelines
1. **Group by Track Segment / Video Sequence:** Do not place consecutive frames from the same patrol recording in both Train and Test. Split by distinct geographic locations or dates to evaluate true generalization.
2. **Class Stratification:** Maintain consistent defect distribution across all three splits, ensuring rare classes (`broken_rail`, `joint_defect`) appear proportionally in validation and test partitions.

---

## 6. Curated Test Assets & Edge-Case Directory (`dataset/sample_test_images/`)

The curated test suite provides a comprehensive set of test images alongside standardized YOLO ground-truth annotations (`.txt`) to test both the preprocessing transforms (CLAHE, bilateral filter, letterboxing) and the detection model under difficult outdoor railway environments:

| Image File | YOLO Label File | Edge Case / Stress Test | Description & Validation Goal |
|---|---|---|---|
| [`track_sample_clean.jpg`](file:///e:/Inferail-Aniket/dataset/sample_test_images/track_sample_clean.jpg) | `track_sample_clean.txt` (empty) | **Baseline (No Defects)** | Normal concrete sleepers, clean steel rails, gravel ballast. Tests system resistance to false positives. |
| [`track_sample_crack_shadow.jpg`](file:///e:/Inferail-Aniket/dataset/sample_test_images/track_sample_crack_shadow.jpg) | `track_sample_crack_shadow.txt` | **Shadow Occlusion & CLAHE** | Diagonal sun shadow across rail head with hairline crack. Verifies CLAHE reveals fractures in deep shadows without grain explosion. |
| [`track_sample_rust_surface.jpg`](file:///e:/Inferail-Aniket/dataset/sample_test_images/track_sample_rust_surface.jpg) | `track_sample_rust_surface.txt` (empty) | **Rust vs. Real Fracture** | Superficial oxidation patches along rail crown. Confirms surface rust is not falsely classified as a structural crack. |
| [`track_sample_missing_fastener.jpg`](file:///e:/Inferail-Aniket/dataset/sample_test_images/track_sample_missing_fastener.jpg) | `track_sample_missing_fastener.txt` | **Missing Fastener / Dislodgement** | Dislodged pandrol clips on sleeper ties with empty bolt seats. Tests detection of small mechanical track fixtures. |
| [`track_sample_specular_glare.jpg`](file:///e:/Inferail-Aniket/dataset/sample_test_images/track_sample_specular_glare.jpg) | `track_sample_specular_glare.txt` | **Harsh Sun Glare & Overexposure** | Specular highlight band along polished rail head. Verifies contrast limiting handles high dynamic range without blooming. |
| [`track_sample_low_light.jpg`](file:///e:/Inferail-Aniket/dataset/sample_test_images/track_sample_low_light.jpg) | `track_sample_low_light.txt` | **Low-Light / Sensor Noise** | Underexposed evening patrol imagery with high ISO grain. Verifies bilateral filter denoising preserves crack edges. |

### 6.2 Generating or Updating Curated Assets
Run the generator script to reproduce or expand test assets:
```bash
python dataset/generate_sample_images.py
```

---

## 7. Quality Assurance & Evaluation Protocol

Before handing off the dataset and test sets to **Team 1 Member 1** and **Team 3**, verify the following checklist:

- [ ] **Image Integrity:** Zero corrupted, truncated, or zero-byte images in the dataset.
- [ ] **Annotation Bounds:** All coordinates satisfy $0.0 \le x, y, w, h \le 1.0$.
- [ ] **Class Balance:** Document any underrepresented classes so confidence thresholds in `detector.py` can be calibrated.
- [ ] **Preprocessing Compatibility:** All images cleanly pass through `detection.preprocessing.validate_image()` without raising uncaught exceptions.
- [ ] **Edge Cases Covered:** Test set explicitly includes:
  - Specular glare on rail head.
  - Deep shadow covering rail base.
  - Wet or rusted rail surface.
  - Pristine, defect-free track.
