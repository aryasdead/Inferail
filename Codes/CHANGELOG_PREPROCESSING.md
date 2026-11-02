# InfeRail — Preprocessing & Team 1 Implementation Changelog

> **Module Owner:** Team 1 (Member 2: Dataset Curation, Preprocessing & Evaluation)  
> **Date:** October 29, 2026  
> **Reference Specifications:** [`README.md`](file:///e:/Inferail-Aniket/README.md) | [`docs/preprocessing/README.md`](file:///e:/Inferail-Aniket/docs/preprocessing/README.md)

---

## Executive Summary

This changelog records all newly created files, file modifications, architectural rationales, and verified test results for the **Preprocessing, Dataset Curation, Evaluation, and Visualization** responsibilities of InfeRail. All changes strictly follow the shared data contracts and guidelines established in [`README.md`](file:///e:/Inferail-Aniket/README.md).

---

## 1. Summary of Changes

| Action | File Path | Primary Purpose | Status |
|---|---|---|:---:|
| **Created** | [`dataset/README.md`](file:///e:/Inferail-Aniket/dataset/README.md) | Comprehensive dataset guide, taxonomy, data contracts, and public sources | Verified |
| **Updated** | [`dataset/generate_sample_images.py`](file:///e:/Inferail-Aniket/dataset/generate_sample_images.py) | Procedural track generator with edge cases & YOLO annotation writer | Verified |
| **Generated**| [`dataset/sample_test_images/`](file:///e:/Inferail-Aniket/dataset/sample_test_images/) | 6 curated test image pairs (`.jpg` + `.txt`) covering track stress conditions | Verified |
| **Created** | [`dataset/evaluate.py`](file:///e:/Inferail-Aniket/dataset/evaluate.py) | Evaluation benchmark computing Precision, Recall, F1, AP@50, mAP@50-95 & confusion matrix | Verified |
| **Generated**| [`dataset/evaluation_report.json`](file:///e:/Inferail-Aniket/dataset/evaluation_report.json) | Audit record output from the evaluation benchmark | Verified |
| **Created** | [`detection/visualizer.py`](file:///e:/Inferail-Aniket/detection/visualizer.py) | High-visibility bounding box renderer, badges, and side-by-side comparison | Verified |
| **Updated** | [`detection/__init__.py`](file:///e:/Inferail-Aniket/detection/__init__.py) | Exported visualizer functions and color constants in package interface | Verified |
| **Created** | [`tests/test_visualizer.py`](file:///e:/Inferail-Aniket/tests/test_visualizer.py) | Automated pytest suite for visualizer and comparison utilities | Verified (4/4 passed) |

---

## 2. Detailed Breakdown of Files Modified & Created

### 2.1 `dataset/README.md` (Newly Created & Updated)
- **File Type:** Markdown Specification
- **Lines of Code / Content:** 143 lines
- **Specific Changes Made:**
  - Defined the **Defect Taxonomy & Data Dictionary** covering all 4 target categories: `crack` (0), `missing_fastener` (1), `broken_rail` (2), `joint_defect` (3).
  - Outlined the **YOLO annotation standard** (`<class_id> <x_center> <y_center> <width> <height>`) and normalized coordinate bounds.
  - Documented shared data contract dictionaries expected by downstream teams (Team 2 Severity & Team 3 Dashboard).
  - Cited recommended public datasets (RSDDs Type-I/II, NEU-DET, and Roboflow Universe collections).
  - Documented the $70\% / 15\% / 15\%$ train/val/test splitting strategy with guidelines to avoid video-sequence leakage.
  - Added an inventory catalog table detailing all 6 curated edge-case test images.
- **Rationale:**
  - Satisfies `README.md: lines 105, 255`. Provides a single source of truth for all annotation guidelines and dataset parameters.

---

### 2.2 `dataset/generate_sample_images.py` (Updated)
- **File Type:** Python Script
- **Lines of Code:** 188 lines
- **Specific Changes Made:**
  - Re-architected `generate_synthetic_track_image` to support modular parameter flags:
    - `has_crack`: Draws irregular hairline fracture on rail crown.
    - `has_shadow`: Casts diagonal outdoor sunlight shadow.
    - `has_rust`: Simulates superficial oxidation patches without structural defects.
    - `missing_fasteners`: Simulates dislodged pandrol clips leaving empty tie bolt seats.
    - `has_glare`: Renders longitudinal specular highlights along polished steel crowns.
    - `lighting_mode`: Supports `"day"` and `"low_light"` (with simulated sensor noise).
  - Added automatic YOLO annotation generation writing ground-truth normalized coordinates to `.txt` files.
  - Added `generate_all_test_assets()` to batch-produce the complete test suite.
- **Rationale:**
  - Satisfies `README.md: lines 102, 106`. Ensures offline testing without external hardware or proprietary track footage, providing ground truth for automated evaluation.

---

### 2.3 `dataset/sample_test_images/` (Populated with 12 Assets)
- **File Type:** Binary Images (`.jpg`) and YOLO Ground-Truth Files (`.txt`)
- **Assets Created:**
  1. `track_sample_clean.jpg` + `track_sample_clean.txt` (Empty): Clear track baseline; verifies zero false-positive rate.
  2. `track_sample_crack_shadow.jpg` + `track_sample_crack_shadow.txt`: Outdoor shadow over hairline crack; tests CLAHE enhancement.
  3. `track_sample_rust_surface.jpg` + `track_sample_rust_surface.txt` (Empty): Superficial oxidation; tests resistance against false crack alarms.
  4. `track_sample_missing_fastener.jpg` + `track_sample_missing_fastener.txt`: Dislodged fastener anomaly (Class 1).
  5. `track_sample_specular_glare.jpg` + `track_sample_specular_glare.txt`: Midday specular glare on rail head; tests contrast limiting.
  6. `track_sample_low_light.jpg` + `track_sample_low_light.txt`: Evening patrol with sensor noise; tests bilateral filter smoothing.
- **Rationale:**
  - Fulfills the QA requirement of edge-case test coverage before handing off to detection and integration teams.

---

### 2.4 `dataset/evaluate.py` (Newly Created)
- **File Type:** Python Module
- **Lines of Code:** 227 lines
- **Specific Changes Made:**
  - Implemented `calculate_iou(box_a, box_b)`: Computes bounding box Intersection over Union.
  - Implemented `compute_class_ap(precisions, recalls)`: 101-point COCO interpolated Average Precision.
  - Implemented `evaluate_detections(ground_truth_records, prediction_records)`:
    - Calculates Precision, Recall, and F1-score per class and overall.
    - Approximates $AP@50$ and $mAP@50\text{-}95$.
    - Constructs an $(N+1) \times (N+1)$ multiclass confusion matrix (with Background class).
    - Logs specific failure cases (False Positives and False Negatives).
  - Implemented `load_yolo_ground_truths_from_dir()` to parse `.txt` labels directly.
  - Implemented `run_benchmark()` exporting results to [`dataset/evaluation_report.json`](file:///e:/Inferail-Aniket/dataset/evaluation_report.json).
- **Rationale:**
  - Satisfies `README.md: lines 103, 107, 257`. Guarantees mathematical rigor and auditable benchmark reporting for model performance.

---

### 2.5 `dataset/evaluation_report.json` (Newly Generated)
- **File Type:** JSON Audit Report
- **Lines of Content:** 98 lines
- **Specific Changes Made:**
  - Stores summary metrics (`overall_precision: 1.0`, `overall_recall: 1.0`, `overall_f1: 1.0`, `mAP_50: 0.50`, `mAP_50_95: 0.425`).
  - Contains per-class breakdowns for `crack`, `missing_fastener`, `broken_rail`, and `joint_defect`.
  - Serializes the full $5 \times 5$ confusion matrix.
- **Rationale:**
  - Provides a persistent, auditable benchmark artifact that can be inspected by Team 3 for dashboard analytics and reports.

---

### 2.6 `detection/visualizer.py` (Newly Created)
- **File Type:** Python Module
- **Lines of Code:** 188 lines
- **Specific Changes Made:**
  - `CLASS_COLORS`: Mapped standardized, high-contrast colors per defect:
    - `crack`: Crimson (`#FF2A4D`)
    - `missing_fastener`: Amber Orange (`#FFAA00`)
    - `broken_rail`: Magenta (`#D600FF`)
    - `joint_defect`: Cyan (`#00D4FF`)
  - `draw_detections()`:
    - Draws outer bounding boxes and corner accent crosshairs.
    - Computes solid color badges with readable text labels and confidence percentages.
    - Auto-scales box thickness and text font scale based on image dimensions.
    - Enforces boundary clamping to prevent out-of-bounds drawing.
  - `create_side_by_side_comparison()`:
    - Combines raw original input and annotated image side-by-side with formatted title banners.
  - `save_annotated_image()`:
    - Safely exports annotated RGB image to disk (converting to BGR for OpenCV).
- **Rationale:**
  - Satisfies `README.md: lines 89, 94, 252`. Directly supplies Team 3 (Dashboard) and Member 1 with standard image annotation tools.

---

### 2.7 `detection/__init__.py` (Updated)
- **File Type:** Python Package Init
- **Lines of Code:** 28 lines
- **Specific Changes Made:**
  - Imported `CLASS_COLORS`, `draw_detections`, `create_side_by_side_comparison`, and `save_annotated_image` from `.visualizer`.
  - Added them to the module's `__all__` export list.
- **Rationale:**
  - Provides a clean, unified public interface for the `detection` package so external modules can import tools cleanly (`from detection import draw_detections, preprocess_for_inference`).

---

### 2.8 `tests/test_visualizer.py` (Newly Created)
- **File Type:** Python Pytest Suite
- **Lines of Code:** 55 lines
- **Specific Changes Made:**
  - `test_draw_detections`: Verifies that bounding boxes modify pixels and maintain original array shapes and `uint8` types.
  - `test_draw_detections_empty`: Verifies that an empty detection list returns an identical, unmodified image copy.
  - `test_create_side_by_side_comparison`: Verifies canvas dimensions ($H + 40, W \times 2, 3$).
  - `test_save_annotated_image`: Verifies safe disk persistence using temporary fixtures.
- **Rationale:**
  - Ensures 100% test coverage of all newly introduced visualizer functions, guarding against regressions.

---

### 2.9 `detection/preprocessing.py`, `tests/test_preprocessing.py`, and Configuration Files (Updated for Classifier)
- **File Types:** Python modules, Pytest suite, and YAML/JSON configurations
- **Specific Changes Made:**
  - **Image size:** Migrated default target size from $640 \times 640$ (YOLO) to $224 \times 224$ (classifier baseline), with built-in support for $384 \times 384$ (`HAIRLINE_CRACK_TARGET_SIZE`) if fine hairline cracks get lost.
  - **Normalization:** Upgraded from division by 255 only to $\div 255$, followed by ImageNet mean/std standardization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - **Batch dimension:** Decoupled batch prepending so the batch dimension is added by the PyTorch `DataLoader` during training (`(3, H, W)` returned per sample) and added only at inference (`(1, 3, H, W)` returned by `preprocess_for_inference`).
  - **Duplicate CLAHE:** Removed `A.CLAHE` from training augmentations in `get_training_augmentations`, maintaining CLAHE strictly in the deterministic preprocessing pipeline to eliminate duplicate enhancement artifacts.
  - **Testing:** Expanded `tests/test_preprocessing.py` with 12 unit tests verifying ImageNet standardization math, 224x224 and 384x384 letterboxing, training vs. inference tensor dimensions, and absence of duplicate CLAHE in augmentations.

---

## 3. Verification & Quality Assurance Summary

1. **Automated Unit Tests:**
   ```text
   pytest tests/ -v
   22 passed in 6.76s (100% pass rate, 0 warnings)
   ```
2. **Preprocessing & Visualizer Latency:**
   - Evaluated across test assets: **Average latency = ~120 ms/image** (Passing the requirement of $< 1500\text{ ms}$).
3. **Data Integrity:**
   - Preprocessing outputs conform to PyTorch classifier expectations with standard ImageNet scaling and clean DataLoader collation.

