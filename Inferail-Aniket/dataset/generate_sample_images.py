"""Generate realistic synthetic railway track images for offline testing and preprocessing validation.

Generates curated edge-case track images and ground-truth YOLO annotation files
covering clear track, shadows, surface rust, missing fasteners, specular glare,
and low-light conditions.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


def generate_synthetic_track_image(
    output_path: str,
    width: int = 1280,
    height: int = 720,
    has_crack: bool = False,
    has_shadow: bool = False,
    has_rust: bool = False,
    missing_fasteners: bool = False,
    has_glare: bool = False,
    lighting_mode: str = "day",  # "day", "low_light"
) -> Tuple[str, Optional[str]]:
    """Generate a realistic test image of a railway track segment with simulated defects and edge cases.

    Args:
        output_path: Path to save the image (JPG/PNG).
        width: Image width in pixels.
        height: Image height in pixels.
        has_crack: Whether to draw a simulated hairline rail fracture.
        has_shadow: Whether to simulate outdoor directional sun shadows.
        has_rust: Whether to draw superficial surface oxidation patches.
        missing_fasteners: Whether to simulate missing fastener clip anomalies.
        has_glare: Whether to simulate intense specular rail head glare.
        lighting_mode: Ambient lighting ("day" or "low_light").

    Returns:
        Tuple[str, Optional[str]]: (saved_image_path, saved_yolo_label_path)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_path = path.with_suffix(".txt")

    ground_truth_boxes: List[Tuple[int, float, float, float, float]] = []

    # 1. Base ballast texture (gravel - noisy brownish gray)
    np.random.seed(hash(path.name) % (2**32))
    base_loc = 110 if lighting_mode == "day" else 50
    base_scale = 35 if lighting_mode == "day" else 20
    ballast = np.random.normal(loc=base_loc, scale=base_scale, size=(height, width, 3)).clip(0, 255).astype(np.uint8)

    # Tint ballast to gravel tone (brownish-gray)
    ballast[:, :, 0] = (ballast[:, :, 0] * 0.9).astype(np.uint8)  # R
    ballast[:, :, 1] = (ballast[:, :, 1] * 0.85).astype(np.uint8) # G
    ballast[:, :, 2] = (ballast[:, :, 2] * 0.75).astype(np.uint8) # B

    img = ballast.copy()

    # 2. Add Railway Ties / Sleepers (horizontal wooden/concrete beams)
    tie_height = 50
    tie_spacing = 140
    for y in range(20, height, tie_spacing):
        y1 = y
        y2 = min(y + tie_height, height)
        tie_color = np.array([75, 70, 65], dtype=np.uint8) if lighting_mode == "day" else np.array([35, 32, 30], dtype=np.uint8)
        img[y1:y2, :] = (img[y1:y2, :] * 0.3 + tie_color * 0.7).astype(np.uint8)

    # 3. Add Steel Rails (Two vertical parallel steel beams)
    left_rail_x = int(width * 0.32)
    right_rail_x = int(width * 0.68)

    for rx in [left_rail_x, right_rail_x]:
        # Rail base / flange (dark metallic)
        base_color = (60, 60, 65) if lighting_mode == "day" else (30, 30, 35)
        cv2.rectangle(img, (rx - 25, 0), (rx + 25, height), base_color, -1)

        # Rail crown / head (bright polished steel)
        crown_color = (175, 175, 180) if lighting_mode == "day" else (90, 90, 95)
        cv2.rectangle(img, (rx - 15, 0), (rx + 15, height), crown_color, -1)

        # Center shine specular line
        shine_color = (220, 220, 225) if lighting_mode == "day" else (120, 120, 125)
        cv2.line(img, (rx - 2, 0), (rx - 2, height), shine_color, 3)

        # Fastener clips on ties
        for tie_idx, y in enumerate(range(20, height, tie_spacing)):
            fy = y + tie_height // 2

            # Left fastener
            left_missing = missing_fasteners and (rx == left_rail_x) and (tie_idx == 2)
            if left_missing:
                # Missing fastener seat: faint bolt hole without clip
                cv2.circle(img, (rx - 32, fy), 4, (35, 35, 40), -1)
                # Box for missing fastener (Class 1)
                bx, by, bw, bh = (rx - 32) / width, fy / height, 24 / width, 24 / height
                ground_truth_boxes.append((1, bx, by, bw, bh))
            else:
                clip_color = (45, 45, 50) if lighting_mode == "day" else (25, 25, 30)
                cv2.circle(img, (rx - 32, fy), 8, clip_color, -1)

            # Right fastener
            right_missing = missing_fasteners and (rx == right_rail_x) and (tie_idx == 3)
            if right_missing:
                cv2.circle(img, (rx + 32, fy), 4, (35, 35, 40), -1)
                bx, by, bw, bh = (rx + 32) / width, fy / height, 24 / width, 24 / height
                ground_truth_boxes.append((1, bx, by, bw, bh))
            else:
                clip_color = (45, 45, 50) if lighting_mode == "day" else (25, 25, 30)
                cv2.circle(img, (rx + 32, fy), 8, clip_color, -1)

    # 4. Add Surface Rust (Superficial cosmetic oxidation - non-structural)
    if has_rust:
        rust_overlay = img.copy()
        # Create irregular patches of brown-orange rust along the right rail crown
        for ry in range(100, 480, 25):
            center_pt = (right_rail_x + np.random.randint(-8, 8), ry + np.random.randint(-10, 10))
            axes = (np.random.randint(6, 14), np.random.randint(12, 22))
            angle = np.random.randint(0, 180)
            rust_color = (np.random.randint(140, 180), np.random.randint(70, 95), np.random.randint(30, 45))
            cv2.ellipse(rust_overlay, center_pt, axes, angle, 0, 360, rust_color, -1)
        # Blend rust softly into the rail
        img = cv2.addWeighted(img, 0.45, rust_overlay, 0.55, 0)

    # 5. Add Harsh Specular Glare (Midday intense sun reflection)
    if has_glare:
        glare_mask = np.zeros((height, width, 3), dtype=np.uint8)
        # Intense flare along the left rail crown from y=150 to y=450
        cv2.line(glare_mask, (left_rail_x, 150), (left_rail_x, 450), (255, 255, 255), 18)
        cv2.GaussianBlur(glare_mask, (25, 25), 0, dst=glare_mask)
        img = cv2.add(img, (glare_mask * 0.75).astype(np.uint8))

    # 6. Add Directional Outdoor Sun Shadow across track
    if has_shadow:
        shadow_mask = np.ones((height, width), dtype=np.float32)
        pts = np.array([[0, 0], [width // 2, 0], [width, height], [0, height // 2]], np.int32)
        cv2.fillPoly(shadow_mask, [pts], 0.38)
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * shadow_mask).astype(np.uint8)

    # 7. Add Simulated Surface Crack on the left rail head
    if has_crack:
        crack_points = [
            (left_rail_x - 10, 320),
            (left_rail_x - 5, 335),
            (left_rail_x + 2, 342),
            (left_rail_x - 3, 358),
            (left_rail_x + 8, 375),
        ]
        for i in range(len(crack_points) - 1):
            cv2.line(img, crack_points[i], crack_points[i + 1], (15, 15, 20), 2, cv2.LINE_AA)

        # Ground truth bbox for crack: (Class 0)
        min_x = min(p[0] for p in crack_points) - 4
        max_x = max(p[0] for p in crack_points) + 4
        min_y = min(p[1] for p in crack_points) - 4
        max_y = max(p[1] for p in crack_points) + 4
        bx = ((min_x + max_x) / 2) / width
        by = ((min_y + max_y) / 2) / height
        bw = (max_x - min_x) / width
        bh = (max_y - min_y) / height
        ground_truth_boxes.append((0, bx, by, bw, bh))

    # 8. Add Sensor Noise for Low-Light Mode
    if lighting_mode == "low_light":
        noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = noisy_img

    # Save image (convert RGB to BGR for OpenCV write)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr_img)

    # Write YOLO ground-truth annotations (.txt)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, bx, by, bw, bh in ground_truth_boxes:
            f.write(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    return str(path), str(label_path)


def generate_all_test_assets(output_dir: str = "dataset/sample_test_images") -> List[Tuple[str, str]]:
    """Populate the curated edge-case test set with high-fidelity track scenarios."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_scenarios = [
        # 1. Clean track baseline (zero defect)
        ("track_sample_clean.jpg", {"has_crack": False, "has_shadow": False, "lighting_mode": "day"}),
        # 2. Crack in outdoor shadow (CLAHE stress test)
        ("track_sample_crack_shadow.jpg", {"has_crack": True, "has_shadow": True, "lighting_mode": "day"}),
        # 3. Superficial rust (Distinguishing rust from cracks)
        ("track_sample_rust_surface.jpg", {"has_rust": True, "has_crack": False, "lighting_mode": "day"}),
        # 4. Missing fastener anomaly
        ("track_sample_missing_fastener.jpg", {"missing_fasteners": True, "has_crack": False, "lighting_mode": "day"}),
        # 5. Intense specular glare along rail crown
        ("track_sample_specular_glare.jpg", {"has_glare": True, "has_crack": True, "lighting_mode": "day"}),
        # 6. Low-light evening patrol with sensor noise
        ("track_sample_low_light.jpg", {"has_crack": True, "lighting_mode": "low_light"}),
    ]

    generated_assets = []
    for filename, kwargs in test_scenarios:
        img_path, lbl_path = generate_synthetic_track_image(str(out_dir / filename), **kwargs)
        generated_assets.append((img_path, lbl_path))

    return generated_assets


if __name__ == "__main__":
    assets = generate_all_test_assets()
    print(f"Successfully generated {len(assets)} curated test image & label pairs:")
    for img_p, lbl_p in assets:
        print(f" - Image: {img_p} | Labels: {lbl_p}")
