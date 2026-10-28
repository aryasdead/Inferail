"""Generate realistic synthetic railway track images for offline testing and preprocessing validation."""

from pathlib import Path
import numpy as np
import cv2


def generate_synthetic_track_image(
    output_path: str,
    width: int = 1280,
    height: int = 720,
    has_crack: bool = True,
    has_shadow: bool = True,
) -> str:
    """Generate a realistic test image of a railway track segment with simulated defects.

    Args:
        output_path: Path to save the image (PNG/JPG).
        width: Image width in pixels.
        height: Image height in pixels.
        has_crack: Whether to draw a simulated hairline rail fracture.
        has_shadow: Whether to simulate outdoor directional sun shadows.

    Returns:
        str: Absolute path of saved test image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Base ballast texture (gravel - noisy brownish gray)
    np.random.seed(42)
    ballast = np.random.normal(loc=110, scale=35, size=(height, width, 3)).clip(0, 255).astype(np.uint8)
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
        # Concrete/wood sleeper color
        tie_color = np.array([75, 70, 65], dtype=np.uint8)
        img[y1:y2, :] = (img[y1:y2, :] * 0.3 + tie_color * 0.7).astype(np.uint8)

    # 3. Add Steel Rails (Two vertical parallel steel beams)
    rail_w = 60
    left_rail_x = int(width * 0.32)
    right_rail_x = int(width * 0.68)

    for rx in [left_rail_x, right_rail_x]:
        # Rail base / flange (dark metallic)
        cv2.rectangle(img, (rx - 25, 0), (rx + 25, height), (60, 60, 65), -1)
        # Rail crown / head (bright polished steel)
        cv2.rectangle(img, (rx - 15, 0), (rx + 15, height), (175, 175, 180), -1)
        # Center shine specular line
        cv2.line(img, (rx - 2, 0), (rx - 2, height), (220, 220, 225), 3)

        # Fastener clips on ties
        for y in range(20, height, tie_spacing):
            fy = y + tie_height // 2
            # Left fastener clip
            cv2.circle(img, (rx - 32, fy), 8, (45, 45, 50), -1)
            # Right fastener clip
            cv2.circle(img, (rx + 32, fy), 8, (45, 45, 50), -1)

    # 4. Add Directional Outdoor Sun Shadow across track
    if has_shadow:
        shadow_mask = np.ones((height, width), dtype=np.float32)
        # Triangular shadow diagonal from top-left to middle-right
        pts = np.array([[0, 0], [width // 2, 0], [width, height], [0, height // 2]], np.int32)
        cv2.fillPoly(shadow_mask, [pts], 0.38)
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * shadow_mask).astype(np.uint8)

    # 5. Add Simulated Surface Crack on the left rail head
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

    # Save image (convert RGB to BGR for OpenCV write)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr_img)
    return str(path)


if __name__ == "__main__":
    out_dir = Path("dataset/sample_test_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    f1 = generate_synthetic_track_image(str(out_dir / "track_sample_crack_shadow.jpg"), has_crack=True, has_shadow=True)
    f2 = generate_synthetic_track_image(str(out_dir / "track_sample_clean.jpg"), has_crack=False, has_shadow=False)
    print(f"Generated sample images:\n1. {f1}\n2. {f2}")
