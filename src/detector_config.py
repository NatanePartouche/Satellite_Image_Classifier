# ============================================================
# 📊 Image classification thresholds for preprocessing
# ============================================================

THRESHOLDS = {
    # --- 1. Black image ---
    # Nearly empty images, often caused by capture errors.
    "black_value": 30,              # Max value for a pixel to be considered black
    "black_ratio": 0.98,           # If ≥98% of pixels are black → BLACK_IMAGE

    # --- 2. Overexposure ---
    # Burnt image (too much light, direct sunlight, etc.)
    "sunburn_value": 240,          # Pixel value to be considered "burnt"
    "sunburn_ratio": 0.60,         # If ≥60% of pixels are burnt → SB_Sunburn

    # --- 3. Earth detection ---
    # Identify presence of continents or oceans.
    "earth_brightness": 100,       # Minimum brightness to identify Earth pixels
    "earth_ratio": 0.03,           # If ≥3% of pixels are Earth → ER_Earth

    # --- 4. Blur ---
    # Reject blurry images (vibration, failed focus).
    "blur_threshold": 100.0,       # Minimum Laplacian variance to consider image sharp

    # --- 5. Noise ---
    # Excessive noise due to faulty sensor or poor lighting.
    "noise_threshold": 30.0,       # Max allowed percentage of noisy pixels

    # --- 6. Empty space ---
    # Image mostly contains black, with little to no bright regions.
    "space_dark_ratio": 0.95,      # If ≥95% of pixels are very dark → SP_Space

    # --- 7. Earth too far ---
    # Earth is present but too small to be useful.
    "min_earth_area_ratio": 0.01,  # If <1% of image is Earth → ignored

    # --- 8. Internal structure ---
    # Camera captures internal satellite parts by mistake.
    "structure_brightness_threshold": 150,  # Brightness to detect metallic structures

    # --- 9. Duplicates ---
    # Avoid transmitting near-duplicate images.
    "similarity_hash_threshold": 10,  # Max Hamming distance to consider images similar

    # --- 10. Optical artifacts ---
    # Hardware defects on the image (scratches, dead pixels, etc.)
    "artifact_pixel_ratio": 0.02     # If ≥2% of pixels are affected → ARTIFACT
}

# ============================================================
# 🎨 Colors associated with each image category or status
# ============================================================

COLORS = {
    # === 🛰️ Main image categories ===
    "HZ_Horizon":      (0, 255, 0),       # Bright green: horizon detected
    "ER_Earth":        (255, 255, 0),     # Yellow: Earth detected
    "SB_Sunburn":      (0, 0, 255),       # Deep blue: overexposed image
    "SP_Space":        (150, 150, 255),   # Light blue: space only
    "BLACK_IMAGE":     (128, 128, 128),   # Medium gray: black image
    "BLUR":            (0, 128, 255),     # Medium blue: blurry image
    "NOISE":           (255, 128, 0),     # Bright orange: noisy image
    "STRUCTURE":       (255, 0, 255),     # Pink: internal structure
    "REDUNDANT":       (0, 255, 255),     # Cyan: duplicated image
    "ARTIFACT":        (255, 0, 0),       # Bright red: artifact detected

    # === ✅ Validation status ===
    "GOOD_IMAGE":      (0, 255, 128),     # Light green: valid image
    "REJECTED":        (50, 50, 50),      # Dark gray: rejected image
    "UNKNOWN":         (100, 100, 150),   # Gray/violet: unclassified

    # === 🧪 Debug and annotation tools ===
    "DEBUG_BOX":       (255, 255, 255),   # White: debug box
    "TEXT_ANNOTATION": (240, 240, 200),   # Pale yellow: overlay text color
    "CONTOUR":         (0, 0, 200),       # Dark blue: detected contour lines
    "CENTER_MARK":     (200, 0, 0),       # Dark red: optical center marker
    "FOCUS_REGION":    (0, 255, 64),      # Fluorescent green: focus zone
    "BORDER_WARNING":  (255, 200, 100),   # Light orange: close to image edge

    # === 🌍 Geographical/Natural elements ===
    "CLOUD_DETECTED":  (220, 220, 220),   # Light gray: clouds present
    "CLOUD_MASK":      (200, 200, 255),   # Pale blue: cloud/ground separation
    "WATER":           (0, 100, 255),     # Dark blue: water detection
    "LAND":            (50, 200, 50),     # Forest green: land detection
    "ICE_SNOW":        (230, 250, 255),   # Bluish white: snow or ice
    "STARS":           (255, 255, 180),   # Pale yellow: stars detected

    # === ⚙️ Hardware-specific defects ===
    "SAT_REFLECTION":  (255, 180, 180),   # Light pink: satellite reflection
    "SOLAR_GLINT":     (255, 255, 102),   # Bright yellow: direct solar reflection
    "LENS_SHADOW":     (80, 80, 80),      # Dark gray: lens internal shadow
    "DEAD_PIXEL_ZONE": (255, 0, 128),     # Dark pink: dead pixels detected

    # === 🔁 Workflow stages ===
    "PRE_FILTERED":    (180, 180, 180),   # Light gray: post-preprocessing
    "POST_PROCESS":    (0, 180, 255),     # Cyan: ready for transmission
    "MANUAL_REVIEW":   (255, 100, 0),     # Reddish orange: manual review needed
    "AUTO_APPROVED":   (0, 255, 200),     # Turquoise green: auto-approved

    # === 🧯 Analysis/diagnostics tracking ===
    "INCOMPLETE":      (200, 200, 100),   # Sandy yellow: incomplete analysis
    "FAILED_ANALYSIS": (255, 50, 50),     # Light red: failed processing
    "PENDING":         (128, 128, 255),   # Mauve blue: awaiting analysis
    "SKIPPED":         (100, 100, 100)    # Dark gray: image skipped by early filtering
}
