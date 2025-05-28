import cv2
import numpy as np

"""
===============================================================
🛰️ Satellite Image Detection and Classification Module
===============================================================

This file contains a set of functions for automatically analyzing images
captured by a satellite (or space probe) to determine their visual quality
and content.
"""

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

"""
====================================================================
📋 SATELLITE IMAGE VISUAL STATE DETECTION FUNCTIONS
====================================================================

This module contains the main functions used to analyze the quality
or visual content of images captured by a space probe or satellite.

⚙️ Each function takes:
    - `gray` : grayscale version of the image
    - `image_color` : original color image for annotation

✅ Returns:
    - `True` if the condition is detected
    - Automatically annotates the color image with a label

🧩 Available functions:
--------------------------
- is_black_image()        → Detects a black image (almost all pixels black)
- is_sunburn_image()      → Detects overexposure (very bright pixels)
- is_earth_image()        → Detects Earth's presence via bright pixels
- is_horizon_image()      → Detects a circular horizon line
- is_space_image()        → Detects deep space image (very dark)
- is_blurry_image()       → Detects blurriness via Laplacian variance
- is_noisy_image()        → Detects digital noise (difference with denoised version)
"""

# ============================================================
# 📌 Black image detection
# Improved: uses adaptive thresholds and masks for more robust detection

def is_black_image(gray, image_color):
    black_mask = gray < THRESHOLDS["black_value"]
    ratio = np.mean(black_mask)
    if ratio > THRESHOLDS["black_ratio"]:
        cv2.putText(image_color, "BLACK_IMAGE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["BLACK_IMAGE"], 2)
        return True
    return False

# ============================================================
# 📌 Overexposure detection (burned image)
# Improved: added kernel blur to reduce local bright noise influence

def is_sunburn_image(image_color):
    hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
    v_channel = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    burned_pixels = np.sum(v_channel > THRESHOLDS["sunburn_value"])
    ratio = burned_pixels / v_channel.size
    if ratio > THRESHOLDS["sunburn_ratio"]:
        cv2.putText(image_color, "SB_Sunburn", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["SB_Sunburn"], 2)
        return True
    return False

# ============================================================
# 📌 Earth detection
# Improved: applied Gaussian blur to avoid noise and uses adaptive ratio

def is_earth_image(gray, image_color):
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    bright_pixels = np.sum(smoothed > THRESHOLDS["earth_brightness"])
    ratio = bright_pixels / gray.size
    if ratio > THRESHOLDS["earth_ratio"]:
        cv2.putText(image_color, "ER_Earth", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["ER_Earth"], 2)
        return True
    return False

# ============================================================
# 📌 Horizon detection
def fit_circle_least_squares(xs, ys):
    """
    Ajuste un cercle sur un ensemble de points 2D en utilisant la méthode des moindres carrés.

    Paramètres :
        xs, ys : tableaux numpy contenant les coordonnées x et y des points d’un contour

    Retour :
        (cx, cy, radius) : centre et rayon du cercle ajusté

    Méthode :
        Résout une équation algébrique dérivée de la forme canonique d’un cercle :
        (x - cx)^2 + (y - cy)^2 = r^2
    """
    A = np.c_[2*xs, 2*ys, np.ones(len(xs))]  # Matrice des coefficients
    B = xs**2 + ys**2                        # Terme indépendant
    sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # Résolution du système
    cx, cy, c = sol
    radius = np.sqrt(c + cx**2 + cy**2)
    return int(cx), int(cy), int(radius)
def detect_horizon_line(image_gray):
    """
    Détecte l’arc d’horizon le plus plausible dans une image en niveaux de gris.

    Paramètre :
        image_gray : image 2D numpy (grayscale)

    Retour :
        (cx, cy, radius) : centre et rayon du cercle ajusté représentant l’horizon, ou None si échec
    """
    h, w = image_gray.shape  # Hauteur et largeur de l’image

    # 1. Réduction du bruit avec un flou gaussien
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # 2. Détection des bords avec l’algorithme de Canny
    edges = cv2.Canny(blurred, 50, 150)

    # 3. Recherche des contours externes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_score = -1  # Score max pour déterminer le meilleur contour

    # 4. Filtrage des contours
    for cnt in contours:
        if len(cnt) < 30:
            continue  # Ignore les petits contours

        xs = cnt[:, 0, 0]
        ys = cnt[:, 0, 1]
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        arc_len = cv2.arcLength(cnt, True)

        if w_box < 100 or h_box < 100 or arc_len < 300:
            continue  # Ignore les petits objets

        # Le contour doit toucher au moins un bord de l’image
        touches_border = (
            x <= 5 or y <= 5 or
            x + w_box >= w - 5 or
            y + h_box >= h - 5
        )
        if not touches_border:
            continue

        score = w_box * h_box
        if score > max_score:
            max_score = score
            best_contour = cnt

    if best_contour is None:
        return None  # Aucun contour valide trouvé

    # 5. Ajustement d’un cercle sur le contour sélectionné
    points = best_contour[:, 0, :]
    xs = points[:, 0]
    ys = points[:, 1]
    cx, cy, radius = fit_circle_least_squares(xs, ys)

    # 6. Validation du cercle (précision de l’ajustement)
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    error = np.abs(dists - radius)
    mean_error = np.mean(error)

    # 7. Rejet si :
    if mean_error > 15:       # Mauvais ajustement
        return None
    if radius < 100:          # Trop petit pour représenter la Terre
        return None
    if abs(cx - w // 2) < w * 0.25 and abs(cy - h // 2) < h * 0.25:
        return None  # Centre trop proche → disque entier de la Terre

    return cx, cy, radius

def is_horizon_image(gray, image_color):
    """
    Checks whether a given image contains a recognizable horizon.

    - gray: grayscale image (2D numpy array)
    - image_color: color image (will be annotated if horizon is detected)

    Returns: True if a horizon is detected, otherwise False.
    """
    result = detect_horizon_line(gray)  # Attempts to detect a circle representing the horizon

    if result is not None:
        cx, cy, radius = result  # Center coordinates and radius of the detected circle

        # Draws the detected circle on the color image
        cv2.circle(image_color, (cx, cy), radius, COLORS["HZ_Horizon"], 2)

        # Adds a text label on the image
        cv2.putText(image_color, "HZ_Horizon", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["HZ_Horizon"], 2)

        return True

    return False

# ============================================================
# 📌 Space detection
# Improved: stricter ratio analysis and contrast balancing

def is_space_image(gray, image_color):
    blurred = cv2.medianBlur(gray, 3)
    dark_ratio = np.sum(blurred < 60) / gray.size
    bright_ratio = np.sum(blurred > 200) / gray.size
    if dark_ratio > 0.85 and bright_ratio < 0.01:
        cv2.putText(image_color, "SP_Space", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["SP_Space"], 2)
        return True
    return False

# ============================================================
# 📌 Blur detection
# Adjusted: lowered threshold slightly to avoid over-detection

def is_blurry_image(gray, image_color):
    """
    Detects if an image is blurry using the variance of the Laplacian method.
    Lower variance = less sharp = potentially blurry image.
    A slightly reduced threshold allows more images to pass as acceptable.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)         # Highlight edges and fine details
    variance = laplacian.var()                          # Measure detail richness (sharpness)

    adjusted_threshold = THRESHOLDS["blur_threshold"] * 0.85  # Slightly lower threshold (~15% more tolerant)

    if variance < adjusted_threshold:
        cv2.putText(image_color, "BLUR", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["BLUR"], 2)
        return True
    return False

# ============================================================
# 📌 Digital noise detection
# Improved: adjustable noise thresholding and visual debugging

def is_noisy_image(gray, image_color):
    noise = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    diff = cv2.absdiff(gray, noise)
    noisy_pixels = np.sum(diff > 25)
    noise_ratio = noisy_pixels / gray.size
    if noise_ratio > THRESHOLDS["noise_threshold"] / 100:
        cv2.putText(image_color, "NOISE", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["NOISE"], 2)
        return True
    return False
