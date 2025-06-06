# 📦 Import standard libraries and custom detection functions
import os                 # Used for file and directory operations
import cv2                # OpenCV library for image processing
import csv
import shutil
from PIL import Image     # Pillow library for image manipulation and compression
from pathlib import Path  # For handling file paths in an OS-independent way

# Import image quality detection functions from a local module
from image_quality_detector import (
    is_black_image,       # Detects if the image is fully black
    is_sunburn_image,     # Detects overexposed (sunburned) images
    is_earth_image,       # Detects presence of Earth in the image
    is_horizon_image,     # Detects if horizon is visible in the image
    is_space_image,       # Detects images taken in space (no Earth)
    is_blurry_image,      # Detects blurry images
    is_noisy_image        # Detects noisy images
)

# 📁 Define input and output directories
INPUT_DIR = Path("DataSet/Input_Image")  # Folder containing unprocessed/raw images
OUTPUT_ROOT = Path("DataSet/Output_Image")  # Base folder where output will be stored
GOOD_DIR = OUTPUT_ROOT / "Good_Image"  # Folder for valid, high-quality images
BAD_DIR = OUTPUT_ROOT / "Bad_Image"  # Folder for images that fail quality checks

# ✅ Create the output folders if they don't exist
GOOD_DIR.mkdir(parents=True, exist_ok=True)
BAD_DIR.mkdir(parents=True, exist_ok=True)

def filter_by_quality(image_path):
    """
    Analyze a satellite image using a set of detection rules and
    save it in the appropriate folder based on the detected quality issues.
    """
    image = cv2.imread(str(image_path))  # Load the image using OpenCV
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for analysis
    annotated = image.copy()  # Copy of the original image for potential annotations
    labels = set()  # Set to store detected issues

    # ============================================================
    # 🧠 Run quality checks using predefined detection functions
    # ============================================================
    if is_black_image(gray, annotated):
        labels.add("BLACK_IMAGE")

    if is_sunburn_image(annotated):
        labels.add("SB_Sunburn")

    if is_space_image(gray, annotated):
        labels.add("SP_Space")

    if is_blurry_image(gray, annotated):
        labels.add("BLUR")

    if is_noisy_image(gray, annotated):
        labels.add("NOISE")

    if is_earth_image(gray, annotated):
        labels.add("ER_Earth")
    else:
        labels.add("NO_EARTH")

    if is_horizon_image(gray, annotated):
        labels.add("HZ_Horizon")
    else:
        labels.add("NO_HORIZON")

    # ============================================================
    # 🧾 Decide where to save the image based on detected labels
    # ============================================================

    # Define which labels are considered defects
    defects = {
        "BLACK_IMAGE", "SB_Sunburn", "SP_Space", "BLUR", "NOISE", "NO_EARTH", "NO_HORIZON"
    }

    # If Earth and Horizon are present and there are no defects → it's a good image
    if "ER_Earth" in labels and "HZ_Horizon" in labels and labels.isdisjoint(defects):
        labels = {"GOOD_IMAGE"}
        save_path = GOOD_DIR / image_path.name
    else:
        # Otherwise, it's a bad image; include reasons in the filename
        label_str = "+".join(sorted(labels))
        save_path = BAD_DIR / f"{label_str}_{image_path.name}"

    # 💾 Save the image in the determined folder
    cv2.imwrite(str(save_path), annotated)
    print(f"✅ {image_path.name} → {', '.join(labels)}")

def compress_images(input_folder, output_folder, report_csv=None):
    """
    Compress images with dynamic quality based on original size.
    Displays size before/after and optionally writes a CSV report.

    Parameters:
        input_folder (str): Folder with original images
        output_folder (str): Destination folder for compressed images
        report_csv (str): Optional path to save a compression report (CSV)
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Store data for optional CSV report
    report_data = []

    # Initial log headers
    print("\n🔧 Starting image compression...")
    print(f"📁 Input folder : {input_folder}")
    print(f"📁 Output folder: {output_folder}\n")

    # Display the table headers for the compression log
    print(f"{'📄 Filename':<35} {'Original (KB)':>14} {'Compressed (KB)':>17} {'Ratio':>8} {'Quality':>9}  Status")
    print("-" * 90)

    # Loop through all files in the input folder
    for filename in sorted(os.listdir(input_folder)):
        # Skip non-image files
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Define full paths for input and output files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Get the size of the original file in KB
            original_size = os.path.getsize(input_path) / 1024

            # Skip files that are too small to justify compression
            if original_size < 30:
                shutil.copy2(input_path, output_path)
                print(
                    f"{filename:<35} {original_size:>14.1f} {original_size:>17.1f} {'100%':>8} {'-':>9}  ⚠️ Copied (too small)")
                report_data.append([
                    filename,
                    round(original_size, 1),
                    round(original_size, 1),
                    "100%",
                    "-"
                ])
                continue

            # Open the image using Pillow
            with Image.open(input_path) as img:
                # Convert to RGB if the image has alpha/transparency or a palette mode
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Dynamically assign a compression quality based on the original size
                if original_size > 3000:
                    quality = 55
                elif original_size > 2000:
                    quality = 65
                elif original_size > 1000:
                    quality = 75
                else:
                    quality = 85

                # Save the image to the output path in JPEG format using the chosen quality
                img.save(output_path, "JPEG", quality=quality, optimize=True)

            # Measure the size of the compressed file
            compressed_size = os.path.getsize(output_path) / 1024
            # Compute the compression ratio (compressed / original)
            ratio = compressed_size / original_size

            # Print a formatted summary line for this image
            print(f"{filename:<35} {original_size:>14.1f} {compressed_size:>17.1f} {ratio:>7.0%} {quality:>9}  ✅ Compressed")

            # Store information for report generation
            report_data.append([
                filename,
                round(original_size, 1),
                round(compressed_size, 1),
                f"{ratio:.0%}",
                quality
            ])

        except Exception as e:
            # Print an error message if anything fails during processing
            print(f"{filename:<35} {'ERR':>14} {'ERR':>17} {'-':>8} {'-':>9}  ❌ Error: {e}")

    # If a CSV report is requested, write it to the specified file
    if report_csv:
        with open(report_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Original Size (KB)", "Compressed Size (KB)", "Compression Ratio", "Quality"])
            writer.writerows(report_data)
        print(f"\n📄 Compression report saved to: {report_csv}")

    # Final log message
    print("\n✅ Compression process complete.\n")


# ▶️ Main entry point when the script is run directly
if __name__ == "__main__":
    """
    Process all images in INPUT_DIR:
    1. Filter and classify each image into GOOD or BAD folders.
    2. Compress all GOOD images and save them in the Final_Output folder.
    """
    # 🔍 Step 1: Run classification
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = INPUT_DIR / filename
            filter_by_quality(path)

    # 💾 Step 2: Compress only GOOD images
    compress_images(
        input_folder=str(GOOD_DIR),
        output_folder=str(OUTPUT_ROOT / "Final_Output"),
    )
