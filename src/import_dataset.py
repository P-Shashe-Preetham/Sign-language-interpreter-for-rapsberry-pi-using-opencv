"""
Sign Language Dataset Importer & Converter
Supports:
  1. Importing external CSV datasets (e.g. Kaggle ASL Alphabet MediaPipe CSVs)
  2. Extracting landmarks from a directory of sign language images using MediaPipe Hands
  3. Generating full 26-letter ASL Alphabet dataset
"""
import os
import sys
import csv
import argparse
import numpy as np
from pathlib import Path

try:
    from . import config
    from .hand_tracker import HandTracker
except ImportError:
    import config
    from hand_tracker import HandTracker


def import_external_csv(csv_path: str):
    """Imports an external CSV file containing sign language landmarks."""
    p = Path(csv_path)
    if not p.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return False

    print(f"[INFO] Importing and parsing external CSV: {csv_path}...")
    imported_rows = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 43: # Minimum 21 landmarks (x, y) or (x, y, z)
                continue
            label = str(row[0]).strip().upper()
            try:
                feats = [float(x) for x in row[1:64]]
                if len(feats) == 63:
                    imported_rows.append([label] + feats)
            except ValueError:
                continue

    if not imported_rows:
        print("[ERROR] No valid 63-feature landmark rows found in CSV.")
        return False

    # Append to config.CSV_PATH
    with open(config.CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(imported_rows)

    print(f"[SUCCESS] Appended {len(imported_rows)} samples to {config.CSV_PATH}")
    return True


def extract_from_image_folder(folder_path: str):
    """
    Scans a folder structured by class name (e.g., folder/HELLO/img1.jpg, folder/PEACE/img2.jpg)
    and extracts 63 MediaPipe normalized landmarks.
    """
    import cv2
    base_folder = Path(folder_path)
    if not base_folder.exists() or not base_folder.is_dir():
        print(f"[ERROR] Directory not found: {folder_path}")
        return False

    tracker = HandTracker()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    extracted_rows = []

    print(f"[INFO] Scanning subdirectories in {base_folder} for sign images...")
    subdirs = [d for d in base_folder.iterdir() if d.is_dir()]
    if not subdirs:
        # Check if images are directly in folder
        subdirs = [base_folder]

    for category_dir in subdirs:
        label = category_dir.name.upper()
        image_files = [f for f in category_dir.iterdir() if f.suffix.lower() in valid_exts]
        print(f"  Processing category '{label}' ({len(image_files)} images)...")

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feats, _, _ = tracker.process_frame(img)
            if feats is not None:
                extracted_rows.append([label] + list(feats))

    tracker.close()

    if not extracted_rows:
        print("[WARNING] No hands were detected in the provided images.")
        return False

    with open(config.CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(extracted_rows)

    print(f"[SUCCESS] Extracted {len(extracted_rows)} hand landmark samples to {config.CSV_PATH}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sign Language Dataset Importer")
    parser.add_argument("--csv", help="Path to an external landmarks CSV file to import")
    parser.add_argument("--images", help="Path to folder of images organized by label (e.g. folder/A/1.jpg)")
    parser.add_argument("--generate", action="store_true", help="Generate full anatomical 10-gesture ground truth")
    args = parser.parse_args()

    if args.csv:
        import_external_csv(args.csv)
    elif args.images:
        extract_from_image_folder(args.images)
    elif args.generate:
        from .generate_anatomical_dataset import generate_anatomical_dataset
        generate_anatomical_dataset()
    else:
        print("Usage:")
        print("  python -m src.import_dataset --generate          # Generate anatomical signs")
        print("  python -m src.import_dataset --csv path.csv       # Import Kaggle CSV")
        print("  python -m src.import_dataset --images path/to/dir # Extract from photo directory")
        return

    # Train model automatically after import
    print("[INFO] Re-training model on newly imported dataset...")
    try:
        from .train_model import train_model
        train_model()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")


if __name__ == "__main__":
    main()
