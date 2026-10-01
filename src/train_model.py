import os
import csv
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

try:
    from . import config
except ImportError:
    import config


def generate_synthetic_dataset():
    """Generates anatomically authentic baseline dataset for initial training."""
    try:
        from .generate_anatomical_dataset import generate_anatomical_dataset
    except ImportError:
        from generate_anatomical_dataset import generate_anatomical_dataset
    generate_anatomical_dataset(samples_per_class=150)


def train_model():
    if not config.CSV_PATH.exists() or config.CSV_PATH.stat().st_size == 0:
        print(f"[WARNING] No dataset found at {config.CSV_PATH}.")
        print("[INFO] Generating anatomical ground-truth dataset so you can run the pipeline immediately.")
        generate_synthetic_dataset()

    print(f"[INFO] Loading dataset from: {config.CSV_PATH}")
    labels = []
    features = []

    with open(config.CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])

    X = np.array(features, dtype=np.float32)
    y_raw = np.array(labels)

    unique_classes = np.unique(y_raw)
    print(f"[INFO] Found {len(X)} samples across {len(unique_classes)} classes: {list(unique_classes)}")

    if len(unique_classes) < 2:
        print("[ERROR] At least 2 different sign classes are required to train a classifier.")
        return

    # Encode class labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    print("[INFO] Training Lightweight RandomForestClassifier...")

    # RandomForest is ideal for embedded edge inference:
    # - Invariant to feature scaling
    # - Negligible inference latency (<1ms on Raspberry Pi 4)
    # - Robust against sensor noise
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)

    # Evaluation
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("=" * 60)
    print(f" MODEL EVALUATION ACCURACY: {acc * 100:.2f}%")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save trained model and label encoder
    joblib.dump(classifier, config.MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    print(f"[SUCCESS] Model saved to: {config.MODEL_PATH}")
    print(f"[SUCCESS] Label encoder saved to: {config.LABEL_ENCODER_PATH}")
    print("You can now run 'python -m src.realtime_inference' to test live recognition!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic samples before training")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset()

    train_model()
