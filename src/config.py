import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Ensure runtime directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset & Model Filepaths
CSV_PATH = DATASET_DIR / "landmarks.csv"
MODEL_PATH = MODELS_DIR / "sign_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "labels.joblib"

# Predefined Gestures / Signs (10 Predefined Classes)
SIGNS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "I_LOVE_YOU",
    "PEACE",
    "OK",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "FIST"
]

# Camera Settings (Optimized for Raspberry Pi 4B)
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# MediaPipe Hand Tracking Parameters
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6

# Classification & Temporal Smoothing Parameters
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_BUFFER_SIZE = 5      # Consecutive frames needed to stabilize prediction
SPEECH_COOLDOWN_SECONDS = 2.0  # Avoid repeated speaking of the same sign

# Text-to-Speech Settings
ENABLE_TTS = True
TTS_SPEECH_RATE = 150          # Words per minute
