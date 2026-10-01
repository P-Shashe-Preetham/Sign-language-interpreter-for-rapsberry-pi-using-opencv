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

# Predefined Gestures & Full ASL Alphabet (26 Letters + Key Conversational Signs)
ASL_ALPHABET = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

CONVERSATIONAL_SIGNS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "I_LOVE_YOU",
    "THUMBS_UP",
    "THUMBS_DOWN"
]

# Total 33 Classes (Full ASL Alphabet + Common Everyday Signs)
SIGNS = ASL_ALPHABET + CONVERSATIONAL_SIGNS

# Camera Settings (Supports USB camera index or IP Camera RTSP/HTTP/MJPEG URL)
# Examples:
#   Local USB: 0
#   Phone IP Webcam: "http://192.168.1.50:8080/video"
#   DroidCam: "http://192.168.1.50:4747/video"
#   RTSP: "rtsp://admin:123456@192.168.1.50:554/stream"
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

def sanitize_camera_source(source):
    """Normalize and auto-fix camera sources (especially IP webcam URLs)."""
    if isinstance(source, int):
        return source
    if isinstance(source, str):
        source = source.strip().strip("'\"")
        if source.isdigit():
            return int(source)
        # If user passed an IP webcam URL without the stream endpoint:
        # e.g. "http://192.168.1.50:8080" -> needs "/video"
        if source.startswith("http://") or source.startswith("https://"):
            if ":8080" in source and not any(source.endswith(x) for x in ["/video", "/mjpeg", ".mjpg", "/shot.jpg"]):
                source = source.rstrip("/") + "/video"
            elif ":4747" in source and not any(source.endswith(x) for x in ["/video", "/mjpegfeed"]):
                source = source.rstrip("/") + "/video"
    return source

CAMERA_SOURCE = sanitize_camera_source(CAMERA_SOURCE)

def open_camera(source, width=640, height=480):
    """Robustly opens a camera source with low-latency settings."""
    import cv2
    clean_src = sanitize_camera_source(source)
    
    # Try default backend first
    cap = cv2.VideoCapture(clean_src)
    
    # If network stream fails to open, try with CAP_FFMPEG backend
    if not cap.isOpened() and isinstance(clean_src, str) and (clean_src.startswith("http") or clean_src.startswith("rtsp")):
        try:
            cap = cv2.VideoCapture(clean_src, cv2.CAP_FFMPEG)
        except Exception:
            pass
            
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, clean_src

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30



# MediaPipe Hand Tracking Parameters
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Classification & Temporal Smoothing Parameters
CONFIDENCE_THRESHOLD = 0.40    # Tuned for responsive 33-class inference
SMOOTHING_BUFFER_SIZE = 3      # Consecutive frames needed to stabilize prediction
SPEECH_COOLDOWN_SECONDS = 2.0  # Avoid repeated speaking of the same sign

# Text-to-Speech Settings
ENABLE_TTS = True
TTS_SPEECH_RATE = 150          # Words per minute
