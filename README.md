# Real-Time Sign Language Recognition Using Raspberry Pi 4B

An embedded computer vision and machine learning project that detects and translates sign language gestures in real time using a camera on a Raspberry Pi 4B.

---

## 🎯 Architecture & Pipeline

```
Camera (USB / CSI)
   │
   ▼
OpenCV (Frame Capture & Preprocessing)
   │
   ▼
MediaPipe Hands (21 3D Landmark Detection)
   │
   ▼
Feature Normalization (Translation & Scale Invariant)
   │  • Shift wrist to (0,0,0)
   │  • Scale by max Euclidean distance
   ▼
Lightweight ML Classifier (RandomForest / <1ms Latency)
   │
   ▼
Temporal Consensus Smoothing (De-bouncing)
   │
   ├──▶ Visual HUD Overlay (Bounding Box, Landmarks, Text)
   ├──▶ Web Stream (View live feed via Browser on Port 8080)
   └──▶ Text-to-Speech (TTS Voice Announcement via pyttsx3 / espeak)
```

---

## 📁 Project Directory Structure

```
Proto-backups/
├── dataset/
│   └── landmarks.csv               # Extracted landmark feature vectors & labels
├── models/
│   ├── sign_classifier.joblib      # Trained lightweight model
│   └── labels.joblib               # Label encoder mappings
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central settings, classes, thresholds
│   ├── hand_tracker.py             # MediaPipe tracker & normalization
│   ├── collect_data.py             # Interactive dataset collection tool
│   ├── train_model.py              # ML model trainer & evaluator
│   ├── tts_engine.py               # Asynchronous non-blocking TTS
│   ├── realtime_inference.py       # Desktop GUI real-time runner
│   └── run_headless.py             # Headless Pi runner (Terminal + Web Browser Stream)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation

#### On Windows (or Raspberry Pi with Python venv):
```bash
pip install -r requirements.txt
```

#### On Raspberry Pi 4B (Debian Bookworm):
```bash
sudo apt update
sudo apt install -y python3-opencv python3-pip espeak
pip install --break-system-packages -r requirements.txt
```

---

### 2. Dataset Collection (Optional / Custom Signs)
To record your own hand gestures using your webcam:
```bash
python -m src.collect_data
```
- Press **`SPACE`** to start/pause recording samples for the active sign.
- Press **`N`** for next sign, **`P`** for previous sign.
- Press **`Q`** to exit.
- Frames are automatically extracted into `dataset/landmarks.csv`.

---

### 3. Training the Model
Train the lightweight classifier:
```bash
python -m src.train_model
```
> **Tip:** To test the pipeline immediately without manual recording, run with `--synthetic`:
> ```bash
> python -m src.train_model --synthetic
> ```
> This creates a baseline dataset, trains the model, and saves it to `models/sign_classifier.joblib`.

---

### 4. Running Real-Time Recognition

#### Option A: Headless Mode on Raspberry Pi (Recommended)
Since the Raspberry Pi runs headless without a monitor, run:
```bash
# Using a local USB camera (index 0)
python3 -m src.run_headless

# OR using an IP Camera / Phone Webcam:
python3 -m src.run_headless --source "http://192.168.1.50:8080/video"
```
- **Terminal:** Prints live recognized signs, confidence, and FPS directly in your SSH shell.
- **Voice:** Speaks the recognized sign through the Pi's audio output / USB speaker.
- **Browser Live View:** Open any web browser on your laptop and navigate to:
  ```
  http://<PI_IP_ADDRESS>:8080
  ```
  You will see the live camera feed with skeleton tracking and recognized sign overlays!

#### Option B: GUI Window Mode (On Laptop or Pi with Monitor)
```bash
# Local USB camera:
python -m src.realtime_inference

# OR IP Camera:
python -m src.realtime_inference --source "http://192.168.1.50:8080/video"
```
Opens an OpenCV window showing real-time hand skeleton, bounding box, recognized text, and live FPS counter. Press **`Q`** to quit.

---

## 📱 Using Your Phone as an IP Camera

If you do not have a physical USB webcam for the Raspberry Pi, you can turn your smartphone into an IP camera:

1. **Install an IP Webcam app on your phone:**
   - **Android:** Download **"IP Webcam"** by Pavel Khlebovich from Google Play Store.
   - **iOS / Android:** Download **"DroidCam"** or **"iVCam"**.
2. **Start the server in the app:**
   - Open the app and tap **"Start Server"**.
   - Note the URL displayed on the phone screen (e.g. `http://192.168.137.45:8080` or `http://192.168.1.50:4747`).
3. **Run the script with the IP Camera stream URL:**
   - For **IP Webcam** (Android): add `/video` to the URL:
     ```bash
     python3 -m src.run_headless --source "http://<PHONE_IP>:8080/video"
     ```
   - For **DroidCam**:
     ```bash
     python3 -m src.run_headless --source "http://<PHONE_IP>:4747/video"
     ```
   - For **RTSP security camera**:
     ```bash
     python3 -m src.run_headless --source "rtsp://username:password@<CAMERA_IP>:554/stream"
     ```
   - You can also set `CAMERA_SOURCE` in `src/config.py` permanently!


---

## ⚙️ Supported Signs: Full ASL Alphabet (26 Letters: A - Z)

The system translates fingerspelling for all 26 letters of the American Sign Language (ASL) alphabet:

`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`


