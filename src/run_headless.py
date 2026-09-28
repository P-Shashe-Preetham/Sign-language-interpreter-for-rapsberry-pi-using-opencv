import cv2
import time
import numpy as np
from collections import deque
import joblib
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from . import config
    from .hand_tracker import HandTracker
    from .tts_engine import TTSEngine
except ImportError:
    import config
    from hand_tracker import HandTracker
    from tts_engine import TTSEngine

# Global frame buffer for HTTP MJPEG web streaming
latest_jpeg = None
frame_lock = threading.Lock()


class VideoStreamHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to stream live annotated frames to any web browser."""
    def do_GET(self):
        global latest_jpeg
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
                <head>
                    <title>Sign Language Recognition - Live Stream</title>
                    <style>
                        body { background: #121212; color: #fff; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
                        h1 { color: #00e676; margin-bottom: 5px; }
                        p { color: #aaa; margin-top: 0; }
                        img { border: 3px solid #333; border-radius: 8px; max-width: 90vw; height: auto; }
                    </style>
                </head>
                <body>
                    <h1>Real-Time Sign Language Recognition</h1>
                    <p>Raspberry Pi 4B Headless Live Feed</p>
                    <img src="/video_feed" />
                </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with frame_lock:
                    if latest_jpeg is None:
                        time.sleep(0.03)
                        continue
                    frame_bytes = latest_jpeg
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-type", "image/jpeg")
                    self.send_header("Content-length", str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.04)  # ~25 fps cap
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            self.send_error(404)


def start_web_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), VideoStreamHandler)
    print(f"[INFO] Live Web Stream available at: http://localhost:{port} (or http://signpi.local:{port})")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()


def main():
    global latest_jpeg

    print("=" * 60)
    print(" HEADLESS SIGN LANGUAGE RECOGNITION (Terminal + Web Stream) ")
    print("=" * 60)

    if not config.MODEL_PATH.exists() or not config.LABEL_ENCODER_PATH.exists():
        print(f"[ERROR] Trained model not found at {config.MODEL_PATH}.")
        print("Please train the model first by running: python -m src.train_model")
        return

    classifier = joblib.load(config.MODEL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

    tracker = HandTracker()
    tts = TTSEngine(enabled=config.ENABLE_TTS)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {config.CAMERA_INDEX}.")
        return

    # Start browser stream
    start_web_server(port=8080)

    prediction_buffer = deque(maxlen=config.SMOOTHING_BUFFER_SIZE)
    last_confirmed_sign = "WAITING..."
    last_confidence = 0.0

    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    print("[INFO] Headless engine running! Press Ctrl+C in terminal to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()

            features, bbox, hand_landmarks = tracker.process_frame(frame)
            tracker.draw_landmarks(frame, hand_landmarks)

            if features is not None:
                probs = classifier.predict_proba([features])[0]
                best_idx = np.argmax(probs)
                best_confidence = probs[best_idx]
                predicted_label = label_encoder.classes_[best_idx]

                if best_confidence >= config.CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(predicted_label)
                else:
                    prediction_buffer.append(None)

                valid_preds = [p for p in prediction_buffer if p is not None]
                if valid_preds:
                    from collections import Counter
                    most_common, count = Counter(valid_preds).most_common(1)[0]
                    if count >= (config.SMOOTHING_BUFFER_SIZE * 0.6):
                        if most_common != last_confirmed_sign:
                            last_confirmed_sign = most_common
                            last_confidence = best_confidence
                            tts.speak(most_common)
                            print(f"\r>>> RECOGNIZED SIGN: {last_confirmed_sign} ({last_confidence*100:.1f}%) | FPS: {fps:.1f}", end="", flush=True)
                        else:
                            last_confidence = best_confidence

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{last_confirmed_sign} ({last_confidence*100:.0f}%)",
                                (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                prediction_buffer.clear()
                if last_confirmed_sign != "NO HAND":
                    last_confirmed_sign = "NO HAND"
                    last_confidence = 0.0

            # Render HUD for browser stream
            cv2.rectangle(frame, (0, 0), (w, 75), (15, 15, 15), -1)
            text_color = (0, 255, 0) if last_confirmed_sign not in ["WAITING...", "NO HAND"] else (180, 180, 180)
            cv2.putText(frame, f"SIGN: {last_confirmed_sign}", (15, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, text_color, 2)
            cv2.putText(frame, f"Confidence: {last_confidence*100:.1f}% | FPS: {fps:.1f}", (15, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Compress for web stream
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with frame_lock:
                latest_jpeg = encoded.tobytes()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        tracker.close()
        tts.close()
        print("[INFO] Cleanup complete.")


if __name__ == "__main__":
    main()
