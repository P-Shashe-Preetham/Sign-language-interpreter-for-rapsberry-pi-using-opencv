import cv2
import time
import numpy as np
from collections import deque
import joblib

try:
    from . import config
    from .hand_tracker import HandTracker
    from .tts_engine import TTSEngine
except ImportError:
    import config
    from hand_tracker import HandTracker
    from tts_engine import TTSEngine


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument(
        "-s", "--source",
        default=str(config.CAMERA_SOURCE),
        help="Camera source: USB device index (e.g. 0, 1) or IP camera URL (e.g. http://192.168.1.50:8080/video or rtsp://...)"
    )
    args = parser.parse_args()

    cam_source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print(" REAL-TIME SIGN LANGUAGE RECOGNITION (Raspberry Pi 4B) ")
    print("=" * 60)
    print(f"[INFO] Connecting to camera source: {cam_source}")

    # Check if model exists
    if not config.MODEL_PATH.exists() or not config.LABEL_ENCODER_PATH.exists():
        print(f"[ERROR] Trained model not found at {config.MODEL_PATH}.")
        print("Please train the model first by running: python -m src.train_model")
        return

    print("[INFO] Loading trained model and label encoder...")
    classifier = joblib.load(config.MODEL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    print(f"[INFO] Classes: {list(label_encoder.classes_)}")

    # Initialize modules
    tracker = HandTracker()
    tts = TTSEngine(enabled=config.ENABLE_TTS)

    # Initialize Camera
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at: {cam_source}")
        return


    # Buffer for temporal smoothing to avoid rapid flickering
    prediction_buffer = deque(maxlen=config.SMOOTHING_BUFFER_SIZE)
    last_confirmed_sign = "WAITING..."
    last_confidence = 0.0

    # FPS counter variables
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    print("[INFO] Starting real-time recognition loop. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed.")
                break

            # Mirror image for intuitive interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()

            # Process hand landmarks
            features, bbox, hand_landmarks = tracker.process_frame(frame)
            tracker.draw_landmarks(frame, hand_landmarks)

            if features is not None:
                # Classify hand landmarks
                probs = classifier.predict_proba([features])[0]
                best_idx = np.argmax(probs)
                best_confidence = probs[best_idx]
                predicted_label = label_encoder.classes_[best_idx]

                if best_confidence >= config.CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(predicted_label)
                else:
                    prediction_buffer.append(None)

                # Temporal smoothing: check if consensus exists in buffer
                valid_preds = [p for p in prediction_buffer if p is not None]
                if valid_preds:
                    from collections import Counter
                    most_common, count = Counter(valid_preds).most_common(1)[0]
                    # Require at least 60% of the buffer to agree
                    if count >= (config.SMOOTHING_BUFFER_SIZE * 0.6):
                        if most_common != last_confirmed_sign:
                            last_confirmed_sign = most_common
                            last_confidence = best_confidence
                            # Non-blocking voice announcement
                            tts.speak(most_common)
                        else:
                            last_confidence = best_confidence

                # Draw bounding box and label
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    box_color = (0, 255, 0) if last_confidence >= config.CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"{last_confirmed_sign} ({last_confidence*100:.0f}%)",
                                (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            else:
                prediction_buffer.clear()
                last_confirmed_sign = "NO HAND"
                last_confidence = 0.0

            # Render HUD Overlay
            # Top Banner
            cv2.rectangle(frame, (0, 0), (w, 80), (15, 15, 15), -1)
            cv2.line(frame, (0, 80), (w, 80), (50, 50, 50), 2)

            # Recognized Sign Text
            text_color = (0, 255, 0) if last_confirmed_sign not in ["WAITING...", "NO HAND"] else (180, 180, 180)
            cv2.putText(frame, f"SIGN: {last_confirmed_sign}", (15, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, text_color, 2)

            # Confidence & FPS Info
            cv2.putText(frame, f"Confidence: {last_confidence*100:.1f}% | FPS: {fps:.1f}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Display Frame
            cv2.imshow("Sign Language Recognition (Raspberry Pi)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        tracker.close()
        tts.close()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully.")


if __name__ == "__main__":
    main()
