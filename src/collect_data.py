import cv2
import csv
import time
import numpy as np
from pathlib import Path

try:
    from . import config
    from .hand_tracker import HandTracker
except ImportError:
    import config
    from hand_tracker import HandTracker


def main():
    print("=" * 60)
    print(" SIGN LANGUAGE DATASET COLLECTOR ")
    print("=" * 60)
    print(f"Target Signs ({len(config.SIGNS)}): {', '.join(config.SIGNS)}")
    print("Controls:")
    print("  [SPACE] : Start / Pause recording samples for active sign")
    print("  [N]     : Switch to NEXT sign")
    print("  [P]     : Switch to PREVIOUS sign")
    print("  [Q]     : Quit and save")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="Sign Language Data Collector with IP/USB Camera support")
    parser.add_argument(
        "-s", "--source",
        default=str(config.CAMERA_SOURCE),
        help="Camera source: USB device index (e.g. 0, 1) or IP camera URL (e.g. http://192.168.1.50:8080/video or rtsp://...)"
    )
    args = parser.parse_args()

    cam_source = int(args.source) if args.source.isdigit() else args.source

    # Prepare CSV file
    file_exists = config.CSV_PATH.exists()
    header = ["label"] + [f"feat_{i}" for i in range(63)]
    
    csv_file = open(config.CSV_PATH, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists or config.CSV_PATH.stat().st_size == 0:
        writer.writerow(header)
        csv_file.flush()

    # Initialize tracker and camera
    tracker = HandTracker()
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at: {cam_source}")
        return


    sign_idx = 0
    recording = False
    samples_per_sign = {s: 0 for s in config.SIGNS}

    # Count existing samples in CSV if any
    if file_exists:
        try:
            with open(config.CSV_PATH, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                next(r, None)  # Skip header
                for row in r:
                    if row and row[0] in samples_per_sign:
                        samples_per_sign[row[0]] += 1
        except Exception:
            pass

    TARGET_SAMPLES = 100

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame.")
                break

            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            features, bbox, hand_landmarks = tracker.process_frame(frame)
            tracker.draw_landmarks(frame, hand_landmarks)

            active_sign = config.SIGNS[sign_idx]

            # Save sample if recording and hand is detected
            if recording and features is not None:
                writer.writerow([active_sign] + list(features))
                csv_file.flush()
                samples_per_sign[active_sign] += 1
                time.sleep(0.03)  # Short delay between recorded frames (~30 fps cap)

            # Draw HUD
            # Background banner
            cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)
            
            # Active sign & status
            color = (0, 255, 0) if recording else (0, 165, 255)
            status_text = "RECORDING" if recording else "PAUSED"
            cv2.putText(frame, f"Sign: {active_sign} [{sign_idx+1}/{len(config.SIGNS)}]", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status_text} | Samples: {samples_per_sign[active_sign]}/{TARGET_SAMPLES}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Sign Language Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Toggle recording
                recording = not recording
            elif key == ord('n') or key == ord('N'):  # Next sign
                recording = False
                sign_idx = (sign_idx + 1) % len(config.SIGNS)
            elif key == ord('p') or key == ord('P'):  # Previous sign
                recording = False
                sign_idx = (sign_idx - 1) % len(config.SIGNS)

    finally:
        cap.release()
        tracker.close()
        csv_file.close()
        cv2.destroyAllWindows()
        print("\nData collection finished. Dataset saved to:", config.CSV_PATH)


if __name__ == "__main__":
    main()
