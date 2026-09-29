"""
Quick Camera Diagnostic & Test Tool
Tests USB webcams or IP camera URLs (Phone IP Webcam, DroidCam, RTSP)
"""
import sys
import time
import cv2

try:
    from src import config
    default_source = config.CAMERA_SOURCE
    open_camera = config.open_camera
    sanitize_camera_source = config.sanitize_camera_source
except ImportError:
    default_source = "0"
    def sanitize_camera_source(s):
        if str(s).isdigit(): return int(s)
        s = str(s).strip()
        if ":8080" in s and not any(s.endswith(x) for x in ["/video", "/mjpeg", ".mjpg"]):
            s = s.rstrip("/") + "/video"
        return s
    def open_camera(s, w=640, h=480):
        src = sanitize_camera_source(s)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened() and isinstance(src, str) and (src.startswith("http") or src.startswith("rtsp")):
            try: cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            except Exception: pass
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap, src


def main():
    print("=" * 60)
    print(" CAMERA DIAGNOSTIC & TEST TOOL ")
    print("=" * 60)

    if len(sys.argv) > 1:
        source_input = sys.argv[1]
    else:
        source_input = str(default_source)

    cap, clean_source = open_camera(source_input)
    print(f"Testing Camera Source: {clean_source}")
    print("Attempting to connect...")

    if not cap.isOpened():
        print("\n❌ FAILED: Could not open camera.")
        if isinstance(clean_source, str) and clean_source.startswith("http"):
            print("\nCommon IP Camera Fixes:")
            print("1. URL Endpoint: Make sure you added '/video' at the end.")
            print("   Example: http://192.168.1.15:8080/video (IP Webcam app)")
            print("   Example: http://192.168.1.15:4747/video (DroidCam app)")
            print("2. Wi-Fi Check: Phone and Raspberry Pi MUST be on the exact same Wi-Fi.")
            print("   Try testing connectivity in terminal: curl -I " + clean_source)
            print("3. App Status: Make sure 'Start Server' is tapped in the phone app.")
        return 1

    print(" Connected successfully!")
    print("Reading test frames...")

    frames_read = 0
    start_time = time.time()
    
    # Try reading 30 frames
    while frames_read < 30:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame from stream.")
            break
        
        frames_read += 1
        h, w, c = frame.shape

        # Try to show window if display is available
        try:
            cv2.putText(frame, f"TEST OK: {w}x{h} - Press Q to exit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera Diagnostic Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            # Headless environment without X11 display
            pass

    elapsed = time.time() - start_time
    fps = frames_read / max(elapsed, 0.001)

    print("\n" + "=" * 60)
    print(f" Camera test PASSED!")
    print(f" Captured {frames_read} frames at {fps:.1f} FPS (Resolution: {w}x{h})")
    print("=" * 60)

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
