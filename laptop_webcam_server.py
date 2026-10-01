"""
Laptop Webcam Network Streamer
Streams your laptop's built-in webcam over your local Wi-Fi network so a Raspberry Pi (or another device) can use it as an IP camera.

Usage:
  python laptop_webcam_server.py
"""
import cv2
import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 5000
latest_frame = None
frame_lock = threading.Lock()


def get_local_ip():
    """Gets the laptop's local Wi-Fi IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting to public DNS address without sending packets to determine local interface IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class CamStreamHandler(BaseHTTPRequestHandler):
    """Serves raw MJPEG video stream to OpenCV on Raspberry Pi or browsers."""
    def log_message(self, format, *args):
        # Suppress noisy HTTP request logs in terminal
        pass

    def do_GET(self):
        global latest_frame
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
                <body style="background:#111; color:#eee; text-align:center; font-family:sans-serif; padding-top:20px;">
                    <h2>Laptop Webcam Stream (Active)</h2>
                    <p>Raspberry Pi Endpoint: <code>/video</code></p>
                    <img src="/video" style="max-width:90%; border-radius:8px; border:2px solid #444;" />
                </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/video" or self.path == "/video.mjpg":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with frame_lock:
                    if latest_frame is None:
                        time.sleep(0.02)
                        continue
                    frame_bytes = latest_frame
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-type", "image/jpeg")
                    self.send_header("Content-length", str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)  # ~30 FPS
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            self.send_error(404)


def camera_capture_loop(cam_index=0):
    global latest_frame
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] Could not open laptop webcam at index {cam_index}.")
        return

    print("[INFO] Camera capture loop active.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        
        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = jpeg.tobytes()


def main():
    local_ip = get_local_ip()
    stream_url = f"http://{local_ip}:{PORT}/video"

    print("=" * 65)
    print(" 💻 LAPTOP WEBCAM STREAMER FOR RASPBERRY PI ")
    print("=" * 65)
    print(f"[INFO] Streaming Laptop Webcam at: {stream_url}")
    print("\n👉 To run on your Raspberry Pi, run:")
    print(f'   python3 -m src.realtime_inference -s "{stream_url}"')
    print("=" * 65)
    print("Press Ctrl+C to stop the server.\n")

    # Start camera capture thread
    cam_thread = threading.Thread(target=camera_capture_loop, daemon=True)
    cam_thread.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", PORT), CamStreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Laptop webcam server stopped.")


if __name__ == "__main__":
    main()
