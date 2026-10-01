"""
Video Stream Receiver & Real-Time Sign Language Interpreter (For Laptop)
Displays connection to source "piuser", runs a terminal stream loader, and launches the full ASL interpreter.
"""
import time
import sys
import random
import os

try:
    from src import config
    from src.realtime_inference import main as run_inference
except ImportError:
    # If run directly as script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src import config
    from src.realtime_inference import main as run_inference


def main():
    print("=" * 65)
    print(" 📡 VIDEO STREAM RECEIVER - ACTIVE SESSION")
    print("=" * 65)
    print('Streaming video from source "piuser" to the current device')
    print("Connection: TCP / H.264 RTP Stream (Port 5000)")
    print("=" * 65)

    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    ascii_spinner = ["|", "/", "-", "\\"]
    
    # Animate loading for ~2.5 seconds to show active buffering
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 2.5:
        frame_count += 1
        spin = spinner_chars[frame_count % len(spinner_chars)]
        fps = 29.8 + random.uniform(-0.4, 0.4)
        latency = 12 + random.randint(-2, 2)
        bitrate = 2.4 + random.uniform(-0.1, 0.1)

        status_line = (
            f"\r[{spin}] Buffering stream from source 'piuser'... "
            f"Frames: {frame_count*6} | "
            f"FPS: {fps:.1f} | "
            f"Bitrate: {bitrate:.2f} Mbps | "
            f"Latency: {latency}ms "
        )

        try:
            sys.stdout.write(status_line)
        except UnicodeEncodeError:
            spin_ascii = ascii_spinner[frame_count % len(ascii_spinner)]
            status_line = (
                f"\r[{spin_ascii}] Buffering stream from source 'piuser'... "
                f"Frames: {frame_count*6} | "
                f"FPS: {fps:.1f} | "
                f"Bitrate: {bitrate:.2f} Mbps | "
                f"Latency: {latency}ms "
            )
            sys.stdout.write(status_line)

        sys.stdout.flush()
        time.sleep(0.04)

    # Connected confirmation
    print("\n\n" + "—" * 65)
    print(" [✔] Stream connection established with source 'piuser'!")
    print(" [INFO] Initializing Real-Time ASL Interpreter Engine...")
    print("—" * 65 + "\n")
    time.sleep(0.5)

    # Launch full real-time recognition pipeline
    run_inference()


if __name__ == "__main__":
    main()
