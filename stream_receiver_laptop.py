"""
Mock / Standalone Video Stream Receiver (For Laptop)
Simulates receiving live video stream from Raspberry Pi user ("piuser").
"""
import time
import sys
import random

def main():
    print("=" * 65)
    print(" 📡 VIDEO STREAM RECEIVER - ACTIVE SESSION")
    print("=" * 65)
    print('Streaming video from source "piuser" to the current device')
    print("Connection: TCP / H.264 RTP Stream (Port 5000)")
    print("=" * 65)
    print("Press Ctrl+C to disconnect.\n")

    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    # Fallback to standard ascii if terminal doesn't support unicode
    ascii_spinner = ["|", "/", "-", "\\"]
    
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame_count += 1
            spin = spinner_chars[frame_count % len(spinner_chars)]
            
            # Subtle realistic fluctuations
            fps = 29.8 + random.uniform(-0.6, 0.6)
            bitrate = 2.4 + random.uniform(-0.15, 0.15)
            latency = 12 + random.randint(-2, 3)

            status_line = (
                f"\r[{spin}] Streaming from piuser... "
                f"Frames: {frame_count:,} | "
                f"FPS: {fps:.1f} | "
                f"Bitrate: {bitrate:.2f} Mbps | "
                f"Latency: {latency}ms "
            )

            try:
                sys.stdout.write(status_line)
            except UnicodeEncodeError:
                # If Windows console cp1252 doesn't encode unicode spinner
                spin_ascii = ascii_spinner[frame_count % len(ascii_spinner)]
                status_line = (
                    f"\r[{spin_ascii}] Streaming from piuser... "
                    f"Frames: {frame_count:,} | "
                    f"FPS: {fps:.1f} | "
                    f"Bitrate: {bitrate:.2f} Mbps | "
                    f"Latency: {latency}ms "
                )
                sys.stdout.write(status_line)

            sys.stdout.flush()
            time.sleep(0.033) # ~30 FPS

    except KeyboardInterrupt:
        print("\n\n[INFO] Stream disconnected by user.")


if __name__ == "__main__":
    main()
