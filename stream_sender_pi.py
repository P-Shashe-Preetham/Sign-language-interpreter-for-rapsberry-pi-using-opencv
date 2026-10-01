"""
Mock / Standalone Video Stream Transmitter (For Raspberry Pi)
Simulates transmitting live camera feed to the laptop device ("LAPTOP-FCCQ1A").
"""
import sys
import time
import random
import argparse

DEFAULT_LAPTOP_IP = "10.250.30.241"
DEVICE_NAME = "LAPTOP-FCCQ1A"

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Video Streamer")
    parser.add_argument(
        "-i", "--ip",
        default=DEFAULT_LAPTOP_IP,
        help=f"Target Laptop IP (default: {DEFAULT_LAPTOP_IP})"
    )
    args = parser.parse_args()

    laptop_ip = args.ip.strip()

    print("=" * 65)
    print(" 🥧 RASPBERRY PI VIDEO STREAM TRANSMITTER")
    print("=" * 65)
    print(f'Streaming video to device {DEVICE_NAME}({laptop_ip})')
    print("Protocol: MJPEG / RTSP Pipeline (Port 5000) [ESTABLISHED]")
    print("=" * 65)
    print("Press Ctrl+C to terminate transmission.\n")

    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    ascii_spinner = ["|", "/", "-", "\\"]

    packet_count = 0

    try:
        while True:
            packet_count += 1
            spin = spinner_chars[packet_count % len(spinner_chars)]

            fps = 29.9 + random.uniform(-0.5, 0.5)
            bitrate = 2.45 + random.uniform(-0.1, 0.1)
            kb_sent = packet_count * 38.5

            status_line = (
                f"\r[{spin}] Transmitting to {DEVICE_NAME}({laptop_ip})... "
                f"Packets: {packet_count:,} | "
                f"Sent: {kb_sent/1024:.1f} MB | "
                f"FPS: {fps:.1f} | "
                f"Bitrate: {bitrate:.2f} Mbps "
            )

            try:
                sys.stdout.write(status_line)
            except UnicodeEncodeError:
                spin_ascii = ascii_spinner[packet_count % len(ascii_spinner)]
                status_line = (
                    f"\r[{spin_ascii}] Transmitting to {DEVICE_NAME}({laptop_ip})... "
                    f"Packets: {packet_count:,} | "
                    f"Sent: {kb_sent/1024:.1f} MB | "
                    f"FPS: {fps:.1f} | "
                    f"Bitrate: {bitrate:.2f} Mbps "
                )
                sys.stdout.write(status_line)

            sys.stdout.flush()
            time.sleep(0.033) # ~30 FPS

    except KeyboardInterrupt:
        print("\n\n[INFO] Transmission stopped by user.")


if __name__ == "__main__":
    main()
