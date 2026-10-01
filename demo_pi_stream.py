"""
Demo Stream Transmitter (Run on Raspberry Pi)
Simulates a live streaming transmitter sending video to the laptop ('LAPTOP-FF') with an animated terminal stream monitor.
"""
import sys
import time
import os

def main():
    # Enable ANSI escape sequences
    os.system("")

    print("=" * 68)
    print(" 🍓 RASPBERRY PI VIDEO TRANSMITTER ")
    print("=" * 68)
    print('[INFO] Initializing hardware video encoder & Wi-Fi uplink...')
    time.sleep(1.0)
    print('[SUCCESS] Streaming video to device LAPTOP-FF')
    print("=" * 68)
    print("Press Ctrl+C to terminate transmission.\n")

    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    blocks = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    
    packet_count = 0
    start_time = time.time()
    idx = 0

    try:
        while True:
            packet_count += 1
            spin = spinner[idx % len(spinner)]
            progress_bar = "".join([blocks[(idx + i * 2) % len(blocks)] for i in range(8)])
            
            elapsed = time.time() - start_time
            fps = packet_count / max(elapsed, 0.001)
            bitrate = 2800 + (idx % 6) * 50 - (idx % 4) * 35

            status_line = (
                f"\r \033[92m{spin}\033[0m [\033[93m{progress_bar}\033[0m] "
                f"\033[1mTRANSMITTING\033[0m | "
                f"Target: \033[96mLAPTOP-FF\033[0m | "
                f"Packets: \033[92m{packet_count}\033[0m | "
                f"Uplink: \033[97m{bitrate/1000:.2f} Mbps\033[0m | "
                f"FPS: \033[92m{min(fps, 30.0):.1f}\033[0m | "
                f"Status: \033[92mACTIVE\033[0m "
            )
            try:
                sys.stdout.write(status_line)
            except UnicodeEncodeError:
                ascii_spin = ["|", "/", "-", "\\"][idx % 4]
                sys.stdout.write(f"\r [{ascii_spin}] TRANSMITTING | Target: LAPTOP-FF | Packets: {packet_count} | FPS: {min(fps, 30.0):.1f} | Status: ACTIVE ")
            sys.stdout.flush()

            idx += 1
            time.sleep(0.04) # ~25-30 FPS visual refresh

    except KeyboardInterrupt:
        print("\n\n[INFO] Transmission stopped by user.")


if __name__ == "__main__":
    main()
