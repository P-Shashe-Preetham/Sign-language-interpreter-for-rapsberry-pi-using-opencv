import threading
import queue
import time
import subprocess
import shutil
from typing import Optional

try:
    from . import config
except ImportError:
    import config


class TTSEngine:
    """
    Non-blocking, thread-safe Text-to-Speech engine.
    Ensures voice output never blocks or stutters the real-time video feed.
    """
    def __init__(self, enabled: bool = config.ENABLE_TTS):
        self.enabled = enabled
        self.queue = queue.Queue()
        self.last_spoken_text: str = ""
        self.last_spoken_time: float = 0.0
        self.running = True
        self.has_espeak_cli = shutil.which("espeak") is not None

        if self.enabled:
            # Start background worker thread
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def speak(self, text: str, force: bool = False):
        """Queue text to be spoken asynchronously if cooldown has passed."""
        if not self.enabled or not text or text in ["WAITING...", "NO HAND"]:
            return

        now = time.time()
        # Prevent repeating the same sign continuously
        if not force and text == self.last_spoken_text and (now - self.last_spoken_time) < config.SPEECH_COOLDOWN_SECONDS:
            return

        self.last_spoken_text = text
        self.last_spoken_time = now
        self.queue.put(text)

    def _worker(self):
        # Initialize pyttsx3 inside the worker thread to satisfy Windows COM STA threading rules
        pyttsx3_engine = None
        try:
            import pyttsx3
            pyttsx3_engine = pyttsx3.init()
            pyttsx3_engine.setProperty("rate", config.TTS_SPEECH_RATE)
        except Exception:
            pyttsx3_engine = None

        while self.running:
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            print(f"[VOICE ANNOUNCEMENT]: {text}")
            spoken = False

            # 1. Try pyttsx3 engine
            if pyttsx3_engine is not None:
                try:
                    pyttsx3_engine.say(text)
                    pyttsx3_engine.runAndWait()
                    spoken = True
                except Exception:
                    pass

            # 2. Windows native PowerShell SpeechSynthesizer fallback
            if not spoken and sys.platform.startswith("win"):
                try:
                    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
                    subprocess.run(cmd, shell=True, check=False)
                    spoken = True
                except Exception:
                    pass

            # 3. Linux / Raspberry Pi espeak fallback
            if not spoken and self.has_espeak_cli:
                try:
                    subprocess.run(["espeak", "-s", str(config.TTS_SPEECH_RATE), text], check=False)
                    spoken = True
                except Exception:
                    pass

            self.queue.task_done()

    def close(self):
        self.running = False
