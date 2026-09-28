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
        
        self.pyttsx3_available = False
        self.pyttsx3_engine = None

        if self.enabled:
            try:
                import pyttsx3
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty("rate", config.TTS_SPEECH_RATE)
                self.pyttsx3_available = True
            except Exception as e:
                # pyttsx3 might fail on some Linux/Pi environments if ALSA/espeak is not linked
                self.pyttsx3_available = False

            # Check if system espeak CLI exists as fallback on Linux / Raspberry Pi
            self.has_espeak_cli = shutil.which("espeak") is not None

            # Start background worker thread
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def speak(self, text: str, force: bool = False):
        """Queue text to be spoken asynchronously if cooldown has passed."""
        if not self.enabled or not text:
            return

        now = time.time()
        # Prevent repeating the same sign continuously
        if not force and text == self.last_spoken_text and (now - self.last_spoken_time) < config.SPEECH_COOLDOWN_SECONDS:
            return

        self.last_spoken_text = text
        self.last_spoken_time = now
        self.queue.put(text)

    def _worker(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if self.pyttsx3_available and self.pyttsx3_engine is not None:
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                elif self.has_espeak_cli:
                    subprocess.run(["espeak", "-s", str(config.TTS_SPEECH_RATE), text], check=False)
                else:
                    print(f"[TTS Output]: {text}")
            except Exception as err:
                print(f"[TTS Error]: {err}")
            finally:
                self.queue.task_done()

    def close(self):
        self.running = False
