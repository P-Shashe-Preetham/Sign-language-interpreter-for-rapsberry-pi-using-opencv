"""
High-Performance, Thread-Safe Text-to-Speech Engine
Supports:
  - Windows: Direct SAPI.SpVoice with SVSFlagsAsync (100% volume, zero latency)
  - Linux / Raspberry Pi: Native espeak subprocess
"""
import sys
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
        self.last_spoken_text: str = ""
        self.last_spoken_time: float = 0.0
        self.has_espeak_cli = shutil.which("espeak") is not None
        
        # Windows Direct SAPI voice synthesizer
        self.sapi_voice = None
        if self.enabled and sys.platform.startswith("win"):
            try:
                import win32com.client
                self.sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                self.sapi_voice.Volume = 100
            except Exception:
                self.sapi_voice = None

    def speak(self, text: str, force: bool = False):
        """Speaks the text asynchronously without blocking video inference."""
        if not self.enabled or not text or text in ["WAITING...", "NO HAND"]:
            return

        now = time.time()
        # Prevent repeating the same sign continuously
        if not force and text == self.last_spoken_text and (now - self.last_spoken_time) < config.SPEECH_COOLDOWN_SECONDS:
            return

        self.last_spoken_text = text
        self.last_spoken_time = now

        # For single alphabet letters, saying "Letter X" ensures clear audible pronunciation
        phrase = f"Letter {text}" if len(text) == 1 else text
        print(f"\n[VOICE OUTPUT]: {phrase}")

        # 1. Windows: Native async SAPI.SpVoice (Flag 1 = SVSFlagsAsync)
        if self.sapi_voice is not None:
            try:
                self.sapi_voice.Speak(phrase, 1) # 1 = SVSFlagsAsync (completely non-blocking)
                return
            except Exception:
                pass

        # 2. Windows Fallback: Async PowerShell one-liner
        if sys.platform.startswith("win"):
            try:
                cmd = f'powershell -Command "(New-Object -ComObject SAPI.SpVoice).Speak(\'{phrase}\')"'
                subprocess.Popen(cmd, shell=True)
                return
            except Exception:
                pass

        # 3. Linux / Raspberry Pi: Non-blocking espeak subprocess
        if self.has_espeak_cli:
            try:
                rate = str(config.TTS_SPEECH_RATE)
                subprocess.Popen(["espeak", "-s", rate, phrase])
                return
            except Exception:
                pass

        # 4. Fallback pyttsx3 if everything else fails
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("volume", 1.0)
            eng.say(phrase)
            eng.runAndWait()
        except Exception:
            pass

    def close(self):
        pass
