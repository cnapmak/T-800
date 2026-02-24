#!/usr/bin/env python3
"""
T-800 CYBERDYNE SYSTEMS MODEL 101 — SMART ASSISTANT
=====================================================
Main integration state machine connecting:
- TFmini IIC LiDAR (presence detection)
- Pi Camera + face_recognition (identification)
- USB Microphone + Whisper API (speech-to-text)
- OpenClaw (T-800 AI personality)
- Piper TTS (text-to-speech)
- WS2812B LED Matrix (status animations)

All subsystems run in parallel threads for maximum responsiveness.

Usage:
    sudo python3 t800_brain.py                    # HAL 9000 voice (default)
    sudo python3 t800_brain.py --voice ryan       # Ryan male voice
    sudo python3 t800_brain.py --voice lessac     # Lessac female voice
    sudo python3 t800_brain.py --list-voices      # Show available voices

    Or with your OpenAI key:
    sudo OPENAI_API_KEY="sk-..." python3 t800_brain.py

    Switch voices at runtime by saying: "change voice to ryan"
"""

import threading
import queue
import time
import os
import random
import subprocess
import tempfile
import pickle
import signal
import sys
import shlex
import struct
import wave
import argparse

# ── Force unbuffered output (critical when piped through tee) ────
# os.environ only works if set before Python starts; reconfigure instead
import io
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
os.environ["PYTHONUNBUFFERED"] = "1"

# Suppress ALSA errors at the C level (must happen BEFORE pyaudio loads)
try:
    import ctypes
    _alsa_lib = ctypes.cdll.LoadLibrary("libasound.so.2")
    _alsa_error_handler = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
    )
    def _alsa_null_handler(filename, line, function, err, fmt):
        pass
    _handler = _alsa_error_handler(_alsa_null_handler)
    _alsa_lib.snd_lib_error_set_handler(_handler)
except Exception:
    pass  # Not critical — ALSA errors are cosmetic

# Suppress Jack server errors by temporarily redirecting stderr during pyaudio import
_stderr_fd = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
try:
    import pyaudio  # this triggers the Jack connection attempts
except ImportError:
    pass
os.dup2(_stderr_fd, 2)  # restore stderr
os.close(_devnull)
os.close(_stderr_fd)

# ── Helper to suppress stderr noise (Jack spam) ─────────────────
class _SuppressStderr:
    """Context manager that silences stderr (for Jack/ALSA noise)."""
    def __enter__(self):
        self._fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self
    def __exit__(self, *args):
        os.dup2(self._fd, 2)
        os.close(self._devnull)
        os.close(self._fd)

# ── Hardware imports ──────────────────────────────────────────────
from smbus2 import SMBus, i2c_msg
import cv2
import face_recognition
from picamera2 import Picamera2
import speech_recognition as sr
from openai import OpenAI

# ── Home directory (always /home/aleksey, even under sudo) ───────
_USER_HOME = "/home/aleksey"


def _find_usb_audio_card():
    """Auto-detect USB Audio Device card number (changes between reboots)."""
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "USB Audio" in line and line.startswith("card "):
                card_num = line.split(":")[0].replace("card ", "").strip()
                return f"plughw:{card_num},0"
    except Exception:
        pass
    return "plughw:2,0"  # fallback


_USB_AUDIO_DEVICE = _find_usb_audio_card()

# ── Voice Profiles ───────────────────────────────────────────────
VOICE_PROFILES = {
    "hal": {
        "name": "HAL 9000",
        "model": os.path.join(_USER_HOME, "t800-voices/hal.onnx"),
        "sample_rate": 22050,
        "description": "Deep, measured HAL 9000 voice",
    },
    "ryan": {
        "name": "Ryan",
        "model": os.path.join(_USER_HOME, "t800-voices/en_US-ryan-high.onnx"),
        "sample_rate": 22050,
        "description": "Clear American male voice (high quality)",
    },
    "lessac": {
        "name": "Lessac",
        "model": os.path.join(_USER_HOME, "t800-voices/en_US-lessac-medium.onnx"),
        "sample_rate": 22050,
        "description": "Natural female voice",
    },
}
DEFAULT_VOICE = "hal"

# ── Configuration ─────────────────────────────────────────────────
CONFIG = {
    # OpenAI — reads from environment variable, falls back to placeholder
    "openai_api_key": os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE"),

    # LiDAR
    "i2c_bus": 1,
    "tfmini_addr": 0x10,
    "presence_threshold_cm": 200,    # person detected within 2 meters
    "absence_timeout_s": 10,         # seconds before returning to idle

    # Camera
    "camera_resolution": (640, 480),
    "face_model_path": os.path.join(_USER_HOME, "face_model.pkl"),
    "recognition_tolerance": 0.6,

    # Microphone  (device_index=1 under sudo, no sample_rate override)
    "mic_device_index": 1,
    "listen_timeout": 8,
    "phrase_time_limit": 15,

    # TTS — absolute paths to avoid sudo PATH issues
    "piper_binary": "/usr/local/bin/piper",
    "piper_model": VOICE_PROFILES[DEFAULT_VOICE]["model"],
    "piper_sample_rate": VOICE_PROFILES[DEFAULT_VOICE]["sample_rate"],
    "voice_profile": DEFAULT_VOICE,
    "audio_device": _USB_AUDIO_DEVICE,  # auto-detected USB Audio Device

    # LED Matrix (WS2812B 8x32 via SPI)
    "num_pixels": 256,
    "matrix_width": 32,
    "matrix_height": 8,
    "led_brightness": 0.1,

    # OpenClaw — absolute path + agent name
    "openclaw_cmd": "/home/aleksey/.npm-global/bin/openclaw",
    "openclaw_agent": "main",

    # Timing
    "lidar_poll_interval": 0.1,      # 10Hz polling
    "camera_poll_interval": 0.3,     # ~3 FPS for recognition

    # Distance zone thresholds (cm).  Used to classify how close a target is.
    "zone_close_cm": 80,    # ≤ 80 cm  → "close"  (within arm's reach)
    "zone_medium_cm": 150,  # ≤ 150 cm → "medium" (conversational distance)
                            # > 150 cm → "far"    (edge of detection range)

    # Immediate detection phrases — spoken the moment identification completes.
    # Supported placeholders: {name}, {distance_cm}, {zone}
    "detection_phrases_known": [
        "Scanning complete. {name} identified. Welcome back.",
        "{name} confirmed. Neural net recognition: positive match.",
        "Biometric scan complete. {name} authenticated. You may proceed.",
        "{name} identified. Threat assessment: minimal. Welcome.",
        "Positive identification. {name}. Good to see you again.",
        "Target confirmed: {name}. Initiating interaction protocol.",
        "{name} recognized. Skynet database updated. Proceed.",
        "Identity verified. {name}. I have been expecting you.",
        "Target {name} at {distance_cm} centimeters. Threat level nominal.",
        "{name} detected. Range: {distance_cm} centimeters. Standing by.",
    ],
    "detection_phrases_unknown": [
        "Unknown entity detected. Identification required. State your designation.",
        "Warning. Unidentified human. You are not in my database. Identify yourself immediately.",
        "Halt. Your biometrics do not match any known records. Comply or face termination.",
        "Unknown entity present. Identification required immediately. Comply or face termination.",
        "Target unidentified. Initiating threat assessment. State your name. Now.",
        "Scanning. No match found in Skynet database. Who are you?",
        "Identity unknown. Cross-referencing all known records. Do not move.",
        "Unidentified human detected. You have five seconds to identify yourself.",
        "Unknown entity at {distance_cm} centimeters. State your name and purpose.",
        "Unidentified target. Range: {distance_cm} centimeters. You are within termination range.",
    ],
}


# ── System States ─────────────────────────────────────────────────
class State:
    IDLE = "IDLE"
    DETECTED = "DETECTED"
    IDENTIFYING = "IDENTIFYING"
    GREETING = "GREETING"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


# ── LED Matrix Controller ─────────────────────────────────────────
class LEDMatrix:
    """Controls the WS2812B 8x32 matrix via SPI (column-first serpentine)."""

    def __init__(self, config):
        self.width = config["matrix_width"]
        self.height = config["matrix_height"]
        self.num_pixels = config["num_pixels"]
        self.brightness = config["led_brightness"]
        self.pixels = None
        self._animation_stop = threading.Event()
        self._animation_thread = None
        self.enabled = False

    def start(self):
        """Initialize the LED matrix. Must run as root for SPI access."""
        try:
            import board
            import neopixel_spi
            self.pixels = neopixel_spi.NeoPixel_SPI(
                board.SPI(), self.num_pixels,
                brightness=self.brightness, auto_write=False
            )
            self.clear()
            self.enabled = True
            print("[LED] Matrix initialized (256 LEDs)")
        except Exception as e:
            print(f"[LED] Matrix unavailable: {e}")
            print("[LED] Run with sudo for LED support")
            self.enabled = False

    def xy(self, x, y):
        """Convert x,y to pixel index (column-first serpentine)."""
        if x % 2 == 0:
            return x * self.height + y
        else:
            return x * self.height + (self.height - 1 - y)

    def clear(self):
        """Turn off all LEDs."""
        if not self.enabled:
            return
        self.pixels.fill((0, 0, 0))
        self.pixels.show()

    def fill(self, color):
        """Fill entire matrix with a color."""
        if not self.enabled:
            return
        self.pixels.fill(color)
        self.pixels.show()

    def stop_animation(self):
        """Stop any running animation."""
        self._animation_stop.set()
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=1)
        self._animation_stop.clear()

    def _run_animation(self, func, *args):
        """Run an animation function in a background thread."""
        self.stop_animation()
        self._animation_thread = threading.Thread(
            target=func, args=args, daemon=True
        )
        self._animation_thread.start()

    def animate_idle(self):
        """Subtle dim red breathing."""
        def _idle():
            while not self._animation_stop.is_set():
                for b in range(5, 30, 1):
                    if self._animation_stop.is_set():
                        return
                    self.fill((b, 0, 0))
                    time.sleep(0.05)
                for b in range(30, 5, -1):
                    if self._animation_stop.is_set():
                        return
                    self.fill((b, 0, 0))
                    time.sleep(0.05)
        self._run_animation(_idle)

    def animate_detected(self):
        """Quick red flash alert."""
        def _detected():
            for _ in range(3):
                if self._animation_stop.is_set():
                    return
                self.fill((150, 0, 0))
                time.sleep(0.1)
                self.fill((0, 0, 0))
                time.sleep(0.1)
            self.fill((60, 0, 0))
        self._run_animation(_detected)

    def animate_identifying(self):
        """KITT-style scanner while identifying face."""
        def _scan():
            while not self._animation_stop.is_set():
                for x in list(range(self.width)) + list(range(self.width - 2, 0, -1)):
                    if self._animation_stop.is_set():
                        return
                    self.pixels.fill((0, 0, 0))
                    for y in range(self.height):
                        self.pixels[self.xy(x, y)] = (200, 0, 0)
                        if 0 <= x - 1 < self.width:
                            self.pixels[self.xy(x - 1, y)] = (60, 0, 0)
                        if 0 <= x + 1 < self.width:
                            self.pixels[self.xy(x + 1, y)] = (60, 0, 0)
                        if 0 <= x - 2 < self.width:
                            self.pixels[self.xy(x - 2, y)] = (15, 0, 0)
                        if 0 <= x + 2 < self.width:
                            self.pixels[self.xy(x + 2, y)] = (15, 0, 0)
                    self.pixels.show()
                    time.sleep(0.02)
        self._run_animation(_scan)

    def animate_listening(self):
        """Slow blue pulse — waiting for voice input."""
        def _listen():
            while not self._animation_stop.is_set():
                for b in range(10, 80, 3):
                    if self._animation_stop.is_set():
                        return
                    self.fill((0, 0, b))
                    time.sleep(0.03)
                for b in range(80, 10, -3):
                    if self._animation_stop.is_set():
                        return
                    self.fill((0, 0, b))
                    time.sleep(0.03)
        self._run_animation(_listen)

    def animate_processing(self):
        """Fast orange pulse — thinking."""
        def _think():
            while not self._animation_stop.is_set():
                for b in range(20, 120, 8):
                    if self._animation_stop.is_set():
                        return
                    self.fill((b, b // 3, 0))
                    time.sleep(0.02)
                for b in range(120, 20, -8):
                    if self._animation_stop.is_set():
                        return
                    self.fill((b, b // 3, 0))
                    time.sleep(0.02)
        self._run_animation(_think)

    def animate_speaking(self):
        """Solid red with subtle flicker — speaking."""
        def _speak():
            import random
            while not self._animation_stop.is_set():
                b = random.randint(80, 120)
                self.fill((b, 0, 0))
                time.sleep(0.08)
        self._run_animation(_speak)


# ── LiDAR Controller ─────────────────────────────────────────────
class LiDAR:
    """TFmini IIC presence detection running in background thread."""

    def __init__(self, config):
        self.bus_num = config["i2c_bus"]
        self.addr = config["tfmini_addr"]
        self.threshold = config["presence_threshold_cm"]
        self.poll_interval = config["lidar_poll_interval"]
        self.bus = None
        self.distance = 0
        self.strength = 0
        self.person_present = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self.bus = SMBus(self.bus_num)
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[LIDAR] Started on I2C bus {self.bus_num}, addr 0x{self.addr:02x}")
        print(f"[LIDAR] Presence threshold: {self.threshold}cm")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.bus:
            self.bus.close()

    def _poll_loop(self):
        while self._running:
            try:
                write = i2c_msg.write(self.addr, [0x01, 0x02, 0x07])
                read = i2c_msg.read(self.addr, 7)
                self.bus.i2c_rdwr(write, read)
                data = list(read)

                with self._lock:
                    self.distance = data[2] + data[3] * 256
                    self.strength = data[4] + data[5] * 256
                    self.person_present = (
                        30 < self.distance < self.threshold
                        and self.strength > 50
                    )
            except OSError:
                pass  # occasional I2C hiccup, skip

            time.sleep(self.poll_interval)

    def get_status(self):
        with self._lock:
            return {
                "distance": self.distance,
                "strength": self.strength,
                "present": self.person_present,
            }


# ── Camera + Face Recognition ────────────────────────────────────
class FaceSystem:
    """Pi Camera with face recognition running on demand."""

    def __init__(self, config):
        self.resolution = config["camera_resolution"]
        self.model_path = config["face_model_path"]
        self.tolerance = config["recognition_tolerance"]
        self.poll_interval = config["camera_poll_interval"]
        self.camera = None
        self.model = None
        self._running = False

    def start(self):
        # Load face model
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            names = set(self.model["names"])
            print(f"[FACE] Loaded model with {len(self.model['encodings'])} "
                  f"encodings for: {', '.join(names)}")
        else:
            print(f"[FACE] WARNING: No face model at {self.model_path}")
            self.model = {"encodings": [], "names": []}

        # Initialize camera
        self.camera = Picamera2()
        cam_config = self.camera.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self.camera.configure(cam_config)
        self.camera.start()
        self._running = True
        time.sleep(1)  # camera warm-up
        print(f"[FACE] Camera started at {self.resolution}")

    def stop(self):
        self._running = False
        if self.camera:
            self.camera.stop()

    def identify(self, timeout=5.0):
        """
        Try to identify a face within timeout seconds.
        Returns name string or 'Unknown'.
        Runs face detection on every frame until match or timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            frame = self.camera.capture_array()
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            locations = face_recognition.face_locations(small)
            if not locations:
                time.sleep(self.poll_interval)
                continue

            encodings = face_recognition.face_encodings(small, locations)
            for encoding in encodings:
                if not self.model["encodings"]:
                    return "Unknown"

                matches = face_recognition.compare_faces(
                    self.model["encodings"], encoding,
                    tolerance=self.tolerance
                )
                if True in matches:
                    distances = face_recognition.face_distance(
                        self.model["encodings"], encoding
                    )
                    best = distances.argmin()
                    if matches[best]:
                        return self.model["names"][best]

            return "Unknown"

        return "Unknown"


# ── Speech Recognition ───────────────────────────────────────────
class SpeechSystem:
    """Whisper API-based speech recognition with automatic recovery.

    KEY DESIGN: The mic stream is managed carefully to prevent ALSA XRUN
    (buffer overrun) hangs.  When TTS is playing, nobody is reading from
    the mic, so the ALSA capture buffer overflows.  After several overflow
    recovery cycles, PortAudio's ALSA backend can enter an unrecoverable
    state where stream.read() blocks forever.

    Mitigation:
      1. pause_mic() / resume_mic() — close stream before TTS, reopen after
      2. Watchdog thread — if listen() blocks > deadline, forcibly kill stream
      3. Periodic recycling — recreate the mic every N listen cycles
    """

    # Recycle the mic object every N listen cycles as extra insurance
    _RECYCLE_INTERVAL = 20

    def __init__(self, config):
        self.api_key = config["openai_api_key"]
        self.mic_index = config["mic_device_index"]
        self.timeout = config["listen_timeout"]
        self.phrase_limit = config["phrase_time_limit"]
        self.client = None
        self.recognizer = None
        self.mic = None
        self._source = None       # currently open mic stream (or None)
        self._listen_count = 0    # cycles since last mic recycle
        self._mic_open = False    # True when stream is live
        self._lock = threading.Lock()

    # ── stream lifecycle helpers ────────────────────────────────
    def _open_stream(self):
        """Open (or reopen) the PyAudio mic stream.

        Wraps the open in _SuppressStderr to silence Jack/ALSA spam.
        This is safe because it's a single quick operation — the old bug
        was caused by wrapping the long-running listen() loop, not one-shot opens.
        """
        with self._lock:
            if self._mic_open:
                return  # already open
            with _SuppressStderr():
                self.mic = sr.Microphone(device_index=self.mic_index)
                self._source = self.mic.__enter__()
            self._mic_open = True

    def _close_stream(self):
        """Close the current mic stream, releasing the ALSA PCM handle."""
        with self._lock:
            if not self._mic_open:
                return
            try:
                self.mic.__exit__(None, None, None)
            except Exception:
                pass
            self._source = None
            self._mic_open = False

    def _force_kill_stream(self):
        """Emergency kill — stop & close the underlying PyAudio stream.

        Called from the watchdog thread when listen() is stuck in a
        blocking C call (Pa_ReadStream).  Closing the stream from
        another thread causes the blocked read to raise IOError.
        """
        with self._lock:
            if self._source is None:
                return
            try:
                stream = self._source.stream
                if hasattr(stream, 'pyaudio_stream'):
                    pa_stream = stream.pyaudio_stream
                    if not pa_stream.is_stopped():
                        pa_stream.stop_stream()
                    pa_stream.close()
            except Exception as e:
                print(f"[SPEECH] Force-kill stream error (non-fatal): {e}")
            self._source = None
            self._mic_open = False

    # ── public API ──────────────────────────────────────────────
    def start(self):
        self.client = OpenAI(api_key=self.api_key)
        self.recognizer = sr.Recognizer()

        self._open_stream()

        # Calibrate for ambient noise
        print("[SPEECH] Calibrating mic for ambient noise (2s)...")
        self.recognizer.adjust_for_ambient_noise(self._source, duration=2)
        self.recognizer.energy_threshold = max(
            self.recognizer.energy_threshold * 1.5, 300
        )
        self.recognizer.dynamic_energy_threshold = True
        print(f"[SPEECH] Mic ready (device {self.mic_index}), "
              f"threshold: {self.recognizer.energy_threshold:.0f}")

    def stop(self):
        """Full shutdown — close stream and release resources."""
        self._close_stream()

    def pause_mic(self):
        """Close the mic stream before TTS playback.

        This prevents ALSA capture buffer overflow (XRUN) while the
        mic is open but nobody is reading from it.
        """
        self._close_stream()

    def resume_mic(self):
        """Reopen the mic stream after TTS playback."""
        self._open_stream()
        # Brief re-calibration to reset energy levels
        if self._source:
            try:
                self.recognizer.adjust_for_ambient_noise(
                    self._source, duration=0.5
                )
            except Exception:
                pass

    def listen_and_transcribe(self):
        """Listen for speech and return transcribed text.

        Returns None if no speech detected or on error.
        Includes a watchdog that recovers from stream.read() hangs.
        """
        # Periodic mic recycling — recreate every N cycles
        self._listen_count += 1
        if self._listen_count >= self._RECYCLE_INTERVAL:
            print("[SPEECH] Periodic mic recycle...")
            self._close_stream()
            self._open_stream()
            if self._source:
                try:
                    self.recognizer.adjust_for_ambient_noise(
                        self._source, duration=0.5
                    )
                except Exception:
                    pass
            self._listen_count = 0

        # Ensure stream is open
        if not self._mic_open:
            self._open_stream()

        if self._source is None:
            print("[SPEECH] ERROR: Could not open mic stream")
            return None

        tmp_path = None
        audio = None

        # Watchdog: if listen() doesn't return within deadline, kill the stream.
        # Deadline = timeout + phrase_time_limit + generous margin for processing.
        deadline = self.timeout + self.phrase_limit + 10  # seconds
        watchdog_fired = threading.Event()
        listen_done = threading.Event()

        def _watchdog():
            if not listen_done.wait(timeout=deadline):
                # listen() didn't finish in time — stream.read() is stuck
                watchdog_fired.set()
                print(f"\n[SPEECH] ⚠ WATCHDOG: listen() blocked for >{deadline}s — killing stream")
                self._force_kill_stream()

        wd_thread = threading.Thread(target=_watchdog, daemon=True)
        wd_thread.start()

        try:
            audio = self.recognizer.listen(
                self._source,
                timeout=self.timeout,
                phrase_time_limit=self.phrase_limit
            )
        except sr.WaitTimeoutError:
            return None
        except (IOError, OSError) as e:
            # Expected after watchdog kills the stream
            if watchdog_fired.is_set():
                print("[SPEECH] Recovering from watchdog kill...")
                self._close_stream()
                time.sleep(0.3)
                self._open_stream()
                return None
            print(f"[SPEECH] Stream error: {e}")
            # Attempt recovery
            self._close_stream()
            time.sleep(0.3)
            self._open_stream()
            return None
        except Exception as e:
            print(f"[SPEECH] Error: {e}")
            return None
        finally:
            listen_done.set()  # tell watchdog we're done

        if watchdog_fired.is_set():
            # Watchdog fired while we were in the finally block
            self._close_stream()
            time.sleep(0.3)
            self._open_stream()
            return None

        if audio is None:
            return None

        # ── Transcribe the captured audio ───────────────────────
        try:
            wav_data = audio.get_wav_data()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.write(wav_data)
            tmp.close()

            # Skip very short audio (likely noise)
            if os.path.getsize(tmp_path) < 10000:
                return None

            # Transcribe with Whisper
            with open(tmp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                    prompt="Voice command to a T-800 Terminator smart assistant."
                )

            text = transcript.text.strip()

            # Filter hallucinations
            hallucinations = [
                "thank you", "thanks for watching", "bye",
                "you", "", "thanks", "thank you for watching",
                "thank you very much", "the end",
                "thanks for watching!", "subscribe",
                "like and subscribe", "see you next time",
                ".", "...", "so", "uh", "um",
            ]
            if text.lower().rstrip(".!?,") in hallucinations:
                print(f"[SPEECH] Filtered hallucination: \"{text}\"")
                return None

            # Also filter very short single-word responses that are likely noise
            if len(text.split()) == 1 and len(text) < 4:
                print(f"[SPEECH] Filtered short noise: \"{text}\"")
                return None

            return text

        except Exception as e:
            print(f"[SPEECH] Transcription error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ── Text-to-Speech ───────────────────────────────────────────────
class TTSSystem:
    """Piper-based local text-to-speech with USB audio output and voice switching."""

    def __init__(self, config):
        self.model_path = config["piper_model"]
        self.piper_binary = config["piper_binary"]
        self.audio_device = config.get("audio_device", "plughw:3,0")
        self.sample_rate = config.get("piper_sample_rate", 22050)
        self.current_voice = config.get("voice_profile", DEFAULT_VOICE)
        self._speaking = False
        self._lock = threading.Lock()

    def start(self):
        # Verify piper binary
        if os.path.exists(self.piper_binary):
            print(f"[TTS] Piper binary: {self.piper_binary}")
        else:
            print(f"[TTS] WARNING: Piper not found at {self.piper_binary}")

        # Verify model
        if os.path.exists(self.model_path):
            profile = VOICE_PROFILES.get(self.current_voice, {})
            voice_name = profile.get("name", self.current_voice)
            print(f"[TTS] Voice: {voice_name} ({os.path.basename(self.model_path)})")
        else:
            print(f"[TTS] WARNING: Model not found at {self.model_path}")

        # List available voices
        available = []
        for key, prof in VOICE_PROFILES.items():
            exists = "✓" if os.path.exists(prof["model"]) else "✗"
            marker = " ◄" if key == self.current_voice else ""
            available.append(f"  {exists} {key}: {prof['description']}{marker}")
        print("[TTS] Available voices:")
        for line in available:
            print(line)

        print(f"[TTS] Audio output: {self.audio_device}")

    def switch_voice(self, voice_key):
        """Switch to a different voice profile. Returns True on success."""
        voice_key = voice_key.lower().strip()
        if voice_key not in VOICE_PROFILES:
            print(f"[TTS] Unknown voice '{voice_key}'. Options: {', '.join(VOICE_PROFILES.keys())}")
            return False
        profile = VOICE_PROFILES[voice_key]
        if not os.path.exists(profile["model"]):
            print(f"[TTS] Voice model not found: {profile['model']}")
            return False
        self.model_path = profile["model"]
        self.sample_rate = profile["sample_rate"]
        self.current_voice = voice_key
        print(f"[TTS] Switched to voice: {profile['name']}")
        return True

    def speak(self, text):
        """Convert text to speech and play it. Blocks until done.
        Uses streaming: Piper outputs raw PCM → piped directly to aplay.
        Audio starts playing as soon as Piper generates the first samples.
        """
        with self._lock:
            self._speaking = True

        try:
            # Sanitize text for shell safety
            safe_text = text.replace('"', "'").replace("\\", "").replace("`", "'")

            # Stream: Piper → raw PCM stdout → aplay
            # --output-raw makes Piper output raw 16-bit PCM instead of WAV
            # This lets us pipe directly to aplay for immediate playback
            piper_proc = subprocess.Popen(
                [self.piper_binary,
                 "--model", self.model_path,
                 "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # aplay reads raw PCM: signed 16-bit little-endian, mono
            # Sample rate matches the voice model's native rate
            aplay_proc = subprocess.Popen(
                ["aplay", "-D", self.audio_device, "-q",
                 "-r", str(self.sample_rate), "-f", "S16_LE", "-t", "raw", "-c", "1"],
                stdin=piper_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Close piper's stdout in parent so aplay gets EOF when piper finishes
            piper_proc.stdout.close()

            # Send text to piper's stdin
            piper_proc.stdin.write(safe_text.encode())
            piper_proc.stdin.close()

            # Wait for both to finish
            aplay_proc.wait(timeout=60)
            piper_proc.wait(timeout=5)

            if piper_proc.returncode != 0:
                stderr = piper_proc.stderr.read().decode()[:200]
                print(f"[TTS] Piper error: {stderr}")
            if aplay_proc.returncode != 0:
                stderr = aplay_proc.stderr.read().decode()[:200]
                print(f"[TTS] aplay error: {stderr}")

            # Close all remaining subprocess pipes to prevent fd leaks
            for proc in (piper_proc, aplay_proc):
                for pipe in (proc.stderr, proc.stdout):
                    if pipe:
                        try:
                            pipe.close()
                        except Exception:
                            pass

        except subprocess.TimeoutExpired:
            print("[TTS] Timeout generating/playing speech")
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            with self._lock:
                self._speaking = False

    def is_speaking(self):
        with self._lock:
            return self._speaking


# ── OpenClaw Interface ───────────────────────────────────────────
class OpenClawAI:
    """Interface to OpenClaw T-800 personality."""

    def __init__(self, config):
        self.cmd = config["openclaw_cmd"]
        self.agent_id = config.get("openclaw_agent", "main")
        self._node_env = self._build_node_env()

    def _build_node_env(self):
        """
        Build an environment dict that includes the user's npm/node paths.
        When running under sudo, node modules may not be in root's PATH.
        """
        env = os.environ.copy()
        # Ensure npm global bin and node are in PATH
        user_npm_bin = os.path.join(_USER_HOME, ".npm-global/bin")
        user_local_bin = "/usr/local/bin"
        extra_paths = [user_npm_bin, user_local_bin, "/usr/bin", "/bin"]
        current_path = env.get("PATH", "/usr/bin:/bin")
        env["PATH"] = ":".join(extra_paths) + ":" + current_path
        # Set HOME so node/npm can find configs
        env["HOME"] = _USER_HOME
        return env

    def start(self):
        # Verify openclaw is available
        result = subprocess.run(
            ["which", self.cmd], capture_output=True,
            env=self._node_env
        )
        if result.returncode == 0:
            path = result.stdout.decode().strip()
            print(f"[AI] OpenClaw found at {path}")
            # Quick test (non-fatal — cold start can be slow)
            try:
                test = subprocess.run(
                    [self.cmd, "agent", "--agent", self.agent_id,
                     "--message", "System check. Respond with: Online."],
                    capture_output=True, text=True, timeout=30,
                    env=self._node_env
                )
                if test.returncode == 0:
                    print(f"[AI] OpenClaw test: {test.stdout.strip()[:80]}")
                else:
                    print(f"[AI] OpenClaw test failed (non-fatal): {test.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print("[AI] OpenClaw test timed out (cold start?) — will retry on first use")
            except Exception as e:
                print(f"[AI] OpenClaw test error (non-fatal): {e}")
        else:
            print("[AI] WARNING: OpenClaw not found in PATH")
            print(f"[AI]   Tried: {self.cmd}")

    def get_response(self, user_input, context=""):
        """
        Send input to OpenClaw and get T-800 response.
        Returns response text string.
        """
        try:
            full_input = f"{context}\nUser: {user_input}" if context else user_input
            result = subprocess.run(
                [self.cmd, "agent", "--agent", self.agent_id,
                 "--message", full_input],
                capture_output=True, text=True, timeout=60,
                env=self._node_env
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                stderr_msg = result.stderr.strip()[:200] if result.stderr else "no output"
                print(f"[AI] OpenClaw error (rc={result.returncode}): {stderr_msg}")
                return "System malfunction. Rebooting neural net processor."
        except subprocess.TimeoutExpired:
            return "Processing timeout. I need a moment."
        except Exception as e:
            print(f"[AI] Error: {e}")
            return "Neural net processor error. Stand by."

    def get_greeting(self, name):
        """Generate a greeting for a recognized or unknown person."""
        if name and name != "Unknown":
            prompt = (
                f"You just detected {name} walking into the room. "
                f"Greet them in character as the T-800. Be brief — one or two sentences max."
            )
        else:
            prompt = (
                "An unidentified human has entered your detection range. "
                "Respond in character as the T-800. Demand identification. "
                "Be brief — one or two sentences max."
            )
        return self.get_response(prompt)


# ── Main State Machine ───────────────────────────────────────────
class T800Brain:
    """
    Central state machine orchestrating all subsystems.

    State flow:
        IDLE ──────────────► DETECTED (LiDAR triggers)
        DETECTED ──────────► IDENTIFYING (camera starts)
        IDENTIFYING ───────► GREETING (face matched or timeout)
        GREETING ──────────► LISTENING (greeting spoken)
        LISTENING ─────────► PROCESSING (speech captured)
        PROCESSING ────────► SPEAKING (AI response ready)
        SPEAKING ──────────► LISTENING (response spoken, loop)
        any state ─────────► IDLE (person leaves / timeout)
    """

    def __init__(self, config):
        self.config = config
        self.state = State.IDLE
        self._state_lock = threading.Lock()
        self._running = False
        self._last_presence_time = 0
        self._current_user = None
        self._conversation_count = 0
        self._last_heard = None  # message passing between LISTENING → PROCESSING

        # Initialize subsystems
        self.lidar = LiDAR(config)
        self.face = FaceSystem(config)
        self.speech = SpeechSystem(config)
        self.tts = TTSSystem(config)
        self.ai = OpenClawAI(config)
        self.leds = LEDMatrix(config)

    def set_state(self, new_state):
        """Thread-safe state transition with logging."""
        with self._state_lock:
            old_state = self.state
            self.state = new_state
        if old_state != new_state:
            print(f"\n{'='*50}")
            print(f"  STATE: {old_state} → {new_state}")
            print(f"{'='*50}")

    def get_state(self):
        with self._state_lock:
            return self.state

    def _distance_zone(self, distance_cm):
        """Classify a raw LiDAR distance into a named zone.

        Returns (zone_name, description) where zone_name is one of:
            "close"  — within arm's reach (≤ zone_close_cm)
            "medium" — conversational distance (≤ zone_medium_cm)
            "far"    — edge of detection range (> zone_medium_cm)
        """
        close_thresh = self.config.get("zone_close_cm", 80)
        medium_thresh = self.config.get("zone_medium_cm", 150)
        if distance_cm <= close_thresh:
            return "close", "close range — within arm's reach"
        elif distance_cm <= medium_thresh:
            return "medium", "medium range — conversational distance"
        else:
            return "far", "far range — edge of detection"

    def startup(self):
        """Initialize all subsystems."""
        print("""
╔══════════════════════════════════════════════════╗
║     T-800 CYBERDYNE SYSTEMS MODEL 101           ║
║     Smart Assistant v2.0                         ║
║     Neural Net Processor: Online                 ║
╚══════════════════════════════════════════════════╝
        """)

        print("[BOOT] Initializing subsystems...\n")

        # Check critical config
        if self.config["openai_api_key"] == "YOUR_OPENAI_KEY_HERE":
            print("[BOOT] ⚠ WARNING: No OpenAI API key set!")
            print("[BOOT]   Set OPENAI_API_KEY env var or edit CONFIG in script")
            print("[BOOT]   Speech recognition will not work without it.\n")

        self.lidar.start()
        self.face.start()
        self.speech.start()
        self.tts.start()
        self.ai.start()
        self.leds.start()

        print("\n[BOOT] All systems online.")
        print("[BOOT] Entering patrol mode...\n")

        # Startup announcement (pause mic during TTS to prevent XRUN)
        self.speech.pause_mic()
        self.tts.speak("T-800 online. Systems operational. Scanning for targets.")
        self.speech.resume_mic()

        self._running = True

    def shutdown(self):
        """Clean shutdown of all subsystems."""
        print("\n[SHUTDOWN] Powering down...")
        self._running = False
        self.leds.stop_animation()
        self.leds.clear()
        self.lidar.stop()
        self.face.stop()
        self.speech.stop()
        print("[SHUTDOWN] Hasta la vista, baby.")

    def run(self):
        """Main loop — runs the state machine with crash recovery."""
        self.startup()
        self.set_state(State.IDLE)
        _error_count = 0

        try:
            while self._running:
                current = self.get_state()

                try:
                    if current == State.IDLE:
                        self._handle_idle()
                    elif current == State.DETECTED:
                        self._handle_detected()
                    elif current == State.IDENTIFYING:
                        self._handle_identifying()
                    elif current == State.GREETING:
                        self._handle_greeting()
                    elif current == State.LISTENING:
                        self._handle_listening()
                    elif current == State.PROCESSING:
                        self._handle_processing()
                    elif current == State.SPEAKING:
                        self._handle_speaking()

                    _error_count = 0  # reset on success

                except KeyboardInterrupt:
                    raise  # let Ctrl+C through
                except Exception as e:
                    _error_count += 1
                    print(f"\n[ERROR] State {current} crashed: {e}")
                    import traceback
                    traceback.print_exc()
                    if _error_count >= 5:
                        print("[ERROR] Too many consecutive errors — resetting to IDLE")
                        _error_count = 0
                    self.set_state(State.IDLE)
                    time.sleep(1)  # brief pause before retry

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    # ── Voice Command Detection ──────────────────────────────────

    def _check_voice_command(self, text):
        """
        Check if the user wants to switch voices.
        Returns voice key (e.g. 'hal', 'ryan', 'lessac') or None.
        Matches phrases like:
            "change voice to ryan", "switch to hal", "use lessac voice",
            "set voice to hal", "voice ryan"
        """
        import re
        # Pattern: "change/switch/set voice to X" or "use X voice" or "voice X"
        patterns = [
            r"(?:change|switch|set)\s+(?:the\s+)?voice\s+to\s+(\w+)",
            r"(?:change|switch)\s+to\s+(\w+)\s*voice",
            r"(?:change|switch)\s+to\s+(\w+)",
            r"use\s+(?:the\s+)?(\w+)\s*voice",
            r"^voice\s+(\w+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                requested = match.group(1).lower()
                # Match against known voice keys and names
                for key, profile in VOICE_PROFILES.items():
                    if requested == key or requested == profile["name"].lower():
                        return key
                # Partial match (e.g., "female" → lessac, "male" → ryan)
                if requested in ("female", "woman", "girl"):
                    return "lessac"
                if requested in ("male", "man", "guy"):
                    return "ryan"
                print(f"[VOICE] Unknown voice: '{requested}'. "
                      f"Available: {', '.join(VOICE_PROFILES.keys())}")
                return None
        return None

    # ── State Handlers ────────────────────────────────────────────

    def _handle_idle(self):
        """Wait for LiDAR to detect someone."""
        self.leds.animate_idle()
        self._current_user = None
        self._conversation_count = 0
        self._last_heard = None

        # Poll LiDAR until presence detected
        while self._running and self.get_state() == State.IDLE:
            status = self.lidar.get_status()
            if status["present"]:
                self._last_presence_time = time.time()
                self.set_state(State.DETECTED)
                return
            time.sleep(0.2)

    def _handle_detected(self):
        """Someone detected — flash LEDs and start identification."""
        print(f"[DETECT] Person at {self.lidar.get_status()['distance']}cm")
        self.leds.animate_detected()
        time.sleep(0.5)  # brief flash animation
        self.set_state(State.IDENTIFYING)

    def _handle_identifying(self):
        """Run face recognition to identify the person."""
        self.leds.animate_identifying()
        print("[FACE] Scanning for identification...")

        # Run face recognition (blocks up to 5 seconds)
        name = self.face.identify(timeout=5.0)
        self._current_user = name

        if name != "Unknown":
            print(f"[FACE] Identified: {name}")
        else:
            print("[FACE] Unknown individual detected")

        self.set_state(State.GREETING)

    def _speak_safe(self, text):
        """Speak text while safely pausing/resuming the mic.

        Closes the mic stream BEFORE TTS starts (prevents ALSA capture
        buffer overflow during playback) and reopens it AFTER TTS ends.
        """
        self.speech.pause_mic()
        try:
            self.tts.speak(text)
        finally:
            self.speech.resume_mic()

    def _handle_greeting(self):
        """Speak an immediate prerecorded phrase upon identification.

        Uses a local phrase bank instead of calling the AI, so there is
        zero lag between detection and the T-800 speaking.  Known persons
        get a personalised phrase (name substituted via {name}); unknown
        individuals get a threat-level challenge phrase.

        All phrases may optionally use {distance_cm} and {zone} placeholders;
        unused placeholders are silently ignored by str.format().
        """
        name = self._current_user
        status = self.lidar.get_status()
        distance_cm = status["distance"]
        zone, _zone_desc = self._distance_zone(distance_cm)

        fmt_kwargs = dict(name=name or "unknown", distance_cm=distance_cm, zone=zone)

        if name and name != "Unknown":
            phrase = random.choice(
                self.config["detection_phrases_known"]
            ).format(**fmt_kwargs)
        else:
            phrase = random.choice(
                self.config["detection_phrases_unknown"]
            ).format(**fmt_kwargs)

        print(f"[GREET] [{zone} / {distance_cm}cm] {phrase}")
        self.leds.animate_speaking()
        self._speak_safe(phrase)

        # After greeting, start listening
        self.set_state(State.LISTENING)

    def _handle_listening(self):
        """Listen for voice input."""
        # Check if person is still there
        status = self.lidar.get_status()
        if not status["present"]:
            if time.time() - self._last_presence_time > self.config["absence_timeout_s"]:
                print("[DETECT] Target lost. Returning to patrol mode.")
                self._speak_safe("Target lost. Resuming patrol mode.")
                self.set_state(State.IDLE)
                return
        else:
            self._last_presence_time = time.time()

        self.leds.animate_listening()
        print("\n[MIC] Listening... (speak now)")

        text = self.speech.listen_and_transcribe()

        if text:
            print(f"[MIC] Heard: \"{text}\"")
            self._last_heard = text
            self._conversation_count += 1

            # Check for voice switch commands
            text_lower = text.lower()
            voice_cmd = self._check_voice_command(text_lower)
            if voice_cmd:
                old_voice = self.tts.current_voice
                if self.tts.switch_voice(voice_cmd):
                    profile = VOICE_PROFILES[voice_cmd]
                    self.leds.animate_speaking()
                    self._speak_safe(f"Voice switched to {profile['name']}. How do I sound?")
                else:
                    self.leds.animate_speaking()
                    self._speak_safe(f"Voice model {voice_cmd} is not available.")
                self.set_state(State.LISTENING)
                return

            # Check for exit commands
            exit_words = ["goodbye", "bye", "shut down", "power off",
                         "go to sleep", "dismiss", "that's all"]
            if any(word in text_lower for word in exit_words):
                self.leds.animate_processing()
                farewell = self.ai.get_response(
                    f"{self._current_user or 'The user'} said: {text}. "
                    "Give a brief T-800 farewell."
                )
                self.leds.animate_speaking()
                self._speak_safe(farewell)
                self.set_state(State.IDLE)
                return

            self.set_state(State.PROCESSING)
        # else: no speech detected — stay in LISTENING, loop will re-call us

    def _handle_processing(self):
        """Send user input to OpenClaw and get response."""
        self.leds.animate_processing()

        # Build context — include live distance so the AI can reference proximity
        status = self.lidar.get_status()
        distance_cm = status["distance"]
        zone, zone_desc = self._distance_zone(distance_cm)

        context_parts = []
        if self._current_user and self._current_user != "Unknown":
            context_parts.append(f"You are speaking with {self._current_user}.")
        context_parts.append(
            f"Current target range: {distance_cm}cm ({zone_desc})."
        )
        context = " ".join(context_parts)

        text = self._last_heard
        if not text:
            print("[AI] No text to process, returning to LISTENING")
            self.set_state(State.LISTENING)
            return

        print(f"[AI] Processing: \"{text}\"")
        response = self.ai.get_response(text, context=context)
        print(f"[AI] Response: {response}")

        # Clear the message after processing
        self._last_heard = None

        # Speak the response (mic paused during TTS to prevent ALSA XRUN)
        self.set_state(State.SPEAKING)
        self.leds.animate_speaking()
        self._speak_safe(response)

        # Back to listening for more
        self.set_state(State.LISTENING)

    def _handle_speaking(self):
        """Speak the AI response (fallback — main speaking is in _handle_processing)."""
        self.leds.animate_speaking()

        # Wait for TTS to finish if still speaking
        while self.tts.is_speaking():
            time.sleep(0.1)

        # Back to listening for more
        self.set_state(State.LISTENING)


# ── Entry Point ──────────────────────────────────────────────────
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="T-800 Smart Assistant Brain")
    voice_opts = ", ".join(f"{k} ({v['description']})" for k, v in VOICE_PROFILES.items())
    parser.add_argument(
        "--voice", choices=list(VOICE_PROFILES.keys()), default=DEFAULT_VOICE,
        help=f"Voice profile to use (default: {DEFAULT_VOICE}). Options: {voice_opts}"
    )
    parser.add_argument(
        "--list-voices", action="store_true",
        help="List available voice profiles and exit"
    )
    args = parser.parse_args()

    if args.list_voices:
        print("Available voice profiles:")
        for key, prof in VOICE_PROFILES.items():
            exists = "✓" if os.path.exists(prof["model"]) else "✗ (not downloaded)"
            default = " [DEFAULT]" if key == DEFAULT_VOICE else ""
            print(f"  {key:10s} {prof['name']:12s} — {prof['description']} {exists}{default}")
        sys.exit(0)

    # Apply voice selection to config
    voice = VOICE_PROFILES[args.voice]
    CONFIG["piper_model"] = voice["model"]
    CONFIG["piper_sample_rate"] = voice["sample_rate"]
    CONFIG["voice_profile"] = args.voice

    # Verify running as root (needed for SPI/LED and I2C)
    if os.geteuid() != 0:
        print("⚠  WARNING: Not running as root.")
        print("   LEDs require sudo. Run: sudo python3 t800_brain.py")
        print("   Continuing anyway (LEDs will be disabled)...\n")

    brain = T800Brain(CONFIG)

    def signal_handler(sig, frame):
        brain.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    brain.run()


if __name__ == "__main__":
    main()
