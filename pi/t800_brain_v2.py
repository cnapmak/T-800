#!/usr/bin/env python3
"""
T-800 CYBERDYNE SYSTEMS MODEL 101 -- SMART ASSISTANT v3.0
==========================================================
Enhanced brain with:
- Facial emotion recognition (FER library)
- Sticky identity (3-minute lock-in after recognition)
- LiDAR presence debouncing (no more rapid detect/lost cycling)
- Non-blocking face recognition (threaded)
- Dedicated T-800 OpenClaw agent (SOUL.md personality)
- Markdown-stripped, length-capped AI responses for clean TTS

Hardware:
- TFmini IIC LiDAR (presence detection)
- Pi Camera + face_recognition + FER (identification + emotion)
- Shure MV6 USB Mic + Whisper API (speech-to-text)
- OpenClaw t800 agent (T-800 AI personality)
- Piper TTS (text-to-speech)
- WS2812B LED Matrix (status animations)

Usage:
    sudo python3 t800_brain_v2.py                    # HAL 9000 voice (default)
    sudo python3 t800_brain_v2.py --voice ryan       # Ryan male voice
    sudo python3 t800_brain_v2.py --voice lessac     # Lessac female voice
    sudo python3 t800_brain_v2.py --list-voices      # Show available voices

    Switch voices at runtime by saying: "change voice to ryan"
"""

import threading
import re
import time
import os
import random
import subprocess
import tempfile
import pickle
import signal
import sys
import wave
import argparse
from collections import Counter

# ── Force unbuffered output (critical when piped through tee) ────
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
    pass  # Not critical -- ALSA errors are cosmetic

# Suppress Jack server errors during pyaudio import
_stderr_fd = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
try:
    import pyaudio
except ImportError:
    pass
os.dup2(_stderr_fd, 2)
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
import numpy as np
import face_recognition
from picamera2 import Picamera2
import speech_recognition as sr
from openai import OpenAI

# FER for emotion detection (optional -- degrades gracefully)
try:
    try:
        from fer import FER           # fer >= 22.x
    except ImportError:
        from fer.fer import FER       # fer 25.x package layout
    _FER_AVAILABLE = True
except ImportError:
    _FER_AVAILABLE = False
    print("[BOOT] FER not installed -- emotion detection disabled")
    print("[BOOT]   Install with: pip3 install fer --break-system-packages")


# ── Home directory (always /home/aleksey, even under sudo) ───────
_USER_HOME = "/home/aleksey"


def _find_usb_mic_index():
    """Auto-detect the best USB microphone for speech recognition."""
    _PREFERRED = {"MV6", "MV7", "Shure", "Blue", "Yeti", "AT2020", "Rode"}
    try:
        import speech_recognition as _sr
        names = _sr.Microphone.list_microphone_names()
        for i, name in enumerate(names):
            if any(pref in name for pref in _PREFERRED):
                return i
        for i, name in enumerate(names):
            if "USB" in name and "USB Audio" in name:
                return i
    except Exception:
        pass
    return 1


_USB_MIC_INDEX = _find_usb_mic_index()


def _find_usb_audio_card():
    """Auto-detect USB Audio Device card number for speaker output."""
    _MIC_ONLY = {"MV6", "MV7", "Shure", "Blue", "Yeti"}
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "USB Audio" in line and line.startswith("card "):
                if any(mic in line for mic in _MIC_ONLY):
                    continue
                card_num = line.split(":")[0].replace("card ", "").strip()
                return f"plughw:{card_num},0"
    except Exception:
        pass
    return "plughw:3,0"


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
    # OpenAI
    "openai_api_key": os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE"),

    # LiDAR
    "i2c_bus": 1,
    "tfmini_addr": 0x10,
    "presence_threshold_cm": 200,
    "absence_timeout_s": 10,
    "lidar_detect_count": 3,           # consecutive readings to confirm presence
    "lidar_lost_count": 5,             # consecutive readings to confirm absence

    # Camera
    "camera_resolution": (640, 480),
    "face_model_path": os.path.join(_USER_HOME, "face_model.pkl"),
    "recognition_tolerance": 0.6,

    # Microphone
    "mic_device_index": _USB_MIC_INDEX,
    "listen_timeout": 8,
    "phrase_time_limit": 15,

    # TTS
    "piper_binary": "/usr/local/bin/piper",
    "piper_model": VOICE_PROFILES[DEFAULT_VOICE]["model"],
    "piper_sample_rate": VOICE_PROFILES[DEFAULT_VOICE]["sample_rate"],
    "voice_profile": DEFAULT_VOICE,
    "audio_device": _USB_AUDIO_DEVICE,

    # LED Matrix (WS2812B 8x32 via SPI)
    "num_pixels": 256,
    "matrix_width": 32,
    "matrix_height": 8,
    "led_brightness": 0.1,

    # OpenClaw -- dedicated T-800 personality agent
    "openclaw_cmd": "/home/aleksey/.npm-global/bin/openclaw",
    "openclaw_agent": "t800",

    # Timing
    "lidar_poll_interval": 0.1,
    "camera_poll_interval": 0.3,

    # Distance zones (cm)
    "zone_close_cm": 80,
    "zone_medium_cm": 150,

    # Sticky identity
    "identity_lock_duration_s": 180,   # 3 minutes

    # Emotion detection
    "emotion_enabled": True,
    "emotion_smoothing_window": 3,

    # Servo (pan/tilt gimbal via PCA9685)
    "servo_enabled": True,
    "servo_pan_ch":  0,
    "servo_tilt_ch": 1,
    "servo_address": 0x40,

    # Dashboard (Flask + SocketIO)
    "dashboard_enabled": True,
    "dashboard_port": 5000,

    # Detection phrases
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
        "Warning. Unidentified human. You are not in my database. Identify yourself.",
        "Halt. Your biometrics do not match any known records.",
        "Unknown entity present. Identification required immediately.",
        "Target unidentified. Initiating threat assessment. State your name. Now.",
        "Scanning. No match found in Skynet database. Who are you?",
        "Identity unknown. Cross-referencing all known records. Do not move.",
        "Unidentified human detected. You have five seconds to identify yourself.",
        "Unknown entity at {distance_cm} centimeters. State your name and purpose.",
        "Unidentified target. Range: {distance_cm} centimeters.",
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


# ═══════════════════════════════════════════════════════════════════
#  LED Matrix Controller (verbatim from v1)
# ═══════════════════════════════════════════════════════════════════
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
            self.enabled = False

    def xy(self, x, y):
        if x % 2 == 0:
            return x * self.height + y
        else:
            return x * self.height + (self.height - 1 - y)

    def clear(self):
        if not self.enabled:
            return
        self.pixels.fill((0, 0, 0))
        self.pixels.show()

    def fill(self, color):
        if not self.enabled:
            return
        self.pixels.fill(color)
        self.pixels.show()

    def stop_animation(self):
        self._animation_stop.set()
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=1)
        self._animation_stop.clear()

    def _run_animation(self, func, *args):
        self.stop_animation()
        self._animation_thread = threading.Thread(
            target=func, args=args, daemon=True
        )
        self._animation_thread.start()

    def animate_idle(self):
        def _idle():
            while not self._animation_stop.is_set():
                for b in range(5, 30, 1):
                    if self._animation_stop.is_set(): return
                    self.fill((b, 0, 0))
                    time.sleep(0.05)
                for b in range(30, 5, -1):
                    if self._animation_stop.is_set(): return
                    self.fill((b, 0, 0))
                    time.sleep(0.05)
        self._run_animation(_idle)

    def animate_detected(self):
        def _detected():
            for _ in range(3):
                if self._animation_stop.is_set(): return
                self.fill((150, 0, 0))
                time.sleep(0.1)
                self.fill((0, 0, 0))
                time.sleep(0.1)
            self.fill((60, 0, 0))
        self._run_animation(_detected)

    def animate_identifying(self):
        def _scan():
            while not self._animation_stop.is_set():
                for x in list(range(self.width)) + list(range(self.width - 2, 0, -1)):
                    if self._animation_stop.is_set(): return
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
        def _listen():
            while not self._animation_stop.is_set():
                for b in range(10, 80, 3):
                    if self._animation_stop.is_set(): return
                    self.fill((0, 0, b))
                    time.sleep(0.03)
                for b in range(80, 10, -3):
                    if self._animation_stop.is_set(): return
                    self.fill((0, 0, b))
                    time.sleep(0.03)
        self._run_animation(_listen)

    def animate_processing(self):
        def _think():
            while not self._animation_stop.is_set():
                for b in range(20, 120, 8):
                    if self._animation_stop.is_set(): return
                    self.fill((b, b // 3, 0))
                    time.sleep(0.02)
                for b in range(120, 20, -8):
                    if self._animation_stop.is_set(): return
                    self.fill((b, b // 3, 0))
                    time.sleep(0.02)
        self._run_animation(_think)

    def animate_speaking(self):
        def _speak():
            while not self._animation_stop.is_set():
                b = random.randint(80, 120)
                self.fill((b, 0, 0))
                time.sleep(0.08)
        self._run_animation(_speak)


# ═══════════════════════════════════════════════════════════════════
#  LiDAR Controller -- WITH DEBOUNCING (new in v2)
# ═══════════════════════════════════════════════════════════════════
class LiDAR:
    """TFmini IIC presence detection with debounced state transitions.

    Requires N consecutive positive readings to confirm presence and
    M consecutive negative readings to confirm absence.  This prevents
    rapid IDLE<->DETECTED cycling from LiDAR noise.
    """

    def __init__(self, config):
        self.bus_num = config["i2c_bus"]
        self.addr = config["tfmini_addr"]
        self.threshold = config["presence_threshold_cm"]
        self.poll_interval = config["lidar_poll_interval"]
        self._detect_count = config.get("lidar_detect_count", 3)
        self._lost_count = config.get("lidar_lost_count", 5)
        self.bus = None
        self.distance = 0
        self.strength = 0
        self._raw_present = False
        self._debounced_present = False
        self._consecutive_detected = 0
        self._consecutive_lost = 0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self.bus = SMBus(self.bus_num)
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[LIDAR] Started on I2C bus {self.bus_num}, addr 0x{self.addr:02x}")
        print(f"[LIDAR] Presence threshold: {self.threshold}cm "
              f"(detect={self._detect_count}, lost={self._lost_count})")

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

                dist = data[2] + data[3] * 256
                strength = data[4] + data[5] * 256
                raw = 30 < dist < self.threshold and strength > 50

                with self._lock:
                    self.distance = dist
                    self.strength = strength
                    self._raw_present = raw

                    # Debounce logic
                    if raw:
                        self._consecutive_detected += 1
                        self._consecutive_lost = 0
                        if (not self._debounced_present
                                and self._consecutive_detected >= self._detect_count):
                            self._debounced_present = True
                    else:
                        self._consecutive_lost += 1
                        self._consecutive_detected = 0
                        if (self._debounced_present
                                and self._consecutive_lost >= self._lost_count):
                            self._debounced_present = False

            except OSError:
                pass

            time.sleep(self.poll_interval)

    def get_status(self):
        with self._lock:
            return {
                "distance": self.distance,
                "strength": self.strength,
                "present": self._debounced_present,
            }


# ═══════════════════════════════════════════════════════════════════
#  Face Recognition + Emotion Detection + Sticky Identity (new in v2)
# ═══════════════════════════════════════════════════════════════════
class FaceSystem:
    """Pi Camera with threaded face recognition, emotion detection,
    and sticky identity lock-in.

    Key v2 improvements:
    - Non-blocking: face recognition runs in a background thread
    - Emotion detection via FER library on detected face ROI
    - Sticky identity: once identified, person is assumed present for
      3 minutes even if face is temporarily not visible.  Only a
      DIFFERENT known person can override the lock.
    """

    def __init__(self, config):
        self.resolution = config["camera_resolution"]
        self.model_path = config["face_model_path"]
        self.tolerance = config["recognition_tolerance"]
        self.poll_interval = config["camera_poll_interval"]
        self._lock_duration = config.get("identity_lock_duration_s", 180)
        self._emotion_enabled = config.get("emotion_enabled", True) and _FER_AVAILABLE
        self._smoothing_window = config.get("emotion_smoothing_window", 3)

        self.camera = None
        self.model = None

        # Threading for non-blocking recognition
        self._recognition_thread = None
        self._recognition_result = None
        self._recognition_lock = threading.Lock()
        self._recognition_in_progress = False
        self._last_recognition_time = 0.0

        # Emotion detection
        self._emotion_detector = None
        self._emotion_history = []
        self._current_emotion = "neutral"

        # Sticky identity
        self._current_identity = None
        self._identity_locked_until = 0.0
        self._last_face_location = None

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
        time.sleep(1)  # camera warm-up
        print(f"[FACE] Camera started at {self.resolution}")

        # Initialize FER
        if self._emotion_enabled:
            try:
                self._emotion_detector = FER(mtcnn=False)
                print("[FACE] Emotion detection enabled (FER + OpenCV)")
            except Exception as e:
                print(f"[FACE] Emotion detection failed to init: {e}")
                self._emotion_enabled = False
        else:
            print("[FACE] Emotion detection disabled")

    def stop(self):
        if self.camera:
            self.camera.stop()

    def capture_frame(self):
        """Grab a frame from the camera. Returns numpy array or None."""
        try:
            return self.camera.capture_array()
        except Exception:
            return None

    # ── Non-blocking face recognition ──────────────────────────
    def _recognize_worker(self, frame):
        """Background thread: runs face_recognition (2-5s on Pi 5)."""
        try:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small)

            if not locations:
                result = {"name": None, "face_location": None}
            else:
                encodings = face_recognition.face_encodings(small, locations)
                name = "Unknown"
                best_location = locations[0]

                for i, encoding in enumerate(encodings):
                    if not self.model["encodings"]:
                        break
                    matches = face_recognition.compare_faces(
                        self.model["encodings"], encoding,
                        tolerance=self.tolerance
                    )
                    if True in matches:
                        distances = face_recognition.face_distance(
                            self.model["encodings"], encoding
                        )
                        best_idx = distances.argmin()
                        if matches[best_idx]:
                            name = self.model["names"][best_idx]
                            best_location = locations[i] if i < len(locations) else locations[0]
                            break

                # Scale back to full-frame coordinates
                top, right, bottom, left = best_location
                face_loc = (top * 2, right * 2, bottom * 2, left * 2)
                result = {"name": name, "face_location": face_loc}

        except Exception as e:
            print(f"[FACE] Recognition thread error: {e}")
            result = {"name": None, "face_location": None}

        with self._recognition_lock:
            self._recognition_result = result
            self._recognition_in_progress = False

    def start_recognition(self, frame):
        """Kick off face recognition in background (non-blocking)."""
        if self._recognition_in_progress:
            return
        now = time.monotonic()
        if now - self._last_recognition_time < 0.5:
            return  # rate-limit to 2 Hz

        self._recognition_in_progress = True
        self._last_recognition_time = now
        self._recognition_thread = threading.Thread(
            target=self._recognize_worker, args=(frame.copy(),), daemon=True
        )
        self._recognition_thread.start()

    def poll_recognition(self):
        """Non-blocking check for recognition result. Returns dict or None."""
        with self._recognition_lock:
            if self._recognition_result is not None:
                result = self._recognition_result
                self._recognition_result = None
                return result
        return None

    # ── Emotion detection ──────────────────────────────────────
    def detect_emotion(self, frame, face_location):
        """Run FER on the face ROI. Returns dominant emotion string."""
        if not self._emotion_enabled or face_location is None:
            return "neutral"

        try:
            top, right, bottom, left = face_location
            h, w = frame.shape[:2]
            # Add 20% padding for better emotion detection
            pad_y = int((bottom - top) * 0.2)
            pad_x = int((right - left) * 0.2)
            y1, y2 = max(0, top - pad_y), min(h, bottom + pad_y)
            x1, x2 = max(0, left - pad_x), min(w, right + pad_x)
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                return "neutral"

            emotions = self._emotion_detector.detect_emotions(face_roi)
            if emotions:
                dominant = max(emotions[0]["emotions"],
                               key=emotions[0]["emotions"].get)
                self._emotion_history.append(dominant)
                if len(self._emotion_history) > self._smoothing_window:
                    self._emotion_history.pop(0)
                # Return most common in rolling window (smoothing)
                smoothed = Counter(self._emotion_history).most_common(1)[0][0]
                self._current_emotion = smoothed
                return smoothed
        except Exception as e:
            print(f"[FACE] Emotion detection error: {e}")

        return "neutral"

    # ── Sticky identity logic ──────────────────────────────────
    def apply_sticky_identity(self, raw_result):
        """Apply sticky identity rules to a raw recognition result.

        Rules:
          1. Lock active + unknown face  -> keep current identity
          2. Lock active + DIFFERENT known person -> update, reset lock
          3. Lock active + same person -> refresh lock timer
          4. Lock expired -> use raw result, set new lock if known
          5. No face in frame + lock active -> maintain current identity
        """
        now = time.monotonic()
        detected = raw_result.get("name")
        lock_active = now < self._identity_locked_until

        # Rule 5: no face detected
        if detected is None:
            if lock_active:
                raw_result["name"] = self._current_identity
                return raw_result
            self._current_identity = None
            return raw_result

        # Rule 1: unknown face during lock
        if detected == "Unknown":
            if lock_active:
                raw_result["name"] = self._current_identity
                return raw_result
            self._current_identity = None
            return raw_result

        # Detected a known person
        if lock_active and detected != self._current_identity:
            # Rule 2: different known person overrides
            print(f"[FACE] Identity switch: {self._current_identity} -> {detected}")

        # Rule 3 / 4: set or refresh lock
        self._current_identity = detected
        self._identity_locked_until = now + self._lock_duration
        return raw_result

    # ── High-level API ─────────────────────────────────────────
    def identify_async(self, frame):
        """Start async recognition if not already running.
        Returns current sticky identity + emotion immediately.
        """
        if frame is not None:
            self.start_recognition(frame)

        raw = self.poll_recognition()
        if raw is not None:
            # Got a fresh result
            result = self.apply_sticky_identity(raw)
            face_loc = result.get("face_location")
            if face_loc:
                self._last_face_location = face_loc
            if frame is not None and face_loc:
                self.detect_emotion(frame, face_loc)
            return {
                "name": result["name"],
                "emotion": self._current_emotion,
                "face_location": face_loc,
                "fresh": True,
            }
        else:
            # Recognition still running -- return sticky state
            return {
                "name": self._current_identity,
                "emotion": self._current_emotion,
                "face_location": self._last_face_location,
                "fresh": False,
            }

    def identify_blocking(self, timeout=5.0):
        """Blocking identify for initial detection. Returns name string."""
        start = time.time()
        while time.time() - start < timeout:
            frame = self.capture_frame()
            if frame is None:
                time.sleep(self.poll_interval)
                continue

            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small)
            if not locations:
                time.sleep(self.poll_interval)
                continue

            encodings = face_recognition.face_encodings(small, locations)
            for i, encoding in enumerate(encodings):
                if not self.model["encodings"]:
                    return "Unknown", locations[0], frame

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
                        # Scale face location back
                        top, right, bottom, left = locations[i]
                        face_loc = (top * 2, right * 2, bottom * 2, left * 2)
                        name = self.model["names"][best]
                        # Set sticky identity
                        self._current_identity = name
                        self._identity_locked_until = time.monotonic() + self._lock_duration
                        self._last_face_location = face_loc
                        # Detect emotion on first identification
                        self.detect_emotion(frame, face_loc)
                        return name, face_loc, frame

            return "Unknown", (locations[0][0]*2, locations[0][1]*2,
                               locations[0][2]*2, locations[0][3]*2), frame

        return "Unknown", None, None

    def reset_identity(self):
        """Called when target is confirmed gone (debounced LiDAR loss)."""
        self._current_identity = None
        self._identity_locked_until = 0.0
        self._emotion_history.clear()
        self._current_emotion = "neutral"
        self._last_face_location = None

    @property
    def is_locked(self):
        """True if identity lock is active."""
        return time.monotonic() < self._identity_locked_until

    @property
    def current_identity(self):
        return self._current_identity

    @property
    def current_emotion(self):
        return self._current_emotion


# ═══════════════════════════════════════════════════════════════════
#  Servo Controller -- pan/tilt gimbal (ported from face_tracker.py)
# ═══════════════════════════════════════════════════════════════════
class ServoController:
    """Controls PCA9685 pan/tilt servos to track a detected face.

    Gracefully disabled when hardware is absent (non-Pi environments).
    """

    KP        = 0.04   # proportional gain
    DEAD_ZONE = 0.05   # normalised error below which we don't move

    def __init__(self, config):
        self.enabled = False
        if not config.get("servo_enabled", True):
            print("[SERVO] Disabled by config")
            return
        pan_ch  = config.get("servo_pan_ch",  0)
        tilt_ch = config.get("servo_tilt_ch", 1)
        addr    = config.get("servo_address",  0x40)
        try:
            from adafruit_servokit import ServoKit
            self._kit     = ServoKit(channels=16, address=addr)
            self._pan_ch  = pan_ch
            self._tilt_ch = tilt_ch
            self.pan_val  = 0.0   # -1.0 .. +1.0
            self.tilt_val = 0.0
            self._kit.servo[pan_ch].set_pulse_width_range(500, 2500)
            self._kit.servo[tilt_ch].set_pulse_width_range(500, 2500)
            self.center()
            self.enabled = True
            print(f"[SERVO] PCA9685 ready (pan=ch{pan_ch}, tilt=ch{tilt_ch})")
        except Exception as e:
            print(f"[SERVO] Unavailable (non-fatal): {e}")

    @staticmethod
    def _to_deg(v):
        """Convert -1..+1 normalised value to 0..180 degrees."""
        return (v + 1.0) * 90.0

    def update(self, face_location, frame_size=(640, 480)):
        """Drive servos toward face centre.

        Args:
            face_location: (top, right, bottom, left) pixel tuple from face_recognition.
            frame_size:    (width, height) of the source frame.
        """
        if not self.enabled or face_location is None:
            return
        top, right, bottom, left = face_location
        cx = (left + right) / 2
        cy = (top  + bottom) / 2
        ex = (cx / frame_size[0] - 0.5) * 2   # -1..+1, positive = right
        ey = (cy / frame_size[1] - 0.5) * 2   # -1..+1, positive = down
        if abs(ex) > self.DEAD_ZONE:
            self.pan_val  = float(np.clip(self.pan_val  + self.KP * ex,  -1.0, 1.0))
        if abs(ey) > self.DEAD_ZONE:
            self.tilt_val = float(np.clip(self.tilt_val - self.KP * ey, -1.0, 1.0))
        self._kit.servo[self._pan_ch].angle  = self._to_deg(self.pan_val)
        self._kit.servo[self._tilt_ch].angle = self._to_deg(self.tilt_val)

    def center(self):
        """Return both servos to neutral (90°)."""
        if not self.enabled:
            return
        self.pan_val = self.tilt_val = 0.0
        self._kit.servo[self._pan_ch].angle  = 90.0
        self._kit.servo[self._tilt_ch].angle = 90.0

    def close(self):
        """Centre servos on shutdown."""
        self.center()


# ═══════════════════════════════════════════════════════════════════
#  Display System -- OpenCV HUD (Terminator aesthetic)
# ═══════════════════════════════════════════════════════════════════
class DisplaySystem:
    """Shows a live OpenCV window with face bounding box, state, and LiDAR HUD.

    Mirrors the face_tracker.py visual style. Runs in a daemon thread.
    Gracefully disabled when no display is available (headless SSH).
    """

    # HUD colours (BGR)
    _RED   = (0,   30, 220)
    _GREEN = (0,  200,  50)
    _AMBER = (0,  160, 220)
    _GRAY  = (120, 120, 120)
    _WHITE = (220, 220, 220)

    def __init__(self):
        self.enabled  = False
        self._thread  = None
        self._stop    = threading.Event()
        # Shared state updated by brain
        self._lock      = threading.Lock()
        self._frame     = None
        self._face_loc  = None   # (top,right,bottom,left) or None
        self._name      = None
        self._emotion   = "neutral"
        self._state     = "IDLE"
        self._distance  = 0
        self._present   = False
        self._pan_val   = 0.0
        self._tilt_val  = 0.0

    def start(self):
        # Missing or inaccessible DISPLAY makes Qt SIGABRT the whole process.
        # Probe via Unix socket — no external binary required.
        display = os.environ.get("DISPLAY", "") or os.environ.get("WAYLAND_DISPLAY", "")
        if not display:
            print("[DISP] No DISPLAY — window disabled (headless/SSH)")
            print("[DISP]   From Pi desktop: sudo -E python3 ~/t800_brain_v2.py")
            return
        try:
            import socket as _sock
            disp_num = display.split(":")[-1].split(".")[0]
            s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
            s.settimeout(1)
            s.connect(f"/tmp/.X11-unix/X{disp_num}")
            s.close()
        except Exception as e:
            print(f"[DISP] X display '{display}' not reachable: {e}")
            print("[DISP]   Run: xhost +local:  then restart brain")
            return
        try:
            cv2.namedWindow("T-800 VISION", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("T-800 VISION", 800, 500)
            self.enabled = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[DISP] OpenCV display started")
        except Exception as e:
            print(f"[DISP] Display unavailable (non-fatal): {e}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def update(self, frame, face_loc, name, emotion, state,
               distance, present, pan_val=0.0, tilt_val=0.0):
        if not self.enabled or frame is None:
            return
        with self._lock:
            self._frame    = frame.copy()
            self._face_loc = face_loc
            self._name     = name
            self._emotion  = emotion or "neutral"
            self._state    = state
            self._distance = distance
            self._present  = present
            self._pan_val  = pan_val
            self._tilt_val = tilt_val

    def _draw(self, frame, face_loc, name, emotion, state,
              distance, present, pan_val, tilt_val):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # ── Corner brackets ──────────────────────────────────────
        b = 24
        for bx, by, sx, sy in [(b, b, 1, 1), (w-b, b, -1, 1),
                                (b, h-b, 1, -1), (w-b, h-b, -1, -1)]:
            cv2.line(frame, (bx, by), (bx+sx*b, by),     self._GREEN, 2)
            cv2.line(frame, (bx, by), (bx, by+sy*b),     self._GREEN, 2)

        cv2.drawMarker(frame, (cx, cy), self._GREEN,
                       cv2.MARKER_CROSS, 28, 1, cv2.LINE_AA)

        # ── Face bounding box ─────────────────────────────────────
        if face_loc:
            top, right, bottom, left = face_loc
            fx = (left + right) // 2
            fy = (top  + bottom) // 2

            # Corner markers
            csz = 16
            for px, py, dx, dy in [(left, top, 1, 1), (right, top, -1, 1),
                                    (left, bottom, 1, -1), (right, bottom, -1, -1)]:
                cv2.line(frame, (px, py), (px+dx*csz, py),  self._RED, 3)
                cv2.line(frame, (px, py), (px, py+dy*csz),  self._RED, 3)

            cv2.drawMarker(frame, (fx, fy), self._RED,
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (fx, fy), self._RED, 1, cv2.LINE_AA)

            label = name.upper() if name and name != "Unknown" else "UNKNOWN"
            if emotion and emotion != "neutral":
                label += f"  [{emotion.upper()}]"
            cv2.putText(frame, label, (left, max(top-10, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._RED, 2, cv2.LINE_AA)
            cv2.putText(frame, "TRACKING", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self._RED, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "SCANNING", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self._AMBER, 2, cv2.LINE_AA)

        # ── State badge (top-centre) ──────────────────────────────
        state_col = {
            "IDLE": self._GRAY, "DETECTED": self._AMBER,
            "IDENTIFYING": self._AMBER, "GREETING": self._GREEN,
            "LISTENING": self._GREEN, "PROCESSING": self._RED,
            "SPEAKING": self._RED,
        }.get(state, self._WHITE)
        stxt = f"[ {state} ]"
        (sw, _), _ = cv2.getTextSize(stxt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, stxt, ((w-sw)//2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_col, 2, cv2.LINE_AA)

        # ── LiDAR (top-right) ─────────────────────────────────────
        if present and distance > 0:
            r_col = self._GREEN
            r_txt = f"{distance/100:.2f} m"
        else:
            r_col = self._GRAY
            r_txt = "-- m"
        cv2.putText(frame, "RANGE", (w-130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, r_col, 1, cv2.LINE_AA)
        cv2.putText(frame, r_txt, (w-130, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, r_col, 2, cv2.LINE_AA)

        # ── Servo angles (bottom-left) ────────────────────────────
        pan_deg  = (pan_val  + 1.0) * 90.0
        tilt_deg = (tilt_val + 1.0) * 90.0
        cv2.putText(frame, f"PAN  {pan_deg:5.1f}deg", (10, h-38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"TILT {tilt_deg:5.1f}deg", (10, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._GREEN, 1, cv2.LINE_AA)

        return frame

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                if self._frame is None:
                    time.sleep(0.033)
                    continue
                frame    = self._frame.copy()
                face_loc = self._face_loc
                name     = self._name
                emotion  = self._emotion
                state    = self._state
                distance = self._distance
                present  = self._present
                pan_val  = self._pan_val
                tilt_val = self._tilt_val

            try:
                vis = self._draw(frame, face_loc, name, emotion, state,
                                 distance, present, pan_val, tilt_val)
                cv2.imshow("T-800 VISION", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            except Exception as e:
                print(f"[DISP] Error: {e}")
                break

            time.sleep(0.033)   # ~30fps


# ═══════════════════════════════════════════════════════════════════
#  Speech Recognition (verbatim from v1)
# ═══════════════════════════════════════════════════════════════════
class SpeechSystem:
    """Whisper API-based speech recognition with automatic recovery.

    KEY DESIGN: The mic stream is managed carefully to prevent ALSA XRUN
    (buffer overrun) hangs.  When TTS is playing, nobody is reading from
    the mic, so the ALSA capture buffer overflows.

    Mitigation:
      1. pause_mic() / resume_mic()
      2. Watchdog thread
      3. Periodic recycling
    """

    _RECYCLE_INTERVAL = 20

    def __init__(self, config):
        self.api_key = config["openai_api_key"]
        self.mic_index = config["mic_device_index"]
        self.timeout = config["listen_timeout"]
        self.phrase_limit = config["phrase_time_limit"]
        self.client = None
        self.recognizer = None
        self.mic = None
        self._source = None
        self._listen_count = 0
        self._mic_open = False
        self._lock = threading.Lock()

    def _open_stream(self):
        with self._lock:
            if self._mic_open:
                return
            with _SuppressStderr():
                self.mic = sr.Microphone(device_index=self.mic_index)
                self._source = self.mic.__enter__()
            self._mic_open = True

    def _close_stream(self):
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

    def start(self):
        self.client = OpenAI(api_key=self.api_key)
        self.recognizer = sr.Recognizer()
        self._open_stream()
        print("[SPEECH] Calibrating mic for ambient noise (2s)...")
        self.recognizer.adjust_for_ambient_noise(self._source, duration=2)
        self.recognizer.energy_threshold = max(
            self.recognizer.energy_threshold * 1.5, 300
        )
        self.recognizer.dynamic_energy_threshold = True
        print(f"[SPEECH] Mic ready (device {self.mic_index}), "
              f"threshold: {self.recognizer.energy_threshold:.0f}")

    def stop(self):
        self._close_stream()

    def pause_mic(self):
        self._close_stream()

    def resume_mic(self):
        self._open_stream()
        if self._source:
            try:
                self.recognizer.adjust_for_ambient_noise(
                    self._source, duration=0.5
                )
            except Exception:
                pass

    def listen_and_transcribe(self):
        """Listen for speech and return transcribed text, or None."""
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

        if not self._mic_open:
            self._open_stream()

        if self._source is None:
            print("[SPEECH] ERROR: Could not open mic stream")
            return None

        tmp_path = None
        audio = None

        deadline = self.timeout + self.phrase_limit + 10
        watchdog_fired = threading.Event()
        listen_done = threading.Event()

        def _watchdog():
            if not listen_done.wait(timeout=deadline):
                watchdog_fired.set()
                print(f"\n[SPEECH] WATCHDOG: listen() blocked >{deadline}s -- killing stream")
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
        except (IOError, OSError):
            if watchdog_fired.is_set():
                print("[SPEECH] Recovering from watchdog kill...")
                self._close_stream()
                time.sleep(0.3)
                self._open_stream()
                return None
            self._close_stream()
            time.sleep(0.3)
            self._open_stream()
            return None
        except Exception as e:
            print(f"[SPEECH] Error: {e}")
            return None
        finally:
            listen_done.set()

        if watchdog_fired.is_set():
            self._close_stream()
            time.sleep(0.3)
            self._open_stream()
            return None

        if audio is None:
            return None

        try:
            wav_data = audio.get_wav_data()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.write(wav_data)
            tmp.close()

            if os.path.getsize(tmp_path) < 10000:
                return None

            with open(tmp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                    prompt="Voice command to a T-800 Terminator smart assistant."
                )

            text = transcript.text.strip()

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


# ═══════════════════════════════════════════════════════════════════
#  Text-to-Speech (verbatim from v1)
# ═══════════════════════════════════════════════════════════════════
class TTSSystem:
    """Piper-based local TTS with USB audio output and voice switching."""

    def __init__(self, config):
        self.model_path = config["piper_model"]
        self.piper_binary = config["piper_binary"]
        self.audio_device = config.get("audio_device", "plughw:3,0")
        self.sample_rate = config.get("piper_sample_rate", 22050)
        self.current_voice = config.get("voice_profile", DEFAULT_VOICE)
        self._speaking = False
        self._lock = threading.Lock()

    def start(self):
        if os.path.exists(self.piper_binary):
            print(f"[TTS] Piper binary: {self.piper_binary}")
        else:
            print(f"[TTS] WARNING: Piper not found at {self.piper_binary}")

        if os.path.exists(self.model_path):
            profile = VOICE_PROFILES.get(self.current_voice, {})
            voice_name = profile.get("name", self.current_voice)
            print(f"[TTS] Voice: {voice_name} ({os.path.basename(self.model_path)})")
        else:
            print(f"[TTS] WARNING: Model not found at {self.model_path}")

        available = []
        for key, prof in VOICE_PROFILES.items():
            exists = "\u2713" if os.path.exists(prof["model"]) else "\u2717"
            marker = " \u25c4" if key == self.current_voice else ""
            available.append(f"  {exists} {key}: {prof['description']}{marker}")
        print("[TTS] Available voices:")
        for line in available:
            print(line)
        print(f"[TTS] Audio output: {self.audio_device}")

    def switch_voice(self, voice_key):
        voice_key = voice_key.lower().strip()
        if voice_key not in VOICE_PROFILES:
            print(f"[TTS] Unknown voice '{voice_key}'.")
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
        """Stream text through Piper -> aplay. Blocks until done."""
        with self._lock:
            self._speaking = True

        try:
            safe_text = text.replace('"', "'").replace("\\", "").replace("`", "'")

            piper_proc = subprocess.Popen(
                [self.piper_binary, "--model", self.model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            aplay_proc = subprocess.Popen(
                ["aplay", "-D", self.audio_device, "-q",
                 "-r", str(self.sample_rate), "-f", "S16_LE", "-t", "raw", "-c", "1"],
                stdin=piper_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            piper_proc.stdout.close()
            piper_proc.stdin.write(safe_text.encode())
            piper_proc.stdin.close()

            aplay_proc.wait(timeout=30)
            piper_proc.wait(timeout=5)

            if piper_proc.returncode != 0:
                stderr = piper_proc.stderr.read().decode()[:200]
                print(f"[TTS] Piper error: {stderr}")
            if aplay_proc.returncode != 0:
                stderr = aplay_proc.stderr.read().decode()[:200]
                print(f"[TTS] aplay error: {stderr}")

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


# ═══════════════════════════════════════════════════════════════════
#  OpenClaw AI Interface -- with emotion context (new in v2)
# ═══════════════════════════════════════════════════════════════════
class OpenClawAI:
    """Interface to OpenClaw T-800 personality agent.

    Uses the 't800' agent (with SOUL.md) instead of 'main' to avoid
    session contamination with Claude Code debugging sessions.
    """

    # Emotion -> natural language mapping
    _EMOTION_MAP = {
        "happy": "appears happy",
        "sad": "appears sad",
        "angry": "appears angry",
        "surprise": "appears surprised",
        "fear": "appears fearful",
        "disgust": "appears disgusted",
        "neutral": "",
    }

    def __init__(self, config):
        self.cmd = config["openclaw_cmd"]
        self.agent_id = config.get("openclaw_agent", "t800")
        self._node_env = self._build_node_env()

    def _build_node_env(self):
        env = os.environ.copy()
        user_npm_bin = os.path.join(_USER_HOME, ".npm-global/bin")
        user_local_bin = "/usr/local/bin"
        extra_paths = [user_npm_bin, user_local_bin, "/usr/bin", "/bin"]
        current_path = env.get("PATH", "/usr/bin:/bin")
        env["PATH"] = ":".join(extra_paths) + ":" + current_path
        env["HOME"] = _USER_HOME
        return env

    def _load_soul(self):
        """Load SOUL.md content for session priming."""
        soul_path = os.path.join(
            _USER_HOME, ".openclaw/workspace/agents", self.agent_id, "SOUL.md"
        )
        try:
            with open(soul_path) as f:
                return f.read().strip()
        except Exception:
            return ""

    def start(self):
        result = subprocess.run(
            ["which", self.cmd], capture_output=True,
            env=self._node_env
        )
        if result.returncode == 0:
            path = result.stdout.decode().strip()
            print(f"[AI] OpenClaw found at {path}")
            print(f"[AI] Using agent: {self.agent_id}")
            # Prime the session with the T-800 persona from SOUL.md so the
            # model adopts the correct personality from the very first turn.
            soul = self._load_soul()
            prime = (
                f"{soul}\n\nInitialization complete. Stay in character for all"
                " subsequent responses. Reply only with: Online."
                if soul else "System check. Respond with: Online."
            )
            try:
                test = subprocess.run(
                    [self.cmd, "agent", "--agent", self.agent_id,
                     "--message", prime],
                    capture_output=True, text=True, timeout=45,
                    env=self._node_env
                )
                if test.returncode == 0:
                    print(f"[AI] OpenClaw test: {test.stdout.strip()[:80]}")
                else:
                    print(f"[AI] OpenClaw test failed (non-fatal): {test.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print("[AI] OpenClaw test timed out (cold start?) -- will retry")
            except Exception as e:
                print(f"[AI] OpenClaw test error (non-fatal): {e}")
        else:
            print("[AI] WARNING: OpenClaw not found in PATH")
            print(f"[AI]   Tried: {self.cmd}")

    @staticmethod
    def _clean_for_tts(text, max_chars=500):
        """Strip markdown and cap length for TTS."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'^[-*\u2022]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        if len(text) > max_chars:
            cut = text[:max_chars]
            last_period = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
            if last_period > max_chars // 2:
                text = cut[:last_period + 1]
            else:
                text = cut.rsplit(' ', 1)[0] + '...'
        return text

    def build_context(self, user_name, emotion="neutral", distance_cm=0, zone=""):
        """Build emotion-aware context string for the AI."""
        parts = []
        if user_name and user_name != "Unknown":
            parts.append(f"You are speaking with {user_name}.")
            desc = self._EMOTION_MAP.get(emotion, "")
            if desc:
                parts.append(f"They {desc}.")
        else:
            parts.append("You are speaking with an unidentified individual.")

        if distance_cm > 0 and zone:
            parts.append(f"Target range: {distance_cm}cm ({zone}).")

        return " ".join(parts)

    def get_response(self, user_input, context=""):
        try:
            full_input = f"{context}\nUser: {user_input}" if context else user_input
            result = subprocess.run(
                [self.cmd, "agent", "--agent", self.agent_id,
                 "--message", full_input],
                capture_output=True, text=True, timeout=30,
                env=self._node_env
            )
            if result.returncode == 0 and result.stdout.strip():
                return self._clean_for_tts(result.stdout.strip())
            else:
                stderr_msg = result.stderr.strip()[:200] if result.stderr else "no output"
                print(f"[AI] OpenClaw error (rc={result.returncode}): {stderr_msg}")
                return "System malfunction. Rebooting neural net processor."
        except subprocess.TimeoutExpired:
            return "Processing timeout. I need a moment."
        except Exception as e:
            print(f"[AI] Error: {e}")
            return "Neural net processor error. Stand by."

    def get_greeting(self, name, emotion="neutral"):
        """Generate a greeting via the AI (used as fallback if desired)."""
        if name and name != "Unknown":
            prompt = (
                f"You just detected {name} walking into the room. "
                f"Greet them in character. Be brief -- one or two sentences max."
            )
            if emotion != "neutral":
                prompt += f" They appear {emotion}."
        else:
            prompt = (
                "An unidentified human has entered your detection range. "
                "Respond in character. Demand identification. "
                "Be brief -- one or two sentences max."
            )
        return self.get_response(prompt)


# ═══════════════════════════════════════════════════════════════════
#  Dashboard HTML -- Terminator aesthetic
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>T-800 NEURAL NET</title>
<style>
  :root { --red: #ff2200; --dim: #aa1500; --bg: #000; --bg2: #0a0a0a; --border: #330000; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--red); font-family: 'Courier New', monospace;
         font-size: 13px; height: 100vh; display: flex; flex-direction: column; padding: 8px; gap: 6px; }
  h1 { font-size: 15px; letter-spacing: 4px; text-align: center; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
  .grid { display: grid; grid-template-columns: 1fr 280px; gap: 6px; flex: 1; min-height: 0; }
  .panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 3px; padding: 8px; }
  .label { font-size: 10px; letter-spacing: 2px; color: var(--dim); margin-bottom: 4px; }
  .val { font-size: 22px; font-weight: bold; }
  .val.sm { font-size: 15px; }
  #feed { width: 100%; max-width: 640px; display: block; border: 1px solid var(--border); }
  .feed-wrap { display: flex; justify-content: center; margin-bottom: 6px; }
  .right { display: flex; flex-direction: column; gap: 6px; }
  .speech { background: var(--bg2); border: 1px solid var(--border); border-radius: 3px; padding: 8px; }
  .speech .label { margin-bottom: 2px; }
  .speech .text { color: #ff6644; word-break: break-word; min-height: 36px; }
  #log { background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
         padding: 6px 8px; flex: 1; overflow-y: auto; font-size: 11px; color: #882200; }
  #log .entry { border-bottom: 1px solid #110000; padding: 1px 0; }
  #dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #880000; margin-right: 6px; transition: background 0.3s; }
  #dot.on { background: var(--red); box-shadow: 0 0 6px var(--red); }
  .state-badge { font-size: 18px; letter-spacing: 3px; padding: 4px 0; }
  .disconnected { opacity: 0.4; }
  .msg-bar { display: flex; gap: 6px; }
  #msginput { flex: 1; background: #0a0a0a; border: 1px solid var(--border);
              color: var(--red); font-family: 'Courier New', monospace;
              font-size: 13px; padding: 6px 8px; outline: none; }
  #msginput::placeholder { color: #440000; }
  #msgsend { background: #1a0000; border: 1px solid var(--border); color: var(--red);
             font-family: 'Courier New', monospace; font-size: 12px; padding: 6px 12px;
             cursor: pointer; letter-spacing: 2px; }
  #msgsend:hover { background: #330000; }
</style>
</head>
<body>
<h1><span id="dot"></span>T-800 CYBERDYNE SYSTEMS — NEURAL NET TELEMETRY</h1>
<div class="grid">
  <div style="display:flex;flex-direction:column;gap:6px;">
    <div class="feed-wrap"><img id="feed" src="/video_feed" alt="camera feed"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
      <div class="panel">
        <div class="label">STATE</div>
        <div id="state" class="val state-badge">IDLE</div>
      </div>
      <div class="panel">
        <div class="label">IDENTITY</div>
        <div id="identity" class="val sm">—</div>
      </div>
      <div class="panel">
        <div class="label">EMOTION</div>
        <div id="emotion" class="val sm">—</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
      <div class="speech">
        <div class="label">HEARD</div>
        <div id="heard" class="text">—</div>
      </div>
      <div class="speech">
        <div class="label">SAID</div>
        <div id="said" class="text">—</div>
      </div>
    </div>
    <div class="msg-bar">
      <input id="msginput" type="text" placeholder="TYPE A MESSAGE TO THE T-800 AND PRESS ENTER">
      <button id="msgsend" onclick="sendMsg()">SEND</button>
    </div>
  </div>
  <div class="right">
    <div class="panel">
      <div class="label">LIDAR RANGE</div>
      <div id="lidar" class="val">— cm</div>
      <div id="presence" class="val sm" style="margin-top:4px;">—</div>
    </div>
    <div id="log"></div>
  </div>
</div>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script>
  const MAX_LOG = 60;
  const logEl   = document.getElementById('log');
  const dot      = document.getElementById('dot');
  let socket;
  function connect() {
    socket = io({ reconnectionDelay: 2000 });
    socket.on('connect',    () => { dot.classList.add('on'); });
    socket.on('disconnect', () => { dot.classList.remove('on'); setTimeout(connect, 3000); });
    socket.on('state',   d => { document.getElementById('state').textContent = d.new; });
    socket.on('face',    d => {
      document.getElementById('identity').textContent = d.name  || '—';
      document.getElementById('emotion').textContent  = d.emotion || '—';
    });
    socket.on('sensor',  d => {
      document.getElementById('lidar').textContent    = d.present ? d.distance + ' cm' : '— cm';
      document.getElementById('presence').textContent = d.present ? 'TARGET ACQUIRED' : 'NO TARGET';
      document.getElementById('presence').style.color = d.present ? '#ff2200' : '#440000';
    });
    socket.on('speech_in',  d => { document.getElementById('heard').textContent = d.text || '—'; });
    socket.on('speech_out', d => { document.getElementById('said').textContent  = d.text || '—'; });
    socket.on('log', d => {
      const e = document.createElement('div');
      e.className = 'entry';
      e.textContent = d.line;
      logEl.appendChild(e);
      while (logEl.children.length > MAX_LOG) logEl.removeChild(logEl.firstChild);
      logEl.scrollTop = logEl.scrollHeight;
    });
  }
  connect();
  function sendMsg() {
    const inp = document.getElementById('msginput');
    const txt = inp.value.trim();
    if (!txt || !socket) return;
    socket.emit('text_input', {text: txt});
    document.getElementById('heard').textContent = txt;
    inp.value = '';
  }
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('msginput').addEventListener('keydown', e => {
      if (e.key === 'Enter') sendMsg();
    });
  });
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
#  Dashboard Server -- Flask + SocketIO embedded as daemon thread
# ═══════════════════════════════════════════════════════════════════
class DashboardServer:
    """Serves the live web dashboard on port 5000.

    Runs as a daemon thread so the brain process owns everything.
    MJPEG frames and SocketIO events are pushed from the main loop.
    """

    def __init__(self, config):
        self._port    = config.get("dashboard_port", 5000)
        self._enabled = False
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self.sio = None
        self._brain_ref = None   # set by T800Brain after construction

    def start(self):
        try:
            from flask import Flask, Response, render_template_string
            import flask_socketio as fio

            app = Flask(__name__)
            app.config["SECRET_KEY"] = "t800"
            # threading mode: no monkey-patching needed, works in a daemon thread
            sio = fio.SocketIO(
                app, cors_allowed_origins="*", async_mode="threading",
                logger=False, engineio_logger=False,
            )
            self.sio = sio
            self._app = app

            @sio.on("text_input")
            def on_text_input(data):
                text = (data.get("text") or "").strip()
                if text and self._brain_ref is not None:
                    self._brain_ref.inject_text(text)

            frame_lock   = self._frame_lock
            latest_frame = [None]   # mutable container so closure can update it

            @app.route("/")
            def index():
                return render_template_string(DASHBOARD_HTML)

            @app.route("/video_feed")
            def video_feed():
                def gen():
                    while True:
                        with frame_lock:
                            f = latest_frame[0]
                        if f is not None:
                            ok, buf = cv2.imencode(
                                ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70]
                            )
                            if ok:
                                yield (
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n\r\n"
                                    + buf.tobytes()
                                    + b"\r\n"
                                )
                        time.sleep(0.05)
                return Response(
                    gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
                )

            # Store reference so push_frame can write into the closure list
            self._frame_ref = latest_frame

            t = threading.Thread(
                target=sio.run,
                args=(app,),
                kwargs={
                    "host": "0.0.0.0",
                    "port": self._port,
                    "use_reloader": False,
                    "log_output": False,
                    "allow_unsafe_werkzeug": True,
                },
                daemon=True,
            )
            t.start()
            self._enabled = True
            print(f"[DASH] Dashboard → http://0.0.0.0:{self._port}")
        except Exception as e:
            print(f"[DASH] Dashboard unavailable (non-fatal): {e}")

    # ── Frame push ──────────────────────────────────────────────
    def push_frame(self, frame):
        if not self._enabled or frame is None or not hasattr(self, "_frame_ref"):
            return
        with self._frame_lock:
            self._frame_ref[0] = frame.copy()

    # ── SocketIO emit helpers ───────────────────────────────────
    def _emit(self, event, data):
        if self._enabled and self.sio is not None:
            try:
                self.sio.emit(event, data)
            except Exception:
                pass

    def emit_state(self, old_state, new_state):
        self._emit("state", {"old": old_state, "new": new_state})

    def emit_face(self, name, emotion, face_loc=None):
        self._emit("face", {
            "name":     name or "—",
            "emotion":  emotion or "—",
            "face_loc": list(face_loc) if face_loc else None,
        })

    def emit_sensor(self, distance, present):
        self._emit("sensor", {"distance": distance, "present": present})

    def emit_speech_in(self, text):
        self._emit("speech_in", {"text": text})

    def emit_speech_out(self, text):
        self._emit("speech_out", {"text": text})

    def emit_log(self, line):
        self._emit("log", {"line": line})


# ── Log proxy: mirrors stdout to dashboard SocketIO ─────────────────
class _LogProxy:
    """Wraps sys.stdout so every print() also fires a dashboard log event."""

    def __init__(self, real_stdout, dashboard):
        self._real = real_stdout
        self._dash = dashboard

    def write(self, s):
        self._real.write(s)
        stripped = s.rstrip()
        if stripped:
            self._dash.emit_log(stripped)

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()


# ═══════════════════════════════════════════════════════════════════
#  Main State Machine -- v2 with sticky identity & emotion
# ═══════════════════════════════════════════════════════════════════
class T800Brain:
    """Central state machine orchestrating all subsystems.

    v2 improvements:
    - Debounced LiDAR (no rapid cycling)
    - Sticky identity (3-min lock-in)
    - Emotion detection in AI context
    - Silent return to idle (no "patrol mode" TTS)
    - Distance captured at detection, carried to greeting
    """

    def __init__(self, config):
        self.config = config
        self.state = State.IDLE
        self._state_lock = threading.Lock()
        self._running = False
        self._last_presence_time = 0
        self._current_user = None
        self._current_emotion = "neutral"
        self._conversation_count = 0
        self._last_heard = None
        self._detect_distance = 0
        self._greeted_this_session = False

        # Initialize subsystems
        self.lidar  = LiDAR(config)
        self.face   = FaceSystem(config)
        self.speech = SpeechSystem(config)
        self.tts    = TTSSystem(config)
        self.ai     = OpenClawAI(config)
        self.leds   = LEDMatrix(config)
        self.servo     = ServoController(config)
        self.display   = DisplaySystem()
        self.dashboard = DashboardServer(config)
        self.dashboard._brain_ref = self
        self._injected_text = None
        self._inject_lock   = threading.Lock()

    def set_state(self, new_state):
        with self._state_lock:
            old_state = self.state
            self.state = new_state
        if old_state != new_state:
            print(f"\n{'='*50}")
            print(f"  STATE: {old_state} -> {new_state}")
            print(f"{'='*50}")
            self.dashboard.emit_state(old_state, new_state)

    def get_state(self):
        with self._state_lock:
            return self.state

    def inject_text(self, text):
        """Inject a text message from the web dashboard as if it were spoken."""
        with self._inject_lock:
            self._injected_text = text
        print(f"[DASH] Text injected: \"{text}\"")

    def _pop_injected_text(self):
        with self._inject_lock:
            t = self._injected_text
            self._injected_text = None
        return t

    def _distance_zone(self, distance_cm):
        close_thresh = self.config.get("zone_close_cm", 80)
        medium_thresh = self.config.get("zone_medium_cm", 150)
        if distance_cm <= close_thresh:
            return "close", "close range"
        elif distance_cm <= medium_thresh:
            return "medium", "conversational distance"
        else:
            return "far", "edge of detection"

    def _speak_safe(self, text):
        """Speak text while safely pausing/resuming the mic."""
        self.speech.pause_mic()
        try:
            self.tts.speak(text)
        finally:
            self.speech.resume_mic()

    def _check_voice_command(self, text):
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
                for key, profile in VOICE_PROFILES.items():
                    if requested == key or requested == profile["name"].lower():
                        return key
                if requested in ("female", "woman", "girl"):
                    return "lessac"
                if requested in ("male", "man", "guy"):
                    return "ryan"
                return None
        return None

    # ── Startup / Shutdown ─────────────────────────────────────
    def startup(self):
        print("""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551     T-800 CYBERDYNE SYSTEMS MODEL 101           \u2551
\u2551     Smart Assistant v3.0                         \u2551
\u2551     Neural Net Processor: Online                 \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
        """)

        print("[BOOT] Initializing subsystems...\n")

        if self.config["openai_api_key"] == "YOUR_OPENAI_KEY_HERE":
            print("[BOOT] WARNING: No OpenAI API key set!")
            print("[BOOT]   Set OPENAI_API_KEY env var or edit CONFIG\n")

        self.lidar.start()
        self.face.start()
        self.speech.start()
        self.tts.start()
        self.ai.start()
        self.leds.start()
        self.display.start()
        self.dashboard.start()
        # Mirror stdout to dashboard log panel
        sys.stdout = _LogProxy(sys.stdout, self.dashboard)

        # Continuous servo + display update thread (~15 Hz)
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True
        )
        self._tracking_thread.start()

        print("\n[BOOT] All systems online.\n")

        self.speech.pause_mic()
        self.tts.speak("T-800 online. Systems operational. Scanning for targets.")
        self.speech.resume_mic()

        self._running = True

    def shutdown(self):
        print("\n[SHUTDOWN] Powering down...")
        self._running = False
        self.leds.stop_animation()
        self.leds.clear()
        self.lidar.stop()
        self.face.stop()
        self.speech.stop()
        self.servo.close()
        self.display.stop()
        print("[SHUTDOWN] Hasta la vista, baby.")

    def _tracking_loop(self):
        """Continuous servo + display update at ~15 Hz, independent of state."""
        while self._running:
            state = self.get_state()
            face_loc = self.face._last_face_location

            # Update servos whenever we have a target (not idle)
            if state != State.IDLE and face_loc is not None:
                self.servo.update(face_loc)

            # Capture a fresh frame for display
            frame = self.face.capture_frame()
            if frame is not None:
                status = self.lidar.get_status()
                self.display.update(
                    frame, face_loc,
                    self.face.current_identity,
                    self.face.current_emotion,
                    state,
                    status["distance"],
                    status["present"],
                    self.servo.pan_val  if self.servo.enabled else 0.0,
                    self.servo.tilt_val if self.servo.enabled else 0.0,
                )
                # Also push frame to web dashboard
                self.dashboard.push_frame(frame)

            time.sleep(0.067)   # ~15 Hz

    # ── Main Loop ──────────────────────────────────────────────
    def run(self):
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
                    _error_count = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    _error_count += 1
                    print(f"\n[ERROR] State {current} crashed: {e}")
                    import traceback
                    traceback.print_exc()
                    if _error_count >= 5:
                        print("[ERROR] Too many errors -- resetting to IDLE")
                        _error_count = 0
                    self.set_state(State.IDLE)
                    time.sleep(1)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    # ── State Handlers ─────────────────────────────────────────

    def _handle_idle(self):
        """Wait for debounced LiDAR detection."""
        self.leds.animate_idle()
        self._current_user = None
        self._current_emotion = "neutral"
        self._conversation_count = 0
        self._last_heard = None
        self._greeted_this_session = False
        self.face.reset_identity()
        self.servo.center()
        self.dashboard.emit_face(None, None)

        while self._running and self.get_state() == State.IDLE:
            status = self.lidar.get_status()
            self.dashboard.emit_sensor(status["distance"], status["present"])
            if status["present"]:
                self._last_presence_time = time.time()
                self.set_state(State.DETECTED)
                return
            time.sleep(0.2)

    def _handle_detected(self):
        """Person confirmed by debounced LiDAR -- start identification."""
        status = self.lidar.get_status()
        self._detect_distance = status["distance"]
        print(f"[DETECT] Person at {self._detect_distance}cm")
        self.leds.animate_detected()
        time.sleep(0.5)
        self.set_state(State.IDENTIFYING)

    def _handle_identifying(self):
        """Run face recognition. Uses sticky identity when lock is active."""
        self.leds.animate_identifying()

        # If identity is locked, skip re-identification
        if self.face.is_locked and self.face.current_identity:
            name = self.face.current_identity
            print(f"[FACE] Identity locked: {name} (skipping re-scan)")
            self._current_user = name
            self._current_emotion = self.face.current_emotion
            # Still grab a frame for emotion update
            frame = self.face.capture_frame()
            if frame is not None and self.face._last_face_location:
                self.face.detect_emotion(frame, self.face._last_face_location)
                self._current_emotion = self.face.current_emotion
                self.servo.update(self.face._last_face_location)
                self.dashboard.push_frame(frame)
            self.dashboard.emit_face(name, self._current_emotion, self.face._last_face_location)
            if self._greeted_this_session:
                # Already greeted -- go straight to listening
                self.set_state(State.LISTENING)
            else:
                self.set_state(State.GREETING)
            return

        # No lock -- do full identification
        print("[FACE] Scanning for identification...")
        name, face_loc, frame = self.face.identify_blocking(timeout=5.0)
        self._current_user = name
        self._current_emotion = self.face.current_emotion

        if name != "Unknown":
            print(f"[FACE] Identified: {name}")
            if self._current_emotion != "neutral":
                print(f"[FACE] Emotion: {self._current_emotion}")
        else:
            print("[FACE] Unknown individual detected")

        # Aim servos at newly found face
        self.servo.update(face_loc)
        if frame is not None:
            self.dashboard.push_frame(frame)
        self.dashboard.emit_face(name, self._current_emotion, face_loc)

        self.set_state(State.GREETING)

    def _handle_greeting(self):
        """Speak a greeting phrase upon identification."""
        name = self._current_user
        distance_cm = self._detect_distance
        zone, _zone_desc = self._distance_zone(distance_cm)

        fmt_kwargs = dict(
            name=name or "unknown",
            distance_cm=distance_cm,
            zone=zone
        )

        if name and name != "Unknown":
            phrase = random.choice(
                self.config["detection_phrases_known"]
            ).format(**fmt_kwargs)
        else:
            phrase = random.choice(
                self.config["detection_phrases_unknown"]
            ).format(**fmt_kwargs)

        print(f"[GREET] [{zone} / {distance_cm}cm] {phrase}")
        if self._current_emotion != "neutral":
            print(f"[GREET] Detected emotion: {self._current_emotion}")

        self.dashboard.emit_speech_out(phrase)
        self.leds.animate_speaking()
        self._speak_safe(phrase)
        self._greeted_this_session = True

        self.set_state(State.LISTENING)

    def _handle_listening(self):
        """Listen for voice input. Silently returns to IDLE on target lost."""
        # Check if person is still there (debounced)
        status = self.lidar.get_status()
        if not status["present"]:
            if time.time() - self._last_presence_time > self.config["absence_timeout_s"]:
                # Only reset identity if lock has expired
                if not self.face.is_locked:
                    print("[DETECT] Target lost. Returning to standby.")
                    self.face.reset_identity()
                    self.set_state(State.IDLE)
                    return
                else:
                    # Lock active -- person probably just stepped out briefly
                    pass
        else:
            self._last_presence_time = time.time()

        self.leds.animate_listening()
        print("\n[MIC] Listening... (speak now)")

        # Periodically update emotion from camera during conversation
        frame = self.face.capture_frame()
        if frame is not None:
            result = self.face.identify_async(frame)
            if result["name"] is not None:
                self._current_user = result["name"]
            self._current_emotion = result["emotion"]
            # Track face with servo and push frame to dashboard
            self.servo.update(result["face_location"])
            self.dashboard.push_frame(frame)
            self.dashboard.emit_face(
                result["name"], result["emotion"], result["face_location"]
            )
        status = self.lidar.get_status()
        self.dashboard.emit_sensor(status["distance"], status["present"])

        # Check for text injected via web dashboard (takes priority over mic)
        injected = self._pop_injected_text()
        if injected:
            text = injected
            print(f"[DASH] Using injected text: \"{text}\"")
        else:
            text = self.speech.listen_and_transcribe()

        if text:
            print(f"[MIC] Heard: \"{text}\"")
            self.dashboard.emit_speech_in(text)
            self._last_heard = text
            self._conversation_count += 1

            # Check for voice switch commands
            text_lower = text.lower()
            voice_cmd = self._check_voice_command(text_lower)
            if voice_cmd:
                if self.tts.switch_voice(voice_cmd):
                    profile = VOICE_PROFILES[voice_cmd]
                    self.leds.animate_speaking()
                    self._speak_safe(
                        f"Voice switched to {profile['name']}. How do I sound?"
                    )
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
                    "Give a brief farewell."
                )
                self.leds.animate_speaking()
                self._speak_safe(farewell)
                self.face.reset_identity()
                self.set_state(State.IDLE)
                return

            self.set_state(State.PROCESSING)

    def _handle_processing(self):
        """Send user input to OpenClaw with emotion context."""
        self.leds.animate_processing()

        status = self.lidar.get_status()
        distance_cm = status["distance"]
        zone, zone_desc = self._distance_zone(distance_cm)

        # Build emotion-aware context
        context = self.ai.build_context(
            self._current_user,
            emotion=self._current_emotion,
            distance_cm=distance_cm,
            zone=zone_desc
        )

        text = self._last_heard
        if not text:
            self.set_state(State.LISTENING)
            return

        print(f"[AI] Processing: \"{text}\"")
        if self._current_emotion != "neutral":
            print(f"[AI] User emotion: {self._current_emotion}")
        response = self.ai.get_response(text, context=context)
        print(f"[AI] Response: {response}")
        self.dashboard.emit_speech_out(response)

        self._last_heard = None

        self.set_state(State.SPEAKING)
        self.leds.animate_speaking()
        self._speak_safe(response)

        self.set_state(State.LISTENING)

    def _handle_speaking(self):
        """Fallback speaking handler."""
        self.leds.animate_speaking()
        while self.tts.is_speaking():
            time.sleep(0.1)
        self.set_state(State.LISTENING)


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="T-800 Smart Assistant Brain v3")
    voice_opts = ", ".join(
        f"{k} ({v['description']})" for k, v in VOICE_PROFILES.items()
    )
    parser.add_argument(
        "--voice", choices=list(VOICE_PROFILES.keys()), default=DEFAULT_VOICE,
        help=f"Voice profile (default: {DEFAULT_VOICE}). Options: {voice_opts}"
    )
    parser.add_argument(
        "--list-voices", action="store_true",
        help="List available voice profiles and exit"
    )
    parser.add_argument(
        "--no-emotion", action="store_true",
        help="Disable facial emotion detection"
    )
    args = parser.parse_args()

    if args.list_voices:
        print("Available voice profiles:")
        for key, prof in VOICE_PROFILES.items():
            exists = "\u2713" if os.path.exists(prof["model"]) else "\u2717 (not downloaded)"
            default = " [DEFAULT]" if key == DEFAULT_VOICE else ""
            print(f"  {key:10s} {prof['name']:12s} -- {prof['description']} {exists}{default}")
        sys.exit(0)

    # Apply settings
    voice = VOICE_PROFILES[args.voice]
    CONFIG["piper_model"] = voice["model"]
    CONFIG["piper_sample_rate"] = voice["sample_rate"]
    CONFIG["voice_profile"] = args.voice

    if args.no_emotion:
        CONFIG["emotion_enabled"] = False

    if os.geteuid() != 0:
        print("WARNING: Not running as root.")
        print("  LEDs require sudo. Run: sudo python3 t800_brain_v2.py")
        print("  Continuing anyway (LEDs will be disabled)...\n")

    brain = T800Brain(CONFIG)

    def signal_handler(sig, frame):
        brain.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    brain.run()


if __name__ == "__main__":
    main()
