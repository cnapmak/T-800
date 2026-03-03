#!/home/aleksey/t800_venv/bin/python3
# If not running inside the venv (mediapipe missing), re-exec under the venv.
import sys as _sys, os as _os
_venv_dir = _os.path.expanduser("~/t800_venv")
_venv_py  = _os.path.join(_venv_dir, "bin/python3")
if (not _os.path.realpath(_sys.prefix).startswith(_os.path.realpath(_venv_dir))
        and _os.path.exists(_venv_py)):
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
"""
T-800 Vision System
Face detection + pan/tilt servo tracking using:
  - picamera2          (Camera Module 3)
  - MediaPipe          (model 0 only — proven reliable with this camera)
  - OpenCV             (display + HUD)
  - adafruit-servokit  (PCA9685 16-ch PWM driver, I2C 0x40)
  - TF Mini Plus       (I2C distance sensor, 0x10)

Detection strategy:
  Pass 1 — full 1280×720 frame          → catches faces 0–2 m
  Pass 2 — center 50% crop → 2× upscale → catches faces 2–4 m
  Both passes use model_selection=0 and results are merged by IoU.
  Relative bounding-box coordinates are remapped after the crop pass.

Usage:
  python face_tracker.py                         # display + servos (PCA9685 ch0/ch1)
  python face_tracker.py --no-servos             # display only
  python face_tracker.py --pan-ch 0 --tilt-ch 1 # explicit channel numbers
"""

import argparse
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2

# ── Configuration ──────────────────────────────────────────────────────────────

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS          = 30

PAN_CH   = 0    # PCA9685 channel for pan servo
TILT_CH  = 1    # PCA9685 channel for tilt servo
PCA9685_ADDR = 0x40

SERVO_MIN_US = 500   # µs pulse width — wide-range servos
SERVO_MAX_US = 2500

KP        = 0.04
DEAD_ZONE = 0.05

DETECT_CONF = 0.3   # model 0 only; works at close and far range

TFMINI_I2C_BUS  = 1
TFMINI_ADDR     = 0x10
TFMINI_CMD      = [0x01, 0x02, 0x07]
TFMINI_READ_LEN = 7
TFMINI_MIN_STR  = 100
TFMINI_MAX_CM   = 1200

RED   = (0,   0,   255)
GREEN = (0,   255, 0  )
AMBER = (0,   191, 255)
GRAY  = (120, 120, 120)


# ── Unified detection container ────────────────────────────────────────────────

@dataclass
class Det:
    xmin:  float
    ymin:  float
    w:     float
    h:     float
    score: float

    @property
    def cx(self) -> float:
        return self.xmin + self.w / 2

    @property
    def cy(self) -> float:
        return self.ymin + self.h / 2


def mp_to_det(mp_det) -> Det:
    bb = mp_det.location_data.relative_bounding_box
    return Det(bb.xmin, bb.ymin, bb.width, bb.height, mp_det.score[0])


def crop_to_det(mp_det, ox: float, oy: float, scale: float) -> Det:
    """
    Remap a detection made inside a centre crop back to full-frame coordinates.
    ox, oy  — top-left corner of the crop in relative full-frame coords
    scale   — crop size as a fraction of the full frame (same in x and y)
    """
    bb = mp_det.location_data.relative_bounding_box
    return Det(
        xmin  = ox + bb.xmin  * scale,
        ymin  = oy + bb.ymin  * scale,
        w     = bb.width       * scale,
        h     = bb.height      * scale,
        score = mp_det.score[0],
    )


def det_iou(a: Det, b: Det) -> float:
    ax2, ay2 = a.xmin + a.w, a.ymin + a.h
    bx2, by2 = b.xmin + b.w, b.ymin + b.h
    ix = max(0.0, min(ax2, bx2) - max(a.xmin, b.xmin))
    iy = max(0.0, min(ay2, by2) - max(a.ymin, b.ymin))
    inter = ix * iy
    if not inter:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def merge_dets(dets: list[Det], iou_thresh: float = 0.35) -> list[Det]:
    """Sort by confidence, drop overlapping lower-confidence duplicates."""
    dets.sort(key=lambda d: d.score, reverse=True)
    kept: list[Det] = []
    for d in dets:
        if not any(det_iou(d, k) > iou_thresh for k in kept):
            kept.append(d)
    return kept



# ── TF Mini reader ─────────────────────────────────────────────────────────────

class TFMiniReader:
    # Response layout: [hdr, rsvd, dist_L, dist_H, str_L, str_H, checksum]
    # Confirmed from raw bytes [1, 0, 82, 0, 161, 0, 3] → 82 cm, strength 161
    _WINDOW = 7   # median filter window — eliminates spikes without adding lag

    def __init__(self):
        from smbus2 import SMBus, i2c_msg
        self._SMBus   = SMBus
        self._i2c_msg = i2c_msg
        self._dbuf: list[int] = []   # rolling distance buffer
        self._sbuf: list[int] = []   # rolling strength buffer
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        bus = self._SMBus(TFMINI_I2C_BUS)
        while not self._stop.is_set():
            try:
                w = self._i2c_msg.write(TFMINI_ADDR, TFMINI_CMD)
                r = self._i2c_msg.read(TFMINI_ADDR, TFMINI_READ_LEN)
                bus.i2c_rdwr(w, r)
                data = list(r)
                # Correct byte offsets — data[2]=dist_L, data[3]=dist_H
                dist = data[2] + data[3] * 256
                strn = data[4] + data[5] * 256
                with self._lock:
                    self._dbuf.append(dist)
                    self._sbuf.append(strn)
                    if len(self._dbuf) > self._WINDOW:
                        self._dbuf.pop(0)
                        self._sbuf.pop(0)
            except OSError:
                pass
            time.sleep(0.05)
        bus.close()

    @property
    def reading(self) -> tuple[int, int]:
        with self._lock:
            if not self._dbuf:
                return 0, 0
            # Median filter — rejects spikes without smoothing real movement
            dist = sorted(self._dbuf)[len(self._dbuf) // 2]
            strn = sorted(self._sbuf)[len(self._sbuf) // 2]
            return dist, strn

    @property
    def reliable(self) -> bool:
        with self._lock:
            if not self._dbuf:
                return False
            dist = sorted(self._dbuf)[len(self._dbuf) // 2]
            strn = sorted(self._sbuf)[len(self._sbuf) // 2]
            return strn >= TFMINI_MIN_STR and 0 < dist <= TFMINI_MAX_CM

    def close(self):
        self._stop.set()


# ── Servo controller (PCA9685 via adafruit-circuitpython-servokit) ─────────────

class ServoController:
    """
    Drives pan/tilt servos through a PCA9685 16-channel PWM driver over I2C.
    Internal position is kept as -1.0 … +1.0 and converted to 0–180° for the
    ServoKit API.
    """

    def __init__(self, pan_ch: int = PAN_CH, tilt_ch: int = TILT_CH,
                 address: int = PCA9685_ADDR):
        from adafruit_servokit import ServoKit
        self._kit = ServoKit(channels=16, address=address)
        self._pan_ch  = pan_ch
        self._tilt_ch = tilt_ch

        # Set pulse-width range for wide-range servos (500–2500 µs)
        self._kit.servo[pan_ch].set_pulse_width_range(SERVO_MIN_US, SERVO_MAX_US)
        self._kit.servo[tilt_ch].set_pulse_width_range(SERVO_MIN_US, SERVO_MAX_US)

        self.pan_val  = 0.0   # −1 … +1
        self.tilt_val = 0.0
        self.center()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_deg(v: float) -> float:
        """Convert -1…+1 to 0…180 degrees."""
        return (v + 1.0) * 90.0

    # ── public interface ──────────────────────────────────────────────────────

    def update(self, ex: float, ey: float) -> None:
        if abs(ex) > DEAD_ZONE:
            self.pan_val  = float(np.clip(self.pan_val  + KP * ex,  -1.0, 1.0))
        if abs(ey) > DEAD_ZONE:
            self.tilt_val = float(np.clip(self.tilt_val - KP * ey, -1.0, 1.0))
        self._kit.servo[self._pan_ch].angle  = self._to_deg(self.pan_val)
        self._kit.servo[self._tilt_ch].angle = self._to_deg(self.tilt_val)

    def center(self) -> None:
        self.pan_val = self.tilt_val = 0.0
        self._kit.servo[self._pan_ch].angle  = 90.0
        self._kit.servo[self._tilt_ch].angle = 90.0

    def close(self) -> None:
        self.center()


# ── HUD ────────────────────────────────────────────────────────────────────────

WHITE = (220, 220, 220)

def draw_hud(frame: np.ndarray, dets: list[Det],
             dist_cm: int, strength: int, reliable: bool,
             pan: float, tilt: float, servo_on: bool,
             fps: float = 0.0) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Corner brackets
    b = 28
    for bx, by, sx, sy in [(b, b, 1, 1), (w-b, b, -1, 1),
                            (b, h-b, 1, -1), (w-b, h-b, -1, -1)]:
        cv2.line(frame, (bx, by), (bx + sx*b, by),     GREEN, 2)
        cv2.line(frame, (bx, by), (bx, by + sy*b),     GREEN, 2)

    cv2.drawMarker(frame, (cx, cy), GREEN, cv2.MARKER_CROSS, 32, 1, cv2.LINE_AA)

    if dets:
        d  = dets[0]
        x  = max(0, int(d.xmin * w))
        y  = max(0, int(d.ymin * h))
        bw = int(d.w * w)
        bh = int(d.h * h)
        fx = x + bw // 2
        fy = y + bh // 2

        cv2.rectangle(frame, (x, y), (x+bw, y+bh), RED, 1)

        csz = 16
        for px, py, dx, dy in [(x, y, 1, 1), (x+bw, y, -1, 1),
                                (x, y+bh, 1, -1), (x+bw, y+bh, -1, -1)]:
            cv2.line(frame, (px, py), (px+dx*csz, py),  RED, 3)
            cv2.line(frame, (px, py), (px, py+dy*csz),  RED, 3)

        cv2.drawMarker(frame, (fx, fy), RED, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (fx, fy), RED, 1, cv2.LINE_AA)

        label = "TARGET  {:.0%}".format(d.score)
        if reliable:
            label += "  |  {:.2f} m".format(dist_cm / 100)
        cv2.putText(frame, label, (x, max(y-8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
        cv2.putText(frame, "TRACKING", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "SCANNING", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, AMBER, 2, cv2.LINE_AA)

    # ── TF Mini panel (top-right) ──────────────────────────────────────────────
    if reliable:
        range_col = GREEN
        range_txt = "{:.2f} m".format(dist_cm / 100)
    elif dist_cm > 0:
        range_col = AMBER
        range_txt = "{:.2f} m?".format(dist_cm / 100)
    else:
        range_col = GRAY
        range_txt = "-- m"

    cv2.putText(frame, "RANGE",    (w-130, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, range_col, 1, cv2.LINE_AA)
    cv2.putText(frame, range_txt,  (w-130, 58),  cv2.FONT_HERSHEY_SIMPLEX, 1.1,  range_col, 2, cv2.LINE_AA)

    # Signal strength bar
    bar_x, bar_y, bar_w, bar_hh = w-130, 68, 120, 6
    filled = int(bar_w * min(strength, 3000) / 3000)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_hh), (60, 60, 60), -1)
    if filled:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+filled, bar_y+bar_hh), range_col, -1)
    cv2.putText(frame, "STR {:d}".format(strength), (w-130, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, range_col, 1, cv2.LINE_AA)

    # ── FPS + resolution (top-left, below TRACKING/SCANNING) ─────────────────
    cv2.putText(frame, "{:.1f} FPS  {}x{}".format(fps, w, h),
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # ── Detections count ──────────────────────────────────────────────────────
    cv2.putText(frame, "FACES: {:d}".format(len(dets)),
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # ── Servo panel (bottom-left) ─────────────────────────────────────────────
    if servo_on:
        pan_deg  = (pan  + 1.0) * 90.0
        tilt_deg = (tilt + 1.0) * 90.0
        cv2.putText(frame, "PAN  {:+.2f}  ({:.0f} deg)".format(pan, pan_deg),
                    (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, "TILT {:+.2f}  ({:.0f} deg)".format(tilt, tilt_deg),
                    (10, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "SERVO: offline",
                    (10, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1, cv2.LINE_AA)

    # ── Controls hint (bottom-right) ─────────────────────────────────────────
    ctrl = "q=quit   c=center"
    tw = cv2.getTextSize(ctrl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(frame, ctrl, (w-tw-10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)


# ── Main tracker ───────────────────────────────────────────────────────────────

class T800Vision:
    def __init__(self, args: argparse.Namespace):
        # Camera — RGB888 so picamera2 gives us true RGB bytes.
        # We convert to BGR once at display time; MediaPipe receives RGB directly.
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)},
            controls={"FrameRate": FPS},
        ))

        # Single detector — model 0 is the only one that works reliably
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=DETECT_CONF,
        )

        # TF Mini
        self.tfmini: TFMiniReader | None = None
        try:
            self.tfmini = TFMiniReader()
            print("[T-800] TF Mini online")
        except Exception as e:
            print("[T-800] TF Mini unavailable:", e)

        # Servos
        self.servos: ServoController | None = None
        if not args.no_servos:
            try:
                self.servos = ServoController(args.pan_ch, args.tilt_ch)
                print("[T-800] Servos online — PCA9685  pan=ch{}  tilt=ch{}".format(
                    args.pan_ch, args.tilt_ch))
            except Exception as e:
                print("[T-800] Servo init failed:", e)

    def _detect(self, rgb: np.ndarray) -> list[Det]:
        """
        Two-pass detection with model 0:
          Pass 1 — full frame        → close range (0–2 m)
          Pass 2 — centre 50% crop, 2× upscaled → long range (2–4 m)
        """
        h, w = rgb.shape[:2]
        dets: list[Det] = []

        # Pass 1: full frame
        r1 = self.detector.process(rgb)
        dets += [mp_to_det(d) for d in (r1.detections or [])]

        # Pass 2: centre 50% crop scaled 2×
        # Crop covers relative coords 0.25→0.75 in each axis (ox=0.25, scale=0.5)
        y0, y1 = h // 4, h * 3 // 4
        x0, x1 = w // 4, w * 3 // 4
        crop    = rgb[y0:y1, x0:x1]           # 360×640
        crop_up = cv2.resize(crop, (w, h))    # → 720×1280  (2× upscale)
        r2 = self.detector.process(crop_up)
        dets += [crop_to_det(d, ox=0.25, oy=0.25, scale=0.5)
                 for d in (r2.detections or [])]

        return merge_dets(dets)

    def run(self) -> None:
        self.cam.start()
        try:
            self.cam.set_controls({"AfMode": 2, "AfSpeed": 1})  # continuous AF
        except Exception:
            pass

        print("[T-800] Vision system online.  q=quit  c=center servos")

        # FPS tracking
        _fps        = 0.0
        _fps_t      = time.time()
        _fps_frames = 0

        try:
            while True:
                # frame from picamera2 RGB888 = true RGB bytes
                frame_rgb = self.cam.capture_array()
                dets      = self._detect(frame_rgb)   # MediaPipe expects RGB

                # Servo tracking
                if dets and self.servos:
                    d  = dets[0]
                    ex = (d.cx - 0.5) * 2   # −1 … +1
                    ey = (d.cy - 0.5) * 2
                    self.servos.update(ex, ey)

                dist_cm, strength = self.tfmini.reading  if self.tfmini else (0, 0)
                reliable          = self.tfmini.reliable if self.tfmini else False
                pan  = self.servos.pan_val  if self.servos else 0.0
                tilt = self.servos.tilt_val if self.servos else 0.0

                # FPS calculation (updated every second)
                _fps_frames += 1
                now = time.time()
                if now - _fps_t >= 1.0:
                    _fps    = _fps_frames / (now - _fps_t)
                    _fps_t  = now
                    _fps_frames = 0

                # Convert RGB → BGR for OpenCV HUD drawing and display
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                draw_hud(frame_bgr, dets, dist_cm, strength, reliable,
                         pan, tilt, servo_on=self.servos is not None, fps=_fps)
                cv2.imshow("T-800 Vision", frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and self.servos:
                    self.servos.center()

        finally:
            self.cam.stop()
            cv2.destroyAllWindows()
            if self.servos:
                self.servos.close()
            if self.tfmini:
                self.tfmini.close()
            print("[T-800] Shutdown.")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-800 Vision System")
    p.add_argument("--no-servos", action="store_true",
                   help="Run without servo control (display only)")
    p.add_argument("--pan-ch",  type=int, default=PAN_CH,
                   help="PCA9685 channel for pan servo  (default: 0)")
    p.add_argument("--tilt-ch", type=int, default=TILT_CH,
                   help="PCA9685 channel for tilt servo (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    T800Vision(parse_args()).run()
