#!/usr/bin/env python3
"""
Car Detection -> MAX7219 LED Matrix Shadow System
=================================================

Two 32x8 MAX7219 modules on CE0 + CE1, sitting side by side = one logical 64x8 strip.
The full camera width is mapped linearly across all 64 columns:
  cols  0-31  -> Module on CE0  (left half of strip)
  cols 32-63  -> Module on CE1  (right half of strip)

Servo control:
  PAN  servo (left/right) -> keyboard ONLY  <- / -> arrow keys  (GPIO17)
  TILT servo (up/down)    -> gyro ONLY      MPU6050 pitch     (GPIO27)

Keyboard controls (global — no need to focus preview window):
  <-  arrow   pan left  (-5° per press, capped at -20°)
  ->  arrow   pan right (+5° per press, capped at +20°)
  C          centre pan back to 0°
  Q          quit
"""

import time
import math
import threading
import logging
import sys
from collections import deque

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas

from mpu6050 import mpu6050
from pynput import keyboard as pynput_keyboard

# -- gpiozero Hardware PWM Setup -----------------------------------------------
from gpiozero import AngularServo, Device
from gpiozero.pins.pigpio import PiGPIOFactory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("CarLED")

try:
    # Force gpiozero to use the hardware DMA via the pigpio daemon
    Device.pin_factory = PiGPIOFactory()
    log.info("Hardware PWM (pigpio) successfully enabled.")
except Exception as e:
    log.error("CRITICAL: Could not connect to the pigpio daemon.")
    log.error("Make sure you have compiled pigpio and run: sudo pigpiod")
    sys.exit(1)


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
MODEL_PATH   = "model/detect.tflite"
LABEL_PATH   = "model/labelmap.txt"
CAMERA_INDEX = 0
FRAME_W      = 320
FRAME_H      = 240

CLASS_THRESHOLDS = {
    "car":        0.40,
    "motorcycle": 0.50,
    "truck":      0.60,
    "bus":        0.70,
}
CAR_CLASSES = set(CLASS_THRESHOLDS.keys())

INFER_EVERY       = 2
NMS_IOU_THRESHOLD = 0.45

# Total logical display = 64 cols x 8 rows across both modules
TOTAL_COLS   = 64     # 32 (CE0) + 32 (CE1)
MODULE_COLS  = 32     # columns per physical module
DISPLAY_ROWS = 8

# Center 2 rows ALWAYS ON (Headlight core beam)
PROTECTED    = {3, 4}  

HISTORY_LEN    = 6
VOTE_THRESHOLD = 3

DARK_THRESHOLD = 70
DIM_THRESHOLD  = 110
LED_BRIGHT_MIN = 32
LED_BRIGHT_MAX = 220

SPI_CASCADE   = 4     # number of MAX7219 ICs per CE line (4 ICs = one 32x8 module)
WARMUP_FRAMES = 20

SHOW_PREVIEW  = True
PREVIEW_SCALE = 2

BOX_COLOURS = {
    "car":        (0,   255,   0),
    "truck":      (255, 128,   0),
    "bus":        (0,   128, 255),
    "motorcycle": (0,   255, 255),
}

# -- Servo ---------------------------------------------------------------------
SERVO_PAN_PIN   = 17
SERVO_TILT_PIN  = 27
SERVO_MIN_PULSE = 0.0005   # 0.5 ms -> -90°
SERVO_MAX_PULSE = 0.0025   # 2.5 ms -> +90°

# --- UPDATED SERVO LIMITS AND SPEEDS ---
MAX_PAN_ANGLE     = 20.0   # Capped at +/- 20 degrees
MAX_TILT_ANGLE    = 90.0
PAN_STEP_DEG      = 5.0
TILT_DEAD_ZONE    = 1.5    # Reduced from 3.0 for finer gyro response
TILT_MIN_INTERVAL = 0.05   # Reduced from 0.15 for faster servo updates

# -- MPU6050 -------------------------------------------------------------------
MPU_ADDRESS = 0x68


# ------------------------------------------------------------------------------
# LABEL LOADER
# ------------------------------------------------------------------------------
def load_labels(path):
    labels = {}
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                l = line.strip()
                if l:
                    labels[i] = l.lower()
        log.info("Loaded %d labels | sample: %s",
                 len(labels), {k: labels[k] for k in list(labels)[:8]})
    except FileNotFoundError:
        log.warning("Label file not found — using built-in COCO map.")
        labels = {
            0: "background", 1: "person",     2: "bicycle",
            3: "car",        4: "motorcycle", 5: "airplane",
            6: "bus",        7: "train",      8: "truck",   9: "boat",
        }
    return labels


# ------------------------------------------------------------------------------
# NMS
# ------------------------------------------------------------------------------
def apply_nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    if not detections:
        return detections
    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_threshold]
    return kept

def _iou(a, b):
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ub = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------------------
# IMAGE ENHANCER
# ------------------------------------------------------------------------------
class ImageEnhancer:
    def __init__(self):
        self.clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._gamma_luts = {}

    def process(self, bgr):
        luma = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        out  = self._white_balance(bgr)
        if luma < DARK_THRESHOLD:
            out = self._gamma(out, 0.45)
            out = self._clahe_lab(out)
            out = self._denoise(out)
            out = self._unsharp(out, 1.4)
        elif luma < DIM_THRESHOLD:
            out = self._gamma(out, 0.65)
            out = self._clahe_lab(out)
            out = self._unsharp(out, 1.2)
        else:
            out = self._unsharp(out, 0.8)
        return out

    @staticmethod
    def _white_balance(bgr):
        b, g, r = cv2.split(bgr.astype(np.float32))
        bm, gm, rm = b.mean(), g.mean(), r.mean()
        glo = (bm + gm + rm) / 3.0
        if bm > 0: b *= glo / bm
        if gm > 0: g *= glo / gm
        if rm > 0: r *= glo / rm
        return cv2.merge([np.clip(b, 0, 255).astype(np.uint8),
                          np.clip(g, 0, 255).astype(np.uint8),
                          np.clip(r, 0, 255).astype(np.uint8)])

    def _gamma(self, bgr, gamma):
        if gamma not in self._gamma_luts:
            inv = 1.0 / gamma
            self._gamma_luts[gamma] = np.array(
                [min(255, int((i / 255.0) ** inv * 255)) for i in range(256)],
                dtype=np.uint8)
        return cv2.LUT(bgr, self._gamma_luts[gamma])

    def _clahe_lab(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self.clahe.apply(l), a, b]),
                            cv2.COLOR_LAB2BGR)

    @staticmethod
    def _unsharp(bgr, strength=1.0):
        blur = cv2.GaussianBlur(bgr, (5, 5), 1.0)
        return cv2.addWeighted(bgr, 1.0 + strength, blur, -strength, 0)

    @staticmethod
    def _denoise(bgr):
        return cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 15)


# ------------------------------------------------------------------------------
# CAR DETECTOR
# ------------------------------------------------------------------------------
class CarDetector:
    def __init__(self, model_path, labels):
        self.labels   = labels
        self.enhancer = ImageEnhancer()
        self.interp   = Interpreter(model_path=model_path, num_threads=4)
        self.interp.allocate_tensors()

        inp = self.interp.get_input_details()[0]
        self.input_idx = inp["index"]
        self.input_h   = inp["shape"][1]
        self.input_w   = inp["shape"][2]
        self.is_quant  = (inp["dtype"] == np.uint8)

        out = self.interp.get_output_details()
        self.out_boxes  = out[0]["index"]
        self.out_cls    = out[1]["index"]
        self.out_scores = out[2]["index"]
        self.out_count  = out[3]["index"]
        log.info("Model loaded | input %dx%d | quantised=%s",
                 self.input_w, self.input_h, self.is_quant)

    def detect(self, bgr):
        enhanced = self.enhancer.process(bgr)
        rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        resized  = cv2.resize(rgb, (self.input_w, self.input_h))
        inp_data = resized.astype(np.uint8) if self.is_quant \
                   else (resized / 255.0).astype(np.float32)
        self.interp.set_tensor(self.input_idx, np.expand_dims(inp_data, 0))
        self.interp.invoke()

        boxes   = self.interp.get_tensor(self.out_boxes)[0]
        classes = self.interp.get_tensor(self.out_cls)[0]
        scores  = self.interp.get_tensor(self.out_scores)[0]
        count   = int(self.interp.get_tensor(self.out_count)[0])

        results = []
        for i in range(count):
            raw_cls = int(classes[i])
            label   = self.labels.get(raw_cls + 1, None)
            if label is None or label not in CAR_CLASSES:
                label = self.labels.get(raw_cls, "unknown")
            if label not in CAR_CLASSES:
                continue
            score = float(scores[i])
            if score < CLASS_THRESHOLDS.get(label, 0.50):
                continue
            y1, x1, y2, x2 = boxes[i]
            results.append({
                "label": label, "conf": score,
                "x1": float(np.clip(x1, 0, 1)), "y1": float(np.clip(y1, 0, 1)),
                "x2": float(np.clip(x2, 0, 1)), "y2": float(np.clip(y2, 0, 1)),
            })
        return apply_nms(results)


# ------------------------------------------------------------------------------
# COLUMN MAPPER
# Maps normalised bbox x (0.0–1.0) -> logical cols 0-63
# ------------------------------------------------------------------------------
class ColumnMapper:
    @staticmethod
    def bbox_to_cols(x1, x2):
        lc1 = max(0,          int(x1 * TOTAL_COLS))
        lc2 = min(TOTAL_COLS, int(x2 * TOTAL_COLS) + 1)
        logical = set(range(lc1, lc2))
        cols_ce0 = {c             for c in logical if c <  MODULE_COLS}
        cols_ce1 = {c - MODULE_COLS for c in logical if c >= MODULE_COLS}
        return cols_ce0, cols_ce1


# ------------------------------------------------------------------------------
# COLUMN SMOOTHER  (one instance per module)
# ------------------------------------------------------------------------------
class ColumnSmoother:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_LEN)

    def update(self, cols):
        self.history.append(cols)
        if len(self.history) < VOTE_THRESHOLD:
            return set()
        votes = {}
        for fc in self.history:
            for c in fc:
                votes[c] = votes.get(c, 0) + 1
        return {c for c, v in votes.items() if v >= VOTE_THRESHOLD}


# ------------------------------------------------------------------------------
# LED CONTROLLER
# CE0 = left module  (logical cols  0-31, local cols 0-31)
# CE1 = right module (logical cols 32-63, local cols 0-31)
# ------------------------------------------------------------------------------
class LEDController:
    def __init__(self):
        ser_ce0 = spi(port=0, device=0, gpio=noop(), cascaded=SPI_CASCADE)
        ser_ce1 = spi(port=0, device=1, gpio=noop(), cascaded=SPI_CASCADE)
        self.dev_ce0 = max7219(ser_ce0, cascaded=SPI_CASCADE,
                               block_orientation=-90, rotate=0)
        self.dev_ce1 = max7219(ser_ce1, cascaded=SPI_CASCADE,
                               block_orientation=-90, rotate=0)
        self.dev_ce0.contrast(50)  
        self.dev_ce1.contrast(50)
        self._lock = threading.Lock()
        
        # State cache to prevent redundant SPI writes
        self._last_state = None 
        
        log.info("LED controller ready | CE0=cols 0-31 (left) | CE1=cols 32-63 (right)")

    def _draw_module(self, draw, shadow_cols, car_detected, invert=False):
        """Render one 32x8 module.
        Protected rows are always ON.
        If a column is shadowed, it turns completely black (except protected rows).
        Otherwise, all pixels are ON.
        """
        for r in range(DISPLAY_ROWS):
            for c in range(MODULE_COLS):
                # Calculate physical coordinates based on inversion
                target_c = (MODULE_COLS - 1 - c) if invert else c
                target_r = (DISPLAY_ROWS - 1 - r) if invert else r
                
                if r in PROTECTED:
                    draw.point((target_c, target_r), fill="white")   # Center rows ALWAYS ON
                elif not car_detected:
                    draw.point((target_c, target_r), fill="white")   # No car = all ON
                elif c not in shadow_cols:
                    draw.point((target_c, target_r), fill="white")   # Outside shadow = ON
                # If it IS a shadow col, do nothing (pixel remains OFF/Black)

    def apply_shadow_cols(self, shadow_ce0, shadow_ce1, car_detected):
        # Freeze sets for quick, immutable state comparison
        current_state = (frozenset(shadow_ce0), frozenset(shadow_ce1), car_detected)
        
        with self._lock:
            if self._last_state == current_state:
                return  # State hasn't changed; skip the expensive canvas and SPI write
                
            self._last_state = current_state
            
            with canvas(self.dev_ce0) as draw:
                # Left side (CE0) is physically inverted
                self._draw_module(draw, shadow_ce0, car_detected, invert=True)
            with canvas(self.dev_ce1) as draw:
                # Right side (CE1) is normal
                self._draw_module(draw, shadow_ce1, car_detected, invert=False)

    def all_on(self):
        with self._lock:
            self._last_state = None  # Invalidate cache
            with canvas(self.dev_ce0) as draw:
                for r in range(DISPLAY_ROWS):
                    for c in range(MODULE_COLS):
                        draw.point((c, r), fill="white")
            with canvas(self.dev_ce1) as draw:
                for r in range(DISPLAY_ROWS):
                    for c in range(MODULE_COLS):
                        draw.point((c, r), fill="white")

    def all_off(self):
        with self._lock:
            self._last_state = None  # Invalidate cache
            with canvas(self.dev_ce0) as draw:
                pass
            with canvas(self.dev_ce1) as draw:
                pass

    def set_brightness(self, level):
        self.dev_ce0.contrast(level)
        self.dev_ce1.contrast(level)


# ------------------------------------------------------------------------------
# AUTO BRIGHTNESS
# ------------------------------------------------------------------------------
class AutoBrightness:
    def __init__(self, led):
        self.led   = led
        self._last = -1

    def update(self, bgr):
        luma  = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        level = int(LED_BRIGHT_MIN + (luma / 255.0) * (LED_BRIGHT_MAX - LED_BRIGHT_MIN))
        if abs(level - self._last) > 8:
            self.led.set_brightness(level)
            self._last = level


# ------------------------------------------------------------------------------
# IMU  (mpu6050 library, background thread)
# ------------------------------------------------------------------------------
class IMUReader:
    def __init__(self, address=MPU_ADDRESS):
        try:
            self._sensor = mpu6050(address)
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to MPU6050: {e}. Check wiring/I2C.") from e
        self._pitch   = 0.0
        self._ema     = None
        self._lock    = threading.Lock()
        self._running = False
        log.info("MPU6050 ready")

    def _read_pitch(self):
        d = self._sensor.get_accel_data()
        x, y, z = d['x'], d['y'], d['z']
        return math.degrees(math.atan2(y, math.sqrt(x * x + z * z)))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        log.info("IMU polling started")

    def _loop(self):
        # --- UPDATED EMA ALPHA FOR FASTER RESPONSE ---
        EMA_ALPHA = 0.35  
        while self._running:
            try:
                p = self._read_pitch()
                with self._lock:
                    if self._ema is None:
                        self._ema = p
                    else:
                        self._ema = EMA_ALPHA * p + (1 - EMA_ALPHA) * self._ema
                    self._pitch = self._ema
            except Exception as e:
                log.warning("IMU error: %s", e)
            time.sleep(0.05)

    def stop(self):
        self._running = False

    def get_pitch(self):
        with self._lock:
            return self._pitch


# ------------------------------------------------------------------------------
# SERVO CONTROLLER
# ------------------------------------------------------------------------------
class ServoController:
    def __init__(self):
        self._pan  = AngularServo(SERVO_PAN_PIN,
                                  min_angle=-90, max_angle=90,
                                  min_pulse_width=SERVO_MIN_PULSE,
                                  max_pulse_width=SERVO_MAX_PULSE)
        self._tilt = AngularServo(SERVO_TILT_PIN,
                                  min_angle=-90, max_angle=90,
                                  min_pulse_width=SERVO_MIN_PULSE,
                                  max_pulse_width=SERVO_MAX_PULSE)
        self._pan_angle   = 0.0
        self._tilt_angle  = 0.0
        self._last_tilt_t = 0.0
        self._lock        = threading.Lock()
        self._pan.angle   = 0
        self._tilt.angle  = 0
        log.info("Servos ready | PAN=GPIO%d keyboard | TILT=GPIO%d gyro",
                 SERVO_PAN_PIN, SERVO_TILT_PIN)

    def pan_left(self):
        with self._lock:
            self._pan_angle = max(-MAX_PAN_ANGLE, self._pan_angle - PAN_STEP_DEG)
            self._pan.angle = self._pan_angle
        log.debug("Pan: %.1f°", self._pan_angle)

    def pan_right(self):
        with self._lock:
            self._pan_angle = min(MAX_PAN_ANGLE, self._pan_angle + PAN_STEP_DEG)
            self._pan.angle = self._pan_angle
        log.debug("Pan: %.1f°", self._pan_angle)

    def pan_centre(self):
        with self._lock:
            self._pan_angle = 0.0
            self._pan.angle = 0
        log.info("Pan centred.")

    def get_pan_angle(self):
        with self._lock:
            return self._pan_angle

    def update_tilt(self, pitch_deg):
        now = time.monotonic()
        with self._lock:
            target = max(-MAX_TILT_ANGLE, min(MAX_TILT_ANGLE, pitch_deg))
            change = abs(target - self._tilt_angle)
            elapsed = now - self._last_tilt_t
            if change >= TILT_DEAD_ZONE and elapsed >= TILT_MIN_INTERVAL:
                self._tilt_angle  = target
                self._tilt.angle  = target
                self._last_tilt_t = now

    def get_tilt_angle(self):
        with self._lock:
            return self._tilt_angle

    def stop(self):
        with self._lock:
            self._pan.angle  = 0
            self._tilt.angle = 0
            time.sleep(0.3)
            self._pan.detach()
            self._tilt.detach()
        log.info("Servos detached.")


# ------------------------------------------------------------------------------
# FPS COUNTER
# ------------------------------------------------------------------------------
class FPSCounter:
    def __init__(self, n=30):
        self.t = deque(maxlen=n)

    def tick(self):
        self.t.append(time.monotonic())

    @property
    def fps(self):
        return 0.0 if len(self.t) < 2 else (len(self.t) - 1) / (self.t[-1] - self.t[0])


# ------------------------------------------------------------------------------
# PREVIEW RENDERER
# ------------------------------------------------------------------------------
def draw_preview(frame, detections, fps, infer_fps, luma,
                 shadow_ce0, shadow_ce1, pitch, pan_angle, tilt_angle):
    h, w = frame.shape[:2]
    out  = frame.copy()

    # Detection boxes
    for d in detections:
        x1p = int(d["x1"] * w); y1p = int(d["y1"] * h)
        x2p = int(d["x2"] * w); y2p = int(d["y2"] * h)
        col = BOX_COLOURS.get(d["label"], (255, 255, 255))
        cv2.rectangle(out, (x1p, y1p), (x2p, y2p), col, 2)
        txt = f'{d["label"]} {d["conf"]:.0%}'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1p, y1p - th - 6), (x1p + tw + 4, y1p), col, -1)
        cv2.putText(out, txt, (x1p + 2, y1p - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Shadow column overlay across full camera width
    col_w   = w / TOTAL_COLS
    overlay = out.copy()
    for lc in range(TOTAL_COLS):
        local_col = lc if lc < MODULE_COLS else lc - MODULE_COLS
        shadow    = shadow_ce0 if lc < MODULE_COLS else shadow_ce1
        if local_col in shadow:
            px = int(lc * col_w)
            cv2.rectangle(overlay, (px, 0), (int(px + col_w), h), (0, 100, 220), -1)
    cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

    # Module boundary
    bx = int(MODULE_COLS * col_w)
    cv2.line(out, (bx, 0), (bx, h), (80, 80, 80), 1)
    cv2.putText(out, "CE0", (4,       h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 255), 1)
    cv2.putText(out, "CE1", (bx + 4,  h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 180), 1)

    # HUD
    mode = "DARK" if luma < DARK_THRESHOLD else "DIM" if luma < DIM_THRESHOLD else "DAY"
    mc   = (80, 80, 255) if mode == "DARK" else (80, 200, 255) if mode == "DIM" else (80, 255, 80)
    hud  = [
        (f"FPS:{fps:.1f} inf:{infer_fps:.1f}",                  (255, 255, 255)),
        (mode,                                                  mc),
        (f"cars:{len(detections)}",                             (255, 255, 255)),
        (f"TILT(gyro) {tilt_angle:+.1f}° pitch:{pitch:+.1f}°", (100, 220, 255)),
        (f"PAN (keys) {pan_angle:+.1f}°",                      (255, 220, 100)),
        ("<- -> pan   C=centre   Q=quit",                      (180, 180, 180)),
    ]
    for i, (txt, colour) in enumerate(hud):
        cv2.putText(out, txt, (4, 14 + i * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)

    # LED grid preview
    cell = 4; gap = 1; step = cell + gap
    gw   = TOTAL_COLS  * step
    gh   = DISPLAY_ROWS * step
    ox   = max(0, w // 2 - gw // 2)
    oy   = h - gh - 10

    cv2.rectangle(out, (ox - 2, oy - 2), (ox + gw + 2, oy + gh + 2), (30, 30, 30), -1)
    mid = ox + MODULE_COLS * step
    cv2.line(out, (mid, oy), (mid, oy + gh), (60, 60, 60), 1)

    for r in range(DISPLAY_ROWS):
        for lc in range(TOTAL_COLS):
            local_col = lc if lc < MODULE_COLS else lc - MODULE_COLS
            shadow    = shadow_ce0 if lc < MODULE_COLS else shadow_ce1
            px = ox + lc * step
            py = oy + r  * step
            if r in PROTECTED:
                colour = (255, 255, 255)
            elif local_col in shadow:
                colour = (25, 25, 25) # Pure black/dark for shadow, no pattern
            else:
                colour = (200, 200, 200)
            cv2.rectangle(out, (px, py), (px + cell - 1, py + cell - 1), colour, -1)

    cv2.putText(out, "CE0", (ox,      oy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 180, 255), 1)
    cv2.putText(out, "CE1", (mid + 2, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 255, 180), 1)

    # Pan position bar
    bar_x = 4; bar_y = h - 6; bar_w = w - 8; bar_h = 4
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cx    = bar_x + bar_w // 2
    
    # Use MAX_PAN_ANGLE for the bar visual scaling to match our new limits
    ind_x = max(bar_x, min(bar_x + bar_w,
                           cx + int((pan_angle / MAX_PAN_ANGLE) * (bar_w // 2))))
    cv2.circle(out, (ind_x, bar_y + bar_h // 2), 3, (255, 220, 100), -1)
    cv2.line(out,   (cx,    bar_y), (cx, bar_y + bar_h), (150, 150, 150), 1)

    return out


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------
class CarLEDPipeline:
    def __init__(self):
        log.info("Initialising ...")
        labels                = load_labels(LABEL_PATH)
        self.detector         = CarDetector(MODEL_PATH, labels)
        self.led              = LEDController()
        self.col_smoother_ce0 = ColumnSmoother()
        self.col_smoother_ce1 = ColumnSmoother()
        self.mapper           = ColumnMapper()
        self.auto_br          = AutoBrightness(self.led)
        self.fps              = FPSCounter()
        self.infer_fps        = FPSCounter()
        self._quit            = False

        # -- Camera ------------------------------------------------------------
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

        log.info("Camera warm-up (%d frames) ...", WARMUP_FRAMES)
        for _ in range(WARMUP_FRAMES):
            ok, _ = self.cap.read()
            if not ok:
                time.sleep(0.05)

        for attempt in range(10):
            ok, test_frame = self.cap.read()
            if ok and test_frame is not None and test_frame.size > 0:
                log.info("Camera ready at %dx%d",
                         test_frame.shape[1], test_frame.shape[0])
                break
            log.warning("Camera not ready (attempt %d) ...", attempt + 1)
            time.sleep(0.2)
        else:
            raise RuntimeError("Camera failed to produce a valid frame")

        # -- IMU + servos ------------------------------------------------------
        self.imu    = IMUReader()
        self.servos = ServoController()
        self.imu.start()

        # -- Global keyboard listener ------------------------------------------
        self._kb_listener = pynput_keyboard.Listener(on_press=self._on_key)
        self._kb_listener.start()

        log.info("Ready | <- -> pan  C=centre  Q=quit")

    def _on_key(self, key):
        try:
            if   key == pynput_keyboard.Key.left:  self.servos.pan_left()
            elif key == pynput_keyboard.Key.right:  self.servos.pan_right()
            elif hasattr(key, 'char'):
                if   key.char in ('c', 'C'): self.servos.pan_centre()
                elif key.char in ('q', 'Q'): self._quit = True
        except AttributeError:
            pass

    def run(self):
        if SHOW_PREVIEW:
            cv2.namedWindow("Car Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Car Detection",
                             FRAME_W * PREVIEW_SCALE, FRAME_H * PREVIEW_SCALE)
        self.led.all_on()

        last_log        = time.monotonic()
        frame_count     = 0
        last_detections = []
        last_shadow_ce0 = set()
        last_shadow_ce1 = set()

        try:
            while not self._quit:
                ok, frame = self.cap.read()
                if not ok or frame is None or frame.size == 0:
                    log.warning("Empty frame — retrying ...")
                    time.sleep(0.05)
                    continue

                frame_count += 1

                # -- Inference -------------------------------------------------
                if frame_count % INFER_EVERY == 0:
                    last_detections = self.detector.detect(frame)
                    self.infer_fps.tick()
                    raw_ce0, raw_ce1 = set(), set()
                    for d in last_detections:
                        c0, c1 = self.mapper.bbox_to_cols(d["x1"], d["x2"])
                        raw_ce0 |= c0
                        raw_ce1 |= c1
                    last_shadow_ce0 = self.col_smoother_ce0.update(raw_ce0)
                    last_shadow_ce1 = self.col_smoother_ce1.update(raw_ce1)

                # -- LEDs ------------------------------------------------------
                car_detected = len(last_detections) > 0
                self.led.apply_shadow_cols(last_shadow_ce0, last_shadow_ce1,
                                           car_detected)
                self.auto_br.update(frame)

                # -- Tilt servo ------------------------------------------------
                pitch = self.imu.get_pitch()
                self.servos.update_tilt(pitch)

                self.fps.tick()

                # -- Preview ---------------------------------------------------
                if SHOW_PREVIEW:
                    luma = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    prev = draw_preview(
                        frame, last_detections,
                        self.fps.fps, self.infer_fps.fps, luma,
                        last_shadow_ce0, last_shadow_ce1,
                        pitch,
                        self.servos.get_pan_angle(),
                        self.servos.get_tilt_angle())
                    prev = cv2.resize(prev,
                                      (FRAME_W * PREVIEW_SCALE, FRAME_H * PREVIEW_SCALE),
                                      interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Car Detection", prev)
                    cv2.waitKey(1)

                # -- Status log ------------------------------------------------
                now = time.monotonic()
                if now - last_log >= 5.0:
                    luma = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    mode = "DARK" if luma < DARK_THRESHOLD else \
                           "DIM"  if luma < DIM_THRESHOLD  else "DAY"
                    log.info(
                        "FPS %.1f (inf %.1f) | cars %d | "
                        "pan %.1f° | tilt %.1f° (pitch %.1f°) | %s",
                        self.fps.fps, self.infer_fps.fps, len(last_detections),
                        self.servos.get_pan_angle(),
                        self.servos.get_tilt_angle(), pitch, mode)
                    last_log = now

        except KeyboardInterrupt:
            log.info("Stopped by Ctrl+C.")
        finally:
            self.led.all_on()
            self.imu.stop()
            self.servos.stop()
            self._kb_listener.stop()
            self.cap.release()
            if SHOW_PREVIEW:
                cv2.destroyAllWindows()
            log.info("Shutdown complete.")


if __name__ == "__main__":
    CarLEDPipeline().run()