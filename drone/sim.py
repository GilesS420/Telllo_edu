"""
FakeTello: a tiny simulator with the same methods we use from djitellopy.

It renders a synthetic downward camera image of a floor with puddles at known
positions, so the whole chain (Jetson path -> flight -> detection -> puddle
coordinates back to the Jetson) can be tested on a laptop without a drone:
    python tello_gui.py --sim
"""

import math
import threading
import time

import numpy as np

from . import config as cfg

# Puddles in the simulated world: (x, y, radius_x, radius_y) in cm, mission frame
SIM_PUDDLES = [(150, 0, 25, 15), (200, 120, 20, 20), (50, 150, 20, 12)]
SIM_IMG_W, SIM_IMG_H = 320, 240


class _FrameRead:
    def __init__(self, tello):
        self._tello = tello

    @property
    def frame(self):
        return self._tello.render()


class FakeTello:
    RESPONSE_TIMEOUT = 7

    def __init__(self, speedup=1.0):
        self.speedup = speedup
        self._lock = threading.Lock()
        self.x = self.y = 0.0
        self.z = 0.0
        self.is_flying = False
        self.battery = 95
        rng = np.random.default_rng(0)
        self._noise = rng.integers(-12, 12, (SIM_IMG_H, SIM_IMG_W)).astype(np.int16)

    # --- connection / video ---
    def connect(self):
        print("[SIM] connected")

    def get_battery(self):
        return self.battery

    def streamon(self):
        pass

    def streamoff(self):
        pass

    def end(self):
        pass

    def send_command_with_return(self, cmd, *a, **kw):
        print(f"[SIM] {cmd}")
        return "ok"

    def get_frame_read(self):
        return _FrameRead(self)

    # --- flight ---
    def takeoff(self):
        time.sleep(1 / self.speedup)
        with self._lock:
            self.z, self.is_flying = 80.0, True

    def land(self):
        time.sleep(1 / self.speedup)
        with self._lock:
            self.z, self.is_flying = 0.0, False

    def emergency(self):
        self.land()

    def send_rc_control(self, *a):
        pass

    def go_xyz_speed(self, x, y, z, speed):
        """Body frame move; the simulated drone keeps yaw 0 relative to its start."""
        if max(abs(x), abs(y), abs(z)) < 20:
            raise Exception("Command 'go' was unsuccessful: out of range")
        start = (self.x, self.y, self.z)
        dist = math.sqrt(x * x + y * y + z * z)
        duration = dist / speed / self.speedup
        t0 = time.time()
        while True:
            f = min((time.time() - t0) / duration, 1.0)
            with self._lock:
                self.x, self.y, self.z = start[0] + x * f, start[1] + y * f, start[2] + z * f
            if f >= 1.0:
                break
            time.sleep(0.02)
        self.battery = max(self.battery - 1, 0)

    def get_distance_tof(self):
        return int(self.z)

    def get_height(self):
        return int(self.z)

    # --- rendering ---
    def render(self):
        with self._lock:
            x, y, z = self.x, self.y, max(self.z, 10.0)
        w, h = SIM_IMG_W, SIM_IMG_H
        s = z * math.tan(math.radians(cfg.CAM_HFOV_DEG) / 2) / (w / 2)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        wx = x - (v - h / 2) * s * cfg.CAM_FORWARD_SIGN
        wy = y - (u - w / 2) * s * cfg.CAM_LEFT_SIGN
        img = np.full((h, w), 150, np.int16) + self._noise
        for px, py, rx, ry in SIM_PUDDLES:
            inside = ((wx - px) / rx) ** 2 + ((wy - py) / ry) ** 2 <= 1
            img[inside] = 70 + self._noise[inside]
        img = np.clip(img, 0, 255).astype(np.uint8)
        return np.dstack([img, img, img])  # RGB like djitellopy
