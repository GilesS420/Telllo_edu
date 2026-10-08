"""
FakeTello: a simulator with the same methods we use from djitellopy.

    python tello_gui.py --sim

What it simulates:
* 'go', rotate and rc commands, takeoff/landing
* drift: slowly changing "wind" pushes the drone off its path and turns it a
  little, also while it hovers between moves (like a real Tello)
* a downward camera (30 fps, with video delay) looking at a textured floor with
  puddles at known positions, so visual odometry and puddle detection can be
  tested end-to-end
* IMU yaw and the height sensor
"""

import collections
import math
import threading
import time

import cv2
import numpy as np

from . import config as cfg

# Puddles in the simulated world: (x, y, radius_x, radius_y) in cm, mission frame
SIM_PUDDLES = [(150, 0, 25, 15), (200, 120, 20, 20), (50, 150, 20, 12)]
SIM_IMG_W, SIM_IMG_H = 320, 240
TEX_RES = 0.5               # cm per texture pixel
TEX_HALF = 600              # texture covers -600..600 cm
RC_SCALE = 0.85             # real cm/s per rc unit (deliberately not exactly RC_CMS_PER_UNIT)
SETTLE_S = 0.8              # the drone hovers this long after a 'go' before answering ok
LATENCY_S = 0.15            # video delay


class _FrameRead:
    def __init__(self, tello):
        self._tello = tello

    @property
    def frame(self):
        return self._tello.latest_frame()


class FakeTello:
    RESPONSE_TIMEOUT = 7

    def __init__(self, drift=True, wind_cms=5.0, yaw_drift_dps=1.0, seed=0, flip_camera=False):
        self._lock = threading.Lock()
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.cam_sign = -1 if flip_camera else 1  # test: camera mounted the other way round
        self.wind_std, self.yaw_std = wind_cms, yaw_drift_dps
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0                      # true heading, mission convention (CCW +)
        self.is_flying = False
        self.battery = 95
        self._wind = np.zeros(2)
        self._yaw_wind = 0.0
        self._go = None                     # remaining world displacement + speed
        self._vz_target = None
        self._rc = (0, 0, 0, 0)
        self._rc_time = 0.0
        self._v = np.zeros(3)
        self._frames = collections.deque(maxlen=20)
        self._texture = self._make_texture()
        self._running = True
        threading.Thread(target=self._physics, daemon=True).start()
        threading.Thread(target=self._camera, daemon=True).start()

    # --- world -----------------------------------------------------------------
    def _make_texture(self):
        n = int(2 * TEX_HALF / TEX_RES)
        rng = np.random.default_rng(42)
        tex = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 3)
        tex = 150 + tex / tex.std() * 10
        coords = (np.arange(n) * TEX_RES - TEX_HALF).astype(np.float32)
        for px, py, rx, ry in SIM_PUDDLES:
            # rows = world x, cols = world y
            r0, r1 = [int((v + TEX_HALF) / TEX_RES) for v in (px - rx, px + rx)]
            c0, c1 = [int((v + TEX_HALF) / TEX_RES) for v in (py - ry, py + ry)]
            xx, yy = np.meshgrid(coords[r0:r1], coords[c0:c1], indexing="ij")
            inside = ((xx - px) / rx) ** 2 + ((yy - py) / ry) ** 2 <= 1
            tex[r0:r1, c0:c1][inside] -= 80
        return tex

    def truth(self):
        with self._lock:
            return self.x, self.y, self.z, self.yaw

    # --- physics ---------------------------------------------------------------
    def _physics(self):
        dt = 0.02
        while self._running:
            time.sleep(dt)
            with self._lock:
                if not self.is_flying:
                    self._v[:] = 0
                    continue
                if self.drift:  # slowly varying wind (random walk with decay)
                    tau = 4.0
                    self._wind += (-self._wind / tau * dt
                                   + self.rng.normal(0, 1, 2) * self.wind_std * math.sqrt(2 * dt / tau))
                    self._yaw_wind += (-self._yaw_wind / 3 * dt
                                       + self.rng.normal() * self.yaw_std * math.sqrt(2 * dt / 3))
                # commanded velocity (world frame)
                cmd = np.zeros(3)
                yaw_rate = 0.0
                if self._go is not None:
                    rem, speed = self._go
                    dist = np.linalg.norm(rem)
                    if dist < 0.5:
                        self._go = None
                    else:
                        cmd = rem / dist * min(speed, dist / dt)
                        rem -= cmd * dt
                elif time.time() - self._rc_time < 1.0:
                    lr, fb, ud, yv = self._rc
                    c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
                    fwd, left = fb * RC_SCALE, -lr * RC_SCALE
                    cmd = np.array([fwd * c - left * s, fwd * s + left * c, ud * RC_SCALE])
                    yaw_rate = -yv * 1.0  # rc yaw is clockwise positive
                if self._vz_target is not None:
                    cmd[2] = self._vz_target
                self._v += (cmd - self._v) * min(1.0, dt / 0.25)  # inertia
                self.x += (self._v[0] + self._wind[0]) * dt
                self.y += (self._v[1] + self._wind[1]) * dt
                self.z = max(0.0, self.z + self._v[2] * dt)
                self.yaw = (self.yaw + (yaw_rate + self._yaw_wind) * dt + 180) % 360 - 180

    def _wait(self, cond, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if cond():
                    return
            time.sleep(0.02)

    # --- camera ----------------------------------------------------------------
    def _camera(self):
        w, h = SIM_IMG_W, SIM_IMG_H
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        while self._running:
            t0 = time.time()
            x, y, z, yaw = self.truth()
            s = max(z, 10.0) * math.tan(math.radians(cfg.CAM_HFOV_DEG) / 2) / (w / 2)
            fwd = -(v - h / 2) * s * cfg.CAM_FORWARD_SIGN * self.cam_sign
            left = -(u - w / 2) * s * cfg.CAM_LEFT_SIGN
            c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            wx = x + fwd * c - left * sn
            wy = y + fwd * sn + left * c
            map_r = ((wx + TEX_HALF) / TEX_RES).astype(np.float32)   # texture row = world x
            map_c = ((wy + TEX_HALF) / TEX_RES).astype(np.float32)
            img = cv2.remap(self._texture, map_c, map_r, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
            img += self.rng.normal(0, 3, img.shape).astype(np.float32)
            img = np.clip(img, 0, 255).astype(np.uint8)
            self._frames.append((t0, np.dstack([img, img, img])))  # RGB like djitellopy
            time.sleep(max(0.0, 1 / 30 - (time.time() - t0)))

    def latest_frame(self):
        cutoff = time.time() - LATENCY_S
        for t, f in reversed(self._frames):
            if t <= cutoff:
                return f
        return self._frames[0][1] if self._frames else None

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
        self._running = False

    def send_command_with_return(self, cmd, *a, **kw):
        print(f"[SIM] {cmd}")
        return "ok"

    def get_frame_read(self):
        return _FrameRead(self)

    # --- flight ---
    def takeoff(self):
        with self._lock:
            self.is_flying = True
            self._vz_target = 60.0
        self._wait(lambda: self.z >= 80)
        with self._lock:
            self._vz_target = None
        time.sleep(SETTLE_S)

    def land(self):
        with self._lock:
            self._go = None
            self._vz_target = -50.0
        self._wait(lambda: self.z <= 0.5)
        with self._lock:
            self._vz_target = None
            self.is_flying = False
            self._wind[:] = 0

    def emergency(self):
        with self._lock:
            self.z, self.is_flying = 0.0, False

    def send_rc_control(self, lr, fb, ud, yv):
        with self._lock:
            self._rc = (lr, fb, ud, yv)
            self._rc_time = time.time()

    def go_xyz_speed(self, x, y, z, speed):
        """Body frame move relative to the current (true) heading."""
        if max(abs(x), abs(y), abs(z)) < 20:
            raise Exception("Command 'go' was unsuccessful: out of range")
        with self._lock:
            c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
            rem = np.array([x * c - y * s, x * s + y * c, float(z)])
            self._go = (rem, float(speed))
        self._wait(lambda: self._go is None)
        time.sleep(SETTLE_S)
        self.battery = max(self.battery - 1, 0)

    def _rotate(self, deg):
        steps = max(1, int(abs(deg) / 90 / 0.02))
        for _ in range(steps):
            with self._lock:
                self.yaw = (self.yaw + deg / steps + 180) % 360 - 180
            time.sleep(0.02)
        time.sleep(0.3)

    def rotate_clockwise(self, deg):
        self._rotate(-deg)

    def rotate_counter_clockwise(self, deg):
        self._rotate(deg)

    def get_yaw(self):
        with self._lock:
            return int(round(-self.yaw))  # Tello convention: clockwise positive

    def get_distance_tof(self):
        with self._lock:
            return int(round(self.z + self.rng.normal(0, 1)))

    def get_height(self):
        return self.get_distance_tof()
