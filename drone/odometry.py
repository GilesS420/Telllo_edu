"""
Odometry: measure where the drone really is, instead of only adding up commands.

* Heading: the IMU yaw of the Tello (state packets, no command needed).
* Height: the time-of-flight sensor under the drone.
* Horizontal movement: visual odometry on the downward camera. The shift of the
  floor texture between two frames (phase correlation) times the ground size of
  a pixel gives the real displacement, also while the drone "hovers" and slowly
  drifts or gets pushed by wind.

When the floor has too little texture (or the image is blurred) the odometry is
"lost" and the pose falls back to the commanded moves, like before.
"""

import math
import threading
import time

import cv2
import numpy as np

from . import config as cfg
from .puddle_detector import cam_to_body, cm_per_pixel


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class Odometry:
    def __init__(self, tello, pose, frame_reader, detector, is_airborne):
        self.tello = tello
        self.pose = pose
        self.frame_reader = frame_reader
        self.detector = detector      # only used for its grey conversion + crop
        self.is_airborne = is_airborne
        self.running = True
        self._window = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _gray(self, frame):
        gray = self.detector.to_gray(frame)
        crop = self.detector.crop
        if crop:
            x, y, w, h = crop
            gray = gray[y:y + h, x:x + w]
        if gray.shape[1] > 320:  # keep it fast; the scale follows the image width
            f = 320 / gray.shape[1]
            gray = cv2.resize(gray, (320, int(round(gray.shape[0] * f))),
                              interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32)

    def _compare(self, ref, g):
        if self._window is None or self._window.shape != g.shape:
            self._window = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if ref["img"].shape != g.shape:
            return (0.0, 0.0), 0.0
        return cv2.phaseCorrelate(ref["img"], g, self._window)

    @staticmethod
    def _position(ref, shift, width, height):
        """Odometry position for a frame shifted `shift` pixels from reference `ref`."""
        sx, sy = shift
        # floor moves down in the image = drone moves forward, etc.
        s = cm_per_pixel(width, (height + ref["height"]) / 2)
        fwd, left = cam_to_body(sy * s * cfg.CAM_FORWARD_SIGN, sx * s * cfg.CAM_LEFT_SIGN)
        ky = math.radians(ref["yaw"])
        return (ref["world"][0] + fwd * math.cos(ky) - left * math.sin(ky),
                ref["world"][1] + fwd * math.sin(ky) + left * math.cos(ky))

    def _height(self):
        try:
            h = self.tello.get_distance_tof()
        except Exception:
            return None
        return float(h) if 10 < h < 1000 else None

    def _loop(self):
        """
        Keyframe visual odometry: every frame is compared with a reference frame
        (keyframe), not with the previous frame, so the small sub-pixel error of
        each comparison does not add up while hovering or flying slowly. A new
        keyframe is taken when the view has shifted far enough, the drone turned
        or the height changed.
        """
        period = 1.0 / cfg.VO_HZ
        key = prev = None           # dict(img, world, yaw, height)
        prev_frame = None
        world = (0.0, 0.0)          # odometry position (only differences are used)
        last_t = time.time()
        last_good = 0.0
        while self.running:
            t0 = time.time()
            dt, last_t = t0 - last_t, t0
            try:
                self.pose.update_imu(self.tello.get_yaw())
            except Exception:
                pass
            yaw = self.pose.yaw

            vo = None
            height = None
            quality = 0.0
            airborne = self.is_airborne()
            if airborne:
                height = self._height()
                frame = self.frame_reader.frame
                usable = (cfg.VO_ENABLED and not self.pose.vo_disabled and height is not None
                          and height >= cfg.VO_MIN_HEIGHT and frame is not None)
                if usable and frame is not prev_frame:  # only new frames carry information
                    prev_frame = frame
                    g = self._gray(frame)
                    current = {"img": g, "yaw": yaw, "height": height}
                    if (key is None or key["img"].shape != g.shape
                            or abs(wrap_deg(yaw - key["yaw"])) > cfg.VO_MAX_YAW_STEP
                            or abs(height / key["height"] - 1) > 0.12):
                        # turned or climbed: the old keyframe can't be compared anymore.
                        # While climbing/descending (takeoff, landing, over a box) the
                        # previous frame is still close enough: measure that last step
                        # first, so the drift during a height change is not lost.
                        if (prev is not None and prev["img"].shape == g.shape
                                and abs(wrap_deg(yaw - prev["yaw"])) <= cfg.VO_MAX_YAW_STEP
                                and abs(height / prev["height"] - 1) <= 0.12):
                            shift, quality = self._compare(prev, g)
                            if quality >= cfg.VO_MIN_RESPONSE:
                                new = self._position(prev, shift, g.shape[1], height)
                                vo = (new[0] - world[0], new[1] - world[1])
                                world = new
                                last_good = t0
                        key = prev = dict(current, world=world)
                    else:
                        # 1) against the keyframe, 2) if that fails against the previous frame
                        for ref in (key, prev):
                            shift, quality = self._compare(ref, g)
                            if quality >= cfg.VO_MIN_RESPONSE:
                                break
                        if quality >= cfg.VO_MIN_RESPONSE:
                            new = self._position(ref, shift, g.shape[1], height)
                            vo = (new[0] - world[0], new[1] - world[1])
                            world = new
                            last_good = t0
                            prev = dict(current, world=world)
                            if (ref is not key or quality < cfg.VO_KEY_QUALITY
                                    or math.hypot(*shift) > cfg.VO_KEY_SHIFT_PX):
                                key = prev
                        elif t0 - last_good > cfg.VO_LOST_S:
                            # lost (no texture/blur): restart from this frame
                            key = prev = dict(current, world=world)
                elif not usable:
                    key = None
            else:
                key = None

            vo_ok = (cfg.VO_ENABLED and not self.pose.vo_disabled and airborne
                     and t0 - last_good < cfg.VO_LOST_S)
            self.pose.step(dt, t0, vo, height if airborne else None, measured=vo_ok,
                           quality=quality)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def stop(self):
        self.running = False
