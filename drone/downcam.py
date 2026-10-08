"""
Downward camera helpers: grey conversion, automatic crop of the black borders
and the geometry to turn image pixels into mission coordinates.

Used by the helipad detector and the visual odometry. Object detection (puddles
etc.) is done by a trained model elsewhere and is not part of this code.
"""

import math
import threading

import cv2
import numpy as np

from . import config as cfg


class DownCam:
    """Shared image preparation (grey + crop) for the downward camera."""

    def __init__(self):
        self.crop = cfg.DOWNCAM_CROP
        self._crop_samples = []
        self._crop_shape = None        # image size the auto crop belongs to
        self._crop_lock = threading.Lock()  # used by the detector and odometry threads

    def to_gray(self, frame):
        if frame.ndim == 2:
            return frame
        code = cv2.COLOR_RGB2GRAY if cfg.FRAME_IS_RGB else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(frame, code)

    def _update_auto_crop(self, gray):
        """The bottom camera image may sit inside black borders; find its box once."""
        if cfg.DOWNCAM_CROP is not None or not cfg.AUTO_CROP:
            return
        with self._crop_lock:
            if gray.shape != self._crop_shape:
                # first frame, or the video changed size (switching between the front
                # and bottom camera): find the crop again for this size
                self._crop_shape = gray.shape
                self.crop = None
                self._crop_samples = []
            if self.crop is not None:
                return
            self._crop_samples.append(gray)
            if len(self._crop_samples) < 10:
                return
            mx = np.max(np.stack(self._crop_samples), axis=0)
            self._crop_samples = []
            self._set_crop(mx, gray.shape)

    def _set_crop(self, mx, shape):
        rows = np.where(mx.max(axis=1) > 10)[0]
        cols = np.where(mx.max(axis=0) > 10)[0]
        if len(rows) == 0 or len(cols) == 0:
            return
        x, y = int(cols[0]), int(rows[0])
        w, h = int(cols[-1] - x + 1), int(rows[-1] - y + 1)
        H, W = shape
        if w * h < 0.9 * W * H:
            print(f"🔍 Auto crop of downward image: x={x} y={y} w={w} h={h}")
            self.crop = (x, y, w, h)
        else:
            self.crop = (0, 0, W, H)

    def cropped(self, gray):
        """Apply the current crop, if it fits this image."""
        crop = self.crop
        if crop:
            x, y, w, h = crop
            if y + h <= gray.shape[0] and x + w <= gray.shape[1]:
                gray = gray[y:y + h, x:x + w]
        return gray

    def prepare(self, frame):
        """Return the cropped grey image."""
        gray = self.to_gray(frame)
        self._update_auto_crop(gray)
        return self.cropped(gray)


def cm_per_pixel(image_width, height_cm):
    """Ground size of one pixel for a downward camera at the given height."""
    half_width_cm = height_cm * math.tan(math.radians(cfg.CAM_HFOV_DEG) / 2)
    return half_width_cm / (image_width / 2)


def cam_to_body(fwd, left):
    """Turn an image-based (forward, left) vector by CAM_ROTATE_DEG into the drone frame."""
    a = math.radians(cfg.CAM_ROTATE_DEG)
    if not a:
        return fwd, left
    return fwd * math.cos(a) - left * math.sin(a), fwd * math.sin(a) + left * math.cos(a)


def pixel_to_world(cx, cy, img_w, img_h, height_cm, pose):
    """
    Convert an image position to mission-frame coordinates (cm).

    pose: (x, y, z, yaw) of the drone when the frame was taken.
    Returns (x, y, cm_per_px).
    """
    s = cm_per_pixel(img_w, height_cm)
    fwd, left = cam_to_body(-(cy - img_h / 2) * s * cfg.CAM_FORWARD_SIGN,
                            -(cx - img_w / 2) * s * cfg.CAM_LEFT_SIGN)
    fwd, left = fwd + cfg.CAM_OFFSET_CM[0], left + cfg.CAM_OFFSET_CM[1]
    x, y, _, yaw = pose
    yaw = math.radians(yaw)
    wx = x + fwd * math.cos(yaw) - left * math.sin(yaw)
    wy = y + fwd * math.sin(yaw) + left * math.cos(yaw)
    return wx, wy, s
