"""
Puddle detection on the Tello's downward (black/white) camera.

Idea: the floor is the dominant surface in the image, so the median grey value
is a good estimate of "dry floor". Wet spots are clearly darker than that
(or, with lights reflecting in them, clearly brighter). We threshold against
the median, clean the mask up and keep blobs that are big and compact enough.

Run this file directly to tune the parameters on saved images or a video:
    python puddle_detector.py capture_123.jpg
    python puddle_detector.py flight.mp4
"""

import math
import threading
import time

import cv2
import numpy as np

import config as cfg


class Detection:
    def __init__(self, cx, cy, area_px, bbox, contour):
        self.cx, self.cy = cx, cy
        self.area_px = area_px
        self.bbox = bbox
        self.contour = contour


class PuddleDetector:
    def __init__(self):
        self.crop = cfg.DOWNCAM_CROP
        self._crop_samples = []

    # -- image preparation --------------------------------------------------
    def to_gray(self, frame):
        if frame.ndim == 2:
            return frame
        code = cv2.COLOR_RGB2GRAY if cfg.FRAME_IS_RGB else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(frame, code)

    def _update_auto_crop(self, gray):
        """The bottom camera image may sit inside black borders; find its box once."""
        if self.crop is not None or not cfg.AUTO_CROP:
            return
        self._crop_samples.append(gray)
        if len(self._crop_samples) < 10:
            return
        mx = np.max(np.stack(self._crop_samples), axis=0)
        self._crop_samples = []
        rows = np.where(mx.max(axis=1) > 10)[0]
        cols = np.where(mx.max(axis=0) > 10)[0]
        if len(rows) == 0 or len(cols) == 0:
            return
        x, y = int(cols[0]), int(rows[0])
        w, h = int(cols[-1] - x + 1), int(rows[-1] - y + 1)
        H, W = gray.shape
        if w * h < 0.9 * W * H:
            print(f"🔍 Auto crop of downward image: x={x} y={y} w={w} h={h}")
            self.crop = (x, y, w, h)
        else:
            self.crop = (0, 0, W, H)

    def prepare(self, frame):
        """Return the cropped grey image the detector works on."""
        gray = self.to_gray(frame)
        self._update_auto_crop(gray)
        if self.crop:
            x, y, w, h = self.crop
            gray = gray[y:y + h, x:x + w]
        return gray

    # -- detection ----------------------------------------------------------
    def detect(self, frame):
        """Return (detections, gray, mask). Pixel coordinates are in the cropped image."""
        gray = self.prepare(frame)
        k = cfg.BLUR_KERNEL | 1
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        floor = float(np.median(blur))

        mask = np.zeros_like(blur)
        if cfg.PUDDLE_MODE in ("dark", "both"):
            mask |= (blur < floor - cfg.DARK_OFFSET).astype(np.uint8) * 255
        if cfg.PUDDLE_MODE in ("bright", "both"):
            mask |= (blur > floor + cfg.BRIGHT_OFFSET).astype(np.uint8) * 255

        m = cfg.BORDER_MARGIN_PX
        if m > 0:
            mask[:m, :] = 0
            mask[-m:, :] = 0
            mask[:, :m] = 0
            mask[:, -m:] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape
        max_area = cfg.MAX_AREA_FRAC * H * W
        edge = m + 2
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.MIN_AREA_PX or area > max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if cfg.REJECT_BORDER_BLOBS and (bx <= edge or by <= edge or
                                            bx + bw >= W - edge or by + bh >= H - edge):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area <= 0 or area / hull_area < cfg.MIN_SOLIDITY:
                continue
            mom = cv2.moments(c)
            cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
            detections.append(Detection(cx, cy, area, (bx, by, bw, bh), c))
        return detections, gray, mask


def cm_per_pixel(image_width, height_cm):
    """Ground size of one pixel for a downward camera at the given height."""
    half_width_cm = height_cm * math.tan(math.radians(cfg.CAM_HFOV_DEG) / 2)
    return half_width_cm / (image_width / 2)


def pixel_to_world(cx, cy, img_w, img_h, height_cm, pose):
    """
    Convert an image position to mission-frame coordinates (cm).

    pose: (x, y, z, yaw) of the drone when the frame was taken.
    Returns (x, y, cm_per_px).
    """
    s = cm_per_pixel(img_w, height_cm)
    fwd = -(cy - img_h / 2) * s * cfg.CAM_FORWARD_SIGN + cfg.CAM_OFFSET_CM[0]
    left = -(cx - img_w / 2) * s * cfg.CAM_LEFT_SIGN + cfg.CAM_OFFSET_CM[1]
    x, y, _, yaw = pose
    yaw = math.radians(yaw)
    wx = x + fwd * math.cos(yaw) - left * math.sin(yaw)
    wy = y + fwd * math.sin(yaw) + left * math.cos(yaw)
    return wx, wy, s


class PuddleTracker:
    """Merges repeated detections of the same puddle and decides when to report."""

    def __init__(self):
        self._lock = threading.Lock()
        self.puddles = []
        self._next_id = 1

    def add(self, x, y, area_cm2):
        """Add one detection. Returns the puddle dict when it should be reported now."""
        with self._lock:
            best, best_d = None, cfg.MERGE_RADIUS_CM
            for p in self.puddles:
                d = math.hypot(p["x"] - x, p["y"] - y)
                if d < best_d:
                    best, best_d = p, d
            now = time.time()
            if best is None:
                best = {"id": self._next_id, "x": x, "y": y, "area_cm2": area_cm2,
                        "hits": 0, "reported": False, "first_seen": now}
                self._next_id += 1
                self.puddles.append(best)
            # running average of position and size
            n = best["hits"]
            best["x"] = (best["x"] * n + x) / (n + 1)
            best["y"] = (best["y"] * n + y) / (n + 1)
            best["area_cm2"] = (best["area_cm2"] * n + area_cm2) / (n + 1)
            best["hits"] = n + 1
            best["last_seen"] = now
            if not best["reported"] and best["hits"] >= cfg.MIN_HITS:
                best["reported"] = True
                return self.export(best)
            return None

    @staticmethod
    def export(p):
        return {"id": p["id"], "x": round(p["x"], 1), "y": round(p["y"], 1),
                "area_cm2": round(p["area_cm2"], 1), "hits": p["hits"]}

    def confirmed(self):
        with self._lock:
            return [self.export(p) for p in self.puddles if p["reported"]]

    def reset(self):
        with self._lock:
            self.puddles = []
            self._next_id = 1


def draw_detections(gray, detections):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in detections:
        cv2.drawContours(vis, [d.contour], -1, (0, 0, 255), 2)
        cv2.circle(vis, (int(d.cx), int(d.cy)), 4, (0, 255, 255), -1)
    return vis


if __name__ == "__main__":
    # Tuning tool: show detection result for images or a video file.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python puddle_detector.py <image|video> [...]")
        sys.exit(1)
    cfg.FRAME_IS_RGB = False  # cv2 loads files as BGR
    for path in sys.argv[1:]:
        cap = cv2.VideoCapture(path)
        det = PuddleDetector()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets, gray, mask = det.detect(frame)
            vis = draw_detections(gray, dets)
            cv2.imshow("detections", vis)
            cv2.imshow("mask", mask)
            print(f"{path}: {len(dets)} puddle(s), floor median={np.median(gray):.0f}")
            if cv2.waitKey(0 if cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 1 else 30) & 0xFF == 27:
                break
    cv2.destroyAllWindows()
