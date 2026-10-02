"""
Puddle detection on the Tello's downward (black/white) camera.

Two backends, chosen with config.DETECTOR_BACKEND:

* "threshold" (default, no training needed): the floor is the dominant surface
  in the image, so the median grey value is a good estimate of "dry floor".
  Wet spots are clearly darker than that (or, with lights reflecting in them,
  clearly brighter). We threshold against the median, clean the mask up and
  keep blobs that are big and compact enough.
* "yolo" / "roboflow": a trained object detection model (bounding boxes), e.g.
  trained on your Roboflow dataset. See README "Eigen model".

Run this file directly to test the detector on saved images, a folder or a video:
    python puddle_detector.py capture_123.jpg
    python puddle_detector.py dataset/
    python puddle_detector.py flight.mp4
"""

import math
import os
import threading
import time

import cv2
import numpy as np

import config as cfg


class Detection:
    def __init__(self, cx, cy, area_px, bbox, contour=None, confidence=None, label=None):
        self.cx, self.cy = cx, cy
        self.area_px = area_px
        self.bbox = bbox              # (x, y, w, h) in pixels
        self.contour = contour        # only for the threshold backend
        self.confidence = confidence  # only for model backends
        self.label = label


def touches_border(bbox, W, H, edge):
    bx, by, bw, bh = bbox
    return bx <= edge or by <= edge or bx + bw >= W - edge or by + bh >= H - edge


class DownCamDetector:
    """Shared image preparation (grey + crop) for all backends."""

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

    def detect(self, frame):
        """Return (detections, gray, mask_or_None). Pixel coordinates are in the cropped image."""
        raise NotImplementedError


class ThresholdDetector(DownCamDetector):
    """Classic image processing, works without a trained model."""

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
            if cfg.REJECT_BORDER_BLOBS and touches_border((bx, by, bw, bh), W, H, edge):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area <= 0 or area / hull_area < cfg.MIN_SOLIDITY:
                continue
            mom = cv2.moments(c)
            cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
            detections.append(Detection(cx, cy, area, (bx, by, bw, bh), c))
        return detections, gray, mask


# Backwards compatible name
PuddleDetector = ThresholdDetector


class ModelDetector(DownCamDetector):
    """
    Base for trained object detection models. Subclasses implement
    _predict(img_bgr) -> [(x1, y1, x2, y2, confidence, label), ...].
    """

    def detect(self, frame):
        gray = self.prepare(frame)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # models expect 3 channels
        H, W = gray.shape
        detections = []
        for x1, y1, x2, y2, conf, label in self._predict(img):
            if conf < cfg.MODEL_CONFIDENCE:
                continue
            if cfg.MODEL_CLASSES and label not in cfg.MODEL_CLASSES:
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if cfg.REJECT_BORDER_BLOBS and touches_border(bbox, W, H, cfg.MODEL_BORDER_PX):
                continue
            # A box doesn't give the real shape: assume an ellipse inside the box
            area = math.pi / 4 * (x2 - x1) * (y2 - y1)
            if area < cfg.MIN_AREA_PX:
                continue
            detections.append(Detection((x1 + x2) / 2, (y1 + y2) / 2, area, bbox,
                                        confidence=float(conf), label=label))
        return detections, gray, None

    def _predict(self, img_bgr):
        raise NotImplementedError


class YoloDetector(ModelDetector):
    """Ultralytics YOLO model (.pt or .onnx), e.g. trained on a Roboflow export."""

    def __init__(self, model_path):
        super().__init__()
        from ultralytics import YOLO  # pip install ultralytics
        self.model = YOLO(model_path)
        print(f"🧠 YOLO model loaded: {model_path}")

    def _predict(self, img_bgr):
        result = self.model.predict(img_bgr, imgsz=cfg.MODEL_IMGSZ, conf=cfg.MODEL_CONFIDENCE,
                                    verbose=False)[0]
        boxes = result.boxes
        out = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                               boxes.cls.tolist()):
            out.append((x1, y1, x2, y2, conf, result.names[int(cls)]))
        return out


class RoboflowDetector(ModelDetector):
    """
    Model trained on Roboflow, run locally with the 'inference' package
    (downloads the weights once, then works offline, also on the Tello Wi-Fi).
    """

    def __init__(self, model_id, api_key):
        super().__init__()
        from inference import get_model  # pip install inference
        self.model = get_model(model_id=model_id, api_key=api_key)
        print(f"🧠 Roboflow model loaded: {model_id}")

    def _predict(self, img_bgr):
        result = self.model.infer(img_bgr, confidence=cfg.MODEL_CONFIDENCE)[0]
        out = []
        for p in result.predictions:  # x, y = box centre
            out.append((p.x - p.width / 2, p.y - p.height / 2,
                        p.x + p.width / 2, p.y + p.height / 2, p.confidence, p.class_name))
        return out


def create_detector(backend=None):
    backend = backend or cfg.DETECTOR_BACKEND
    if backend == "threshold":
        return ThresholdDetector()
    if backend == "yolo":
        return YoloDetector(cfg.MODEL_PATH)
    if backend == "roboflow":
        api_key = cfg.ROBOFLOW_API_KEY or os.environ.get("ROBOFLOW_API_KEY")
        return RoboflowDetector(cfg.ROBOFLOW_MODEL_ID, api_key)
    raise ValueError(f"unknown DETECTOR_BACKEND '{backend}'")


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

    def add(self, x, y, area_cm2, confidence=None):
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
                        "hits": 0, "reported": False, "first_seen": now, "confidence": None}
                self._next_id += 1
                self.puddles.append(best)
            # running average of position and size
            n = best["hits"]
            best["x"] = (best["x"] * n + x) / (n + 1)
            best["y"] = (best["y"] * n + y) / (n + 1)
            best["area_cm2"] = (best["area_cm2"] * n + area_cm2) / (n + 1)
            best["hits"] = n + 1
            best["last_seen"] = now
            if confidence is not None:
                best["confidence"] = max(best["confidence"] or 0.0, confidence)
            if not best["reported"] and best["hits"] >= cfg.MIN_HITS:
                best["reported"] = True
                return self.export(best)
            return None

    @staticmethod
    def export(p):
        out = {"id": p["id"], "x": round(p["x"], 1), "y": round(p["y"], 1),
               "area_cm2": round(p["area_cm2"], 1), "hits": p["hits"]}
        if p["confidence"] is not None:
            out["confidence"] = round(p["confidence"], 2)
        return out

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
        if d.contour is not None:
            cv2.drawContours(vis, [d.contour], -1, (0, 0, 255), 2)
        else:
            x, y, w, h = d.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, f"{d.label} {d.confidence:.2f}", (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.circle(vis, (int(d.cx), int(d.cy)), 4, (0, 255, 255), -1)
    return vis


if __name__ == "__main__":
    # Test tool: show detection results for images, folders of images or videos.
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="image(s), folder(s) or video(s)")
    ap.add_argument("--backend", choices=["threshold", "yolo", "roboflow"],
                    help="override config.DETECTOR_BACKEND")
    args = ap.parse_args()

    cfg.FRAME_IS_RGB = False  # cv2 loads files as BGR
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            files += sorted(f for f in glob.glob(os.path.join(path, "*"))
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        else:
            files.append(path)
    det = create_detector(args.backend)
    print("ESC = stop, any other key = next image")
    for path in files:
        cap = cv2.VideoCapture(path)
        still = cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets, gray, mask = det.detect(frame)
            cv2.imshow("detections", draw_detections(gray, dets))
            if mask is not None:
                cv2.imshow("mask", mask)
            info = ", ".join(f"{d.confidence:.2f}" if d.confidence is not None
                             else f"{d.area_px:.0f}px" for d in dets)
            print(f"{path}: {len(dets)} puddle(s) {info}")
            if cv2.waitKey(0 if still else 30) & 0xFF == 27:
                cv2.destroyAllWindows()
                raise SystemExit
    cv2.destroyAllWindows()
