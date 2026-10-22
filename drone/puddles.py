"""
Puddle detection with a trained object detection model (YOLO, Ultralytics).

The model is trained on downward camera frames (Roboflow dataset, see README
"Plassen herkennen"). It runs in its own thread on the newest frame, so a slow
model never holds up the H detection or the camera view. Every box is turned
into mission coordinates with the height and pose of the frame; boxes at the
same place are merged into one puddle. A puddle counts once it was seen
PUDDLE_MIN_HITS times: then on_new(puddle) is called (the app sends it to the
Jetson).

Without ultralytics or without a model file the detector is simply off.

Test the model on saved images, a folder or a video (opens a window):
    python -m drone.puddles capture_123.png
    python -m drone.puddles dataset/test/images
    python -m drone.puddles flight.mp4
"""

import collections
import math
import os
import threading
import time

import cv2

from . import config as cfg
from .downcam import cm_per_pixel, pixel_to_world


class Box:
    """One detection in one frame (pixels of the cropped downward image)."""

    def __init__(self, x1, y1, x2, y2, conf):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.conf = conf
        self.world = None     # (x, y) in cm, mission frame, or None (ground, edge)
        self.puddle_id = None

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


class Puddle:
    """A puddle on the floor, merged from all its detections."""

    def __init__(self, pid, x, y, w_cm, h_cm, conf):
        self.id = pid
        self.x, self.y = x, y
        self.w_cm, self.h_cm = w_cm, h_cm
        self.conf = conf
        self.hits = 1
        self.last_seen = time.time()
        self.photo = None          # file of the photo taken straight above it
        self.photo_offset_cm = None  # drone-puddle distance when the photo was taken
        self.photo_tried = False

    @property
    def confirmed(self):
        return self.hits >= cfg.PUDDLE_MIN_HITS

    def to_msg(self):
        return {"id": self.id, "x": round(self.x, 1), "y": round(self.y, 1),
                "size_cm": [round(self.w_cm), round(self.h_cm)],
                "area_cm2": round(self.w_cm * self.h_cm * math.pi / 4),
                "confidence": round(self.conf, 2), "hits": self.hits,
                "photo": self.photo}


def load_model(path):
    """Load a YOLO model, or return (None, reason)."""
    if not path or not os.path.exists(path):
        return None, f"geen model gevonden ({path})"
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics is niet geïnstalleerd (pip install ultralytics)"
    try:
        return YOLO(path, task="detect"), None
    except Exception as e:
        return None, f"model laden mislukt: {e}"


def run_model(model, gray):
    """Run the model on a grey (or BGR) image; return a list of Box."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    kw = {"imgsz": cfg.PUDDLE_IMGSZ, "conf": cfg.PUDDLE_CONF, "verbose": False}
    if cfg.PUDDLE_DEVICE is not None:
        kw["device"] = cfg.PUDDLE_DEVICE
    result = model.predict(img, **kw)[0]
    boxes = []
    for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        boxes.append(Box(x1, y1, x2, y2, conf))
    return boxes


class PuddleDetector:
    def __init__(self, path=None, on_new=None, model=None):
        """
        Args:
            path: model file (default PUDDLE_MODEL)
            on_new: callback(Puddle) when a puddle is confirmed (detector thread)
            model: an already loaded model (tests); skips loading
        """
        self.path = path or cfg.PUDDLE_MODEL
        self.on_new = on_new
        self.model = model
        self.status = "model laden…" if model is None else "actief"
        self.enabled = model is not None or os.path.exists(self.path)
        if not self.enabled:
            self.status = f"uit: geen model ({self.path})"
        self._lock = threading.Lock()
        self._job = None
        self._wake = threading.Event()
        self._boxes = ([], 0.0, None)   # (boxes, frame time, image shape)
        self._puddles = []
        self._next_id = 1
        self._sightings = collections.deque(maxlen=100)   # (t_frame, x, y, conf)
        self.infer_ms = None
        self._running = True
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    # -- called from the camera thread ----------------------------------------
    def submit(self, gray, t_frame, height, pose, airborne):
        """Hand over the newest frame. Older, not yet processed frames are dropped."""
        if not self.enabled:
            return
        with self._lock:
            self._job = (gray, t_frame, height, pose, airborne)
        self._wake.set()

    def latest_boxes(self, shape):
        """Boxes of the last processed frame, if recent and of the same image size."""
        with self._lock:
            boxes, t, shp = self._boxes
        # older boxes no longer sit on the puddle in a moving image
        max_age = 2.0 / cfg.PUDDLE_HZ + 0.2
        if shp != shape or time.time() - t > max_age:
            return []
        return boxes

    def sighting_near(self, x, y, since, radius=None):
        """Newest detection (t_frame, x, y, conf) after `since` near (x, y), or None."""
        radius = radius or cfg.PUDDLE_MERGE_CM
        with self._lock:
            for s in reversed(self._sightings):
                if s[0] <= since:
                    break
                if math.hypot(s[1] - x, s[2] - y) <= radius:
                    return s
        return None

    def puddles(self, confirmed=True):
        with self._lock:
            return [p for p in self._puddles if p.confirmed or not confirmed]

    def reset(self):
        """New mission frame: the old puddle positions no longer fit."""
        with self._lock:
            self._puddles = []
            self._next_id = 1
            self._sightings.clear()

    def stop(self):
        self._running = False
        self._wake.set()

    # -- detector thread -----------------------------------------------------
    def _loop(self):
        if self.model is None:
            self.model, err = load_model(self.path)
            if self.model is None:
                self.enabled = False
                self.status = f"uit: {err}"
                print(f"⚠️  Plasdetectie {self.status}")
                return
            self.status = "actief"
            print(f"💧 Plasmodel geladen: {self.path}")
        period = 1.0 / cfg.PUDDLE_HZ
        while self._running:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            t0 = time.time()
            try:
                self.process(*job)
            except Exception as e:   # keep the thread alive
                if self.status != f"fout: {e}":
                    self.status = f"fout: {e}"
                    print(f"⚠️  Plasdetectie: {e!r}")
            else:
                self.status = "actief"
            self.infer_ms = (time.time() - t0) * 1000
            time.sleep(max(0.0, period - (time.time() - t0)))

    def process(self, gray, t_frame, height, pose, airborne):
        """Detect in one frame, add the positions to the puddle list."""
        boxes = run_model(self.model, gray)
        h, w = gray.shape[:2]
        new = []
        with self._lock:
            for b in boxes:
                if not airborne or not height or height < 10:
                    continue   # on the ground or no height: only shown in the camera view
                e = cfg.PUDDLE_EDGE_PX
                if b.x1 <= e or b.y1 <= e or b.x2 >= w - e or b.y2 >= h - e:
                    continue   # partly out of view: its middle is not the puddle's middle
                x, y, _ = pixel_to_world(b.cx, b.cy, w, h, height, pose)
                s = cm_per_pixel(w, height)
                b.world = (x, y)
                self._sightings.append((t_frame, x, y, b.conf))
                p = self._merge(x, y, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s, b.conf)
                b.puddle_id = p.id
                if p.hits == cfg.PUDDLE_MIN_HITS:
                    new.append(p)
            self._boxes = (boxes, t_frame, (h, w))
        for p in new:
            print(f"💧 Plas {p.id} op ({p.x:.0f}, {p.y:.0f}) cm, "
                  f"~{p.w_cm:.0f}x{p.h_cm:.0f} cm")
            if self.on_new:
                self.on_new(p)

    def _merge(self, x, y, w_cm, h_cm, conf):
        best, best_d = None, cfg.PUDDLE_MERGE_CM
        for p in self._puddles:
            d = math.hypot(p.x - x, p.y - y)
            if d < best_d:
                best, best_d = p, d
        if best is None:
            best = Puddle(self._next_id, x, y, w_cm, h_cm, conf)
            self._next_id += 1
            self._puddles.append(best)
            return best
        k = 1.0 / min(best.hits + 1, 10)   # running mean, keeps adapting a little
        best.x += (x - best.x) * k
        best.y += (y - best.y) * k
        best.w_cm += (w_cm - best.w_cm) * k
        best.h_cm += (h_cm - best.h_cm) * k
        best.conf = max(best.conf, conf)
        best.hits += 1
        best.last_seen = time.time()
        return best


def draw_puddles(vis, boxes):
    """Draw puddle boxes on a BGR image (in place)."""
    for b in boxes:
        color = (248, 140, 129)   # C["water"] of the GUI (BGR)
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(vis, p1, p2, color, 2)
        text = f"plas {b.conf:.2f}"
        if b.world is not None:
            text += f" ({b.world[0]:.0f},{b.world[1]:.0f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = p1[1] - 4 if p1[1] > th + 6 else p2[1] + th + 4
        cv2.rectangle(vis, (p1[0], ty - th - 3), (p1[0] + tw + 4, ty + 3), color, -1)
        cv2.putText(vis, text, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Test the puddle model on images or a video")
    ap.add_argument("paths", nargs="+", help="image(s), folder(s) or video(s)")
    ap.add_argument("--model", default=cfg.PUDDLE_MODEL)
    args = ap.parse_args()
    model, err = load_model(args.model)
    if model is None:
        raise SystemExit(err)
    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*"))
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        else:
            files.append(p)

    def show(img):
        boxes = run_model(model, img)
        cv2.imshow("plassen (spatie = volgende, q = stoppen)", draw_puddles(img.copy(), boxes))
        return boxes

    for f in files:
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            cap = cv2.VideoCapture(f)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                show(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise SystemExit
            continue
        img = cv2.imread(f)
        if img is None:
            print(f"⚠️  {f}: geen afbeelding")
            continue
        boxes = show(img)
        print(f"{f}: {len(boxes)} plas(sen)" + "".join(f"  {b.conf:.2f}" for b in boxes))
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
