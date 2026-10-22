"""
Helipad ("H") detection on the downward camera, for landing on a landing pad.

A printed H (dark on light paper, or light on a dark pad) is found with plain
image processing, no training needed:
1. threshold the image both ways, so both colour versions work: once for the
   whole image (Otsu) and once against the local brightness around every pixel
   (adaptive). From higher up the H is small and the floor around the paper
   decides the global threshold; the local one still separates the H from its
   paper;
2. every blob with roughly square outline that fills its hull only partly
   (the two gaps of an H) is straightened along its smallest enclosing
   rectangle, so the rotation doesn't matter;
3. the straightened blob must have the H structure: two full side bars, a
   filled middle and empty gaps above and below the cross bar.

Run this file to test it on saved images or a video:
    python -m drone.helipad dataset/
"""

import math

import cv2
import numpy as np

from . import config as cfg
from .downcam import DownCam, cm_per_pixel


N = 48   # size of the straightened blob


def _h_score(m):
    """How well a straightened N x N 0/1 mask looks like an H, 0 = perfect, None = no H."""
    e, g0, g1 = int(0.1 * N), int(0.35 * N), int(0.65 * N)
    q, c0, c1 = int(0.25 * N), int(0.3 * N), int(0.7 * N)
    best = None
    for mm in (m, m.T):   # bars vertical or horizontal in the rectangle
        bars = min(mm[:, :e].mean(), mm[:, -e:].mean())   # two full side bars
        rows = mm[:, g0:g1].mean(axis=1)                    # middle columns, per row
        cross = rows[c0:c1].max()                           # a filled cross bar ...
        gaps = max(rows[:q].mean(), rows[-q:].mean())       # ... with empty gaps around it
        if bars < 0.65 or cross < 0.8 or gaps > 0.2:
            continue
        score = 1.0 - (min(bars, cross) - gaps)
        best = score if best is None else min(best, score)
    return best


class Helipad:
    def __init__(self, cx, cy, size_px, score, contour):
        self.cx, self.cy = cx, cy      # centre in pixels (cropped image)
        self.size_px = size_px         # longest side of the H
        self.score = score             # difference with an ideal H, 0 = perfect
        self.contour = contour


class HelipadDetector:
    def find(self, gray, height_cm=None):
        candidates = self.find_candidates(gray, height_cm)
        return candidates[0] if candidates else None

    def find_candidates(self, gray, height_cm=None):
        """
        Best Helipad in an already prepared (grey, cropped) image, or None.
        height_cm (ToF): when given, blobs that can't be an H of a real size
        (HELIPAD_SIZE_RANGE_CM) at that height are skipped.
        """
        H, W = gray.shape
        size_px = None
        if height_cm and height_cm > 10:
            cm_px = cm_per_pixel(W, height_cm)
            size_px = tuple(v / cm_px for v in cfg.HELIPAD_SIZE_RANGE_CM)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks = [dark, 255 - dark]   # dark H on light paper / light H on dark pad
        if cfg.HELIPAD_ADAPTIVE:
            block = cfg.HELIPAD_ADAPTIVE_BLOCK | 1
            for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
                masks.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode,
                                                   block, cfg.HELIPAD_ADAPTIVE_C))
        candidates = []
        for mask in masks:
            # Repair narrow compression/noise breaks in a bar. Keep the original
            # mask too: closing alone can fill the gaps of a small distant H.
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            for variant in (mask, closed):
                contours, _ = cv2.findContours(variant, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    cand = self._check(c, W, H, size_px)
                    if cand:
                        candidates.append(cand)
        distinct = []
        for cand in sorted(candidates, key=lambda pad: pad.score):
            if not any(math.hypot(cand.cx - old.cx, cand.cy - old.cy)
                       < max(5, 0.2 * min(cand.size_px, old.size_px)) for old in distinct):
                distinct.append(cand)
        return distinct

    @staticmethod
    def _check(c, W, H, size_px=None):
        area = cv2.contourArea(c)
        if area < cfg.HELIPAD_MIN_AREA_PX or area > 0.5 * W * H:
            return None
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            return None   # cut off by the image edge: centre would be wrong
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 1.8:
            return None
        if size_px and not size_px[0] <= max(rw, rh) <= size_px[1]:
            return None   # far too small or too big for an H at this height
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or not 0.4 <= area / hull <= 0.85:
            return None   # an H fills its hull only partly (two gaps)
        # straighten the blob (drawn in a small image around it, not the whole frame)
        blob = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(blob, [c - (x - 1, y - 1)], -1, 1, -1)
        box = (cv2.boxPoints(rect) - (x - 1, y - 1)).astype(np.float32)
        warp = cv2.getAffineTransform(box[:3], np.float32([[0, N], [0, 0], [N, 0]]))
        score = _h_score(cv2.warpAffine(blob, warp, (N, N), flags=cv2.INTER_NEAREST))
        if score is None or score > cfg.HELIPAD_MAX_SCORE:
            return None
        m = cv2.moments(c)
        return Helipad(m["m10"] / m["m00"], m["m01"] / m["m00"], max(rw, rh), score, c)


class HelipadTracker:
    """Confirm a consistent H across distinct frames; never refresh a stale hit."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pad = None
        self.last_seen = float("-inf")
        self.hits = 0
        self.confirmed = False
        self.shape = None
        self.velocity = (0.0, 0.0)
        self.last_raw = None

    def update(self, candidates, timestamp, shape):
        if shape != self.shape or timestamp - self.last_seen > cfg.HELIPAD_TRACK_HOLD_S:
            self.reset()
            self.shape = shape
        if timestamp <= self.last_seen:
            return self.pad if self.confirmed else None
        if self.pad is not None:
            dt = min(timestamp - self.last_seen, cfg.HELIPAD_TRACK_HOLD_S)
            predicted = (self.pad.cx + self.velocity[0] * dt,
                         self.pad.cy + self.velocity[1] * dt)
            gate = max(12, shape[1] * 0.04, self.pad.size_px * 0.45)
            gate += min(12, math.hypot(*self.velocity) * dt * 0.5)
            matching = [pad for pad in candidates
                        if math.hypot(pad.cx - predicted[0], pad.cy - predicted[1]) < gate
                        and 0.6 < pad.size_px / self.pad.size_px < 1.7]
            chosen = min(matching, key=lambda pad: pad.score +
                         0.15 * math.hypot(pad.cx - predicted[0], pad.cy - predicted[1]) / gate,
                         default=None)
        else:
            chosen = min(candidates, key=lambda pad: pad.score, default=None)
        if chosen is not None:
            raw = (chosen.cx, chosen.cy)
            if self.last_raw is not None and timestamp > self.last_seen:
                dt = timestamp - self.last_seen
                self.velocity = tuple(0.5 * old + 0.5 * max(-200, min(200, (new - before) / dt))
                                      for old, new, before in zip(self.velocity, raw, self.last_raw))
            self.last_raw = raw
            if self.pad is not None:
                alpha = 0.7  # suppress centre jitter without a long control lag
                chosen = Helipad(self.pad.cx + alpha * (chosen.cx - self.pad.cx),
                                 self.pad.cy + alpha * (chosen.cy - self.pad.cy),
                                 self.pad.size_px + alpha * (chosen.size_px - self.pad.size_px),
                                 chosen.score, chosen.contour)
            self.pad = chosen
            self.last_seen = timestamp
            self.hits += 1
            self.confirmed = self.hits >= cfg.HELIPAD_MIN_SIGHTINGS
        # On a miss, keep the confirmed overlay briefly; the original timestamp
        # lets flight control distinguish it from a fresh measurement.
        return self.pad if self.confirmed else None


def draw_helipad(vis, pad, stale=False):
    """Draw a found H on a BGR image (in place)."""
    if pad is None:
        return vis
    color = (0, 170, 255) if stale else (0, 220, 0)
    cv2.drawContours(vis, [pad.contour], -1, color, 2)
    cx, cy = int(pad.cx), int(pad.cy)
    cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(vis, "H (geheugen)" if stale else "H", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="image(s), folder(s) or video(s)")
    args = ap.parse_args()
    cfg.FRAME_IS_RGB = False  # cv2 loads files as BGR
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            files += sorted(f for f in glob.glob(os.path.join(path, "*"))
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        else:
            files.append(path)
    cam, det = DownCam(), HelipadDetector()
    print("ESC = stop, any other key = next image")
    for path in files:
        cap = cv2.VideoCapture(path)
        still = cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cam.prepare(frame)
            pad = det.find(gray)
            print(f"{path}: " + (f"H op ({pad.cx:.0f}, {pad.cy:.0f}), score {pad.score:.3f}"
                                 if pad else "geen H"))
            cv2.imshow("helipad", draw_helipad(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), pad))
            if cv2.waitKey(0 if still else 30) & 0xFF == 27:
                cv2.destroyAllWindows()
                raise SystemExit
    cv2.destroyAllWindows()
