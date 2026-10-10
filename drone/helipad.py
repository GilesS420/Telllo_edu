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
        best = None
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cand = self._check(c, W, H, size_px)
                if cand and (best is None or cand.score < best.score):
                    best = cand
        return best

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


def draw_helipad(vis, pad):
    """Draw a found H on a BGR image (in place)."""
    if pad is None:
        return vis
    cv2.drawContours(vis, [pad.contour], -1, (0, 220, 0), 2)
    cx, cy = int(pad.cx), int(pad.cy)
    cv2.drawMarker(vis, (cx, cy), (0, 220, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(vis, "H", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
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
