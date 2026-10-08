"""
Helipad ("H") detection on the downward camera, for landing on a landing pad.

A printed H (dark on light paper, or light on a dark pad) is found with plain
image processing, no training needed:
1. threshold the image both ways (Otsu), so both colour versions work;
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
from .puddle_detector import DownCamDetector


N = 48   # size of the straightened blob


def _h_score(m):
    """How well a straightened N x N 0/1 mask looks like an H, 0 = perfect, None = no H."""
    e, g0, g1 = int(0.15 * N), int(0.4 * N), int(0.6 * N)
    t, c0, c1 = int(0.3 * N), int(0.42 * N), int(0.58 * N)
    best = None
    for mm in (m, m.T):   # bars vertical or horizontal in the rectangle
        bars = min(mm[:, :e].mean(), mm[:, -e:].mean())
        cross = mm[c0:c1, g0:g1].mean()
        gaps = max(mm[:t, g0:g1].mean(), mm[-t:, g0:g1].mean())
        if bars < 0.8 or cross < 0.75 or gaps > 0.15:
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


class HelipadDetector(DownCamDetector):
    def detect(self, frame):
        """Return the best Helipad in the frame or None, plus the grey image."""
        gray = self.prepare(frame)
        return self.find(gray), gray

    def find(self, gray):
        """Best Helipad in an already prepared (grey, cropped) image, or None."""
        H, W = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        best = None
        for mask in (dark, 255 - dark):   # dark H on light paper / light H on dark pad
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cand = self._check(c, W, H)
                if cand and (best is None or cand.score < best.score):
                    best = cand
        return best

    @staticmethod
    def _check(c, W, H):
        area = cv2.contourArea(c)
        if area < cfg.HELIPAD_MIN_AREA_PX or area > 0.5 * W * H:
            return None
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            return None   # cut off by the image edge: centre would be wrong
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 1.8:
            return None
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or not 0.4 <= area / hull <= 0.85:
            return None   # an H fills its hull only partly (two gaps)
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        warp = cv2.getAffineTransform(box[:3], np.float32([[0, N], [0, 0], [N, 0]]))
        blob = np.zeros((H, W), np.uint8)
        cv2.drawContours(blob, [c], -1, 1, -1)
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
    det = HelipadDetector()
    print("ESC = stop, any other key = next image")
    for path in files:
        cap = cv2.VideoCapture(path)
        still = cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pad, gray = det.detect(frame)
            print(f"{path}: " + (f"H op ({pad.cx:.0f}, {pad.cy:.0f}), score {pad.score:.3f}"
                                 if pad else "geen H"))
            cv2.imshow("helipad", draw_helipad(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), pad))
            if cv2.waitKey(0 if still else 30) & 0xFF == 27:
                cv2.destroyAllWindows()
                raise SystemExit
    cv2.destroyAllWindows()
