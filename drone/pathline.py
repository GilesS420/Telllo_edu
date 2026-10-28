"""
Path recognition: a strip of tape or paper on the floor, seen by the downward
camera (classic image processing, no model needed).

The strip is the largest long, narrow shape that is clearly darker (PATH_DARK)
or lighter than the floor. Shorter shapes (H bars, paper puddles) are ignored. Near the drone the strip is fitted with a
straight line: that gives the point of the strip closest to the drone and its
local direction, enough to follow a gently curving path.
"""

import math

import cv2
import numpy as np

from . import config as cfg
from .downcam import cm_per_pixel, pixel_to_body


class PathLine:
    """The strip in one image (pixels) and, relative to the drone, in cm."""

    def __init__(self, near_px, dir_px, ends_px, width_cm, length_cm, touches_border, mask):
        self.near_px = near_px          # point of the centre line closest to the image centre
        self.dir_px = dir_px            # unit direction of the strip in the image
        self.ends_px = ends_px          # the two extreme points along dir_px
        self.width_cm = width_cm
        self.length_cm = length_cm
        self.touches_border = touches_border   # (end -dir, end +dir) runs out of the image
        self.mask = mask                # pixels of the strip

    def contains(self, cx, cy, margin=3):
        """Is image point (cx, cy) on the strip (e.g. an 'H' found in the tape)?"""
        h, w = self.mask.shape
        x0, x1 = max(0, int(cx) - margin), min(w, int(cx) + margin + 1)
        y0, y1 = max(0, int(cy) - margin), min(h, int(cy) + margin + 1)
        return bool(self.mask[y0:y1, x0:x1].any())


class PathDetector:
    def find(self, gray, height_cm):
        """PathLine or None."""
        if not height_cm or height_cm < 15:
            return None
        h, w = gray.shape
        s = cm_per_pixel(w, height_cm)
        img = cv2.GaussianBlur(gray, (5, 5), 0)
        flag = cv2.THRESH_BINARY_INV if cfg.PATH_DARK else cv2.THRESH_BINARY
        _, mask = cv2.threshold(img, 0, 255, flag + cv2.THRESH_OTSU)
        k = max(3, int(round(cfg.PATH_WIDTH_CM[0] / s / 2)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area * s * s < cfg.PATH_MIN_LENGTH_CM * cfg.PATH_WIDTH_CM[0]:
                continue
            ys, xs = np.nonzero(labels == i)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            length_cm = max(rw, rh) * s
            width_cm = area / max(rw, rh, 1) * s       # mean width, also for a curve
            if length_cm < cfg.PATH_MIN_LENGTH_CM:
                continue
            if not cfg.PATH_WIDTH_CM[0] <= width_cm <= cfg.PATH_WIDTH_CM[1]:
                continue
            if length_cm < 2.5 * width_cm:
                continue
            inside = img[ys, xs].mean()
            ring = cv2.dilate((labels == i).astype(np.uint8), np.ones((15, 15), np.uint8))
            around = img[(ring > 0) & (labels != i)]
            if around.size == 0 or abs(float(around.mean()) - inside) < cfg.PATH_MIN_CONTRAST:
                continue
            if best is None or area > best[0]:
                best = (area, pts, width_cm, length_cm, i)
        if best is None:
            return None
        _, pts, width_cm, length_cm, label = best
        # local fit around the part of the strip closest to the image centre
        c = np.array([w / 2, h / 2], np.float32)
        d = np.linalg.norm(pts - c, axis=1)
        local = pts[d <= d.min() + 0.3 * max(w, h)]
        if len(local) < 10:
            local = pts
        vx, vy, x0, y0 = cv2.fitLine(local, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        u = np.array([vx, vy], np.float32)
        p0 = np.array([x0, y0], np.float32)
        near = p0 + u * float(np.dot(c - p0, u))
        t = (pts - near) @ u
        ends = (near + u * float(t.min()), near + u * float(t.max()))
        border = []
        for e, sel in ((ends[0], t <= t.min() + 3), (ends[1], t >= t.max() - 3)):
            q = pts[sel]
            border.append(bool(((q[:, 0] <= 2) | (q[:, 0] >= w - 3) |
                                (q[:, 1] <= 2) | (q[:, 1] >= h - 3)).any()))
        return PathLine(tuple(near), (float(u[0]), float(u[1])),
                        (tuple(ends[0]), tuple(ends[1])), width_cm, length_cm, tuple(border),
                        labels == label)


def to_body(line, img_w, img_h, height_cm):
    """
    The strip relative to the drone, in cm: (near point (fwd, left), unit direction
    (fwd, left), distance to each end along that direction (-, +)).
    """
    near = pixel_to_body(*line.near_px, img_w, img_h, height_cm)
    ahead = pixel_to_body(line.near_px[0] + line.dir_px[0] * 20,
                          line.near_px[1] + line.dir_px[1] * 20, img_w, img_h, height_cm)
    d = (ahead[0] - near[0], ahead[1] - near[1])
    n = math.hypot(*d) or 1.0
    d = (d[0] / n, d[1] / n)
    ends = []
    for e in line.ends_px:
        b = pixel_to_body(*e, img_w, img_h, height_cm)
        ends.append(float((b[0] - near[0]) * d[0] + (b[1] - near[1]) * d[1]))
    return (float(near[0]), float(near[1])), (float(d[0]), float(d[1])), (ends[0], ends[1])


def draw_path(vis, line):
    """Draw the recognised strip (orange) into a BGR image."""
    if line is None:
        return vis
    a = tuple(int(v) for v in line.ends_px[0])
    b = tuple(int(v) for v in line.ends_px[1])
    cv2.line(vis, a, b, (0, 165, 255), 2)
    n = tuple(int(v) for v in line.near_px)
    cv2.circle(vis, n, 5, (0, 165, 255), -1)
    cv2.putText(vis, f"pad {line.width_cm:.0f} cm", (n[0] + 6, n[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    return vis
