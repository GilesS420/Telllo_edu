"""
Ground map: the downward camera frames laid next to each other on the floor,
so the flown area becomes one picture of the ground (a mosaic).

Each frame is placed with the pose and the ToF height at the moment it was
taken (pixel_to_world, the same geometry as the H and puddle positions), in the
frame of the path view (S1 = (0, 0)). Where frames overlap, the new one is
blended with what is there. The map is as good as the odometry: drift shows as
seams.
"""

import math

import cv2
import numpy as np

from drone.downcam import pixel_to_world

RES_CM = 1.0            # cm per map pixel
HALF_CM = 800           # the map covers -800..800 cm around S1
EDGE = 0.08             # leave this part of each image edge out (vignetting, crop rest)
MIN_HEIGHT_CM = 20      # lower: too little ground in view, odometry unreliable
MAX_HEIGHT_CM = 300


class GroundMap:
    def __init__(self):
        n = int(2 * HALF_CM / RES_CM)
        self.size = n
        self.img = np.zeros((n, n), np.uint8)       # row = x (forward), col = y (left)
        self.seen = np.zeros((n, n), np.uint8)
        self.box = None                             # (r0, r1, c0, c1) covered part
        self.version = 0

    def clear(self):
        self.img[:] = 0
        self.seen[:] = 0
        self.box = None
        self.version += 1

    def _px(self, x, y):
        """Path frame (cm) -> map (col, row)."""
        return y / RES_CM + self.size / 2, x / RES_CM + self.size / 2

    def add(self, gray, height, pose):
        """Lay one downward frame on the map. pose = (x, y, z, yaw) in the path frame."""
        if not height or not MIN_HEIGHT_CM <= height <= MAX_HEIGHT_CM:
            return False
        h, w = gray.shape
        mx, my = int(w * EDGE), int(h * EDGE)
        src = [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]
        dst = [self._px(*pixel_to_world(cx, cy, w, h, height, pose)[:2]) for cx, cy in src]
        c0 = max(0, int(min(p[0] for p in dst)))
        c1 = min(self.size, int(math.ceil(max(p[0] for p in dst))) + 1)
        r0 = max(0, int(min(p[1] for p in dst)))
        r1 = min(self.size, int(math.ceil(max(p[1] for p in dst))) + 1)
        if c1 - c0 < 2 or r1 - r0 < 2:
            return False
        M = cv2.getAffineTransform(np.float32(src[:3]),
                                   np.float32([(c - c0, r - r0) for c, r in dst[:3]]))
        size = (c1 - c0, r1 - r0)
        part = cv2.warpAffine(gray, M, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.full((h, w), 255, np.uint8)[my:h - my, mx:w - mx],
                              M @ np.array([[1, 0, mx], [0, 1, my], [0, 0, 1]], np.float64),
                              size, flags=cv2.INTER_NEAREST) > 0
        img = self.img[r0:r1, c0:c1]
        seen = self.seen[r0:r1, c0:c1]
        old = mask & (seen > 0)
        new = mask & (seen == 0)
        img[new] = part[new]
        img[old] = ((img[old].astype(np.uint16) + part[old]) // 2).astype(np.uint8)
        seen[mask] = 1
        b = self.box
        self.box = ((r0, r1, c0, c1) if b is None else
                    (min(b[0], r0), max(b[1], r1), min(b[2], c0), max(b[3], c1)))
        self.version += 1
        return True

    def view(self, step=2):
        """(RGBA image, extent for imshow with x = y-left, y = x-forward) or None."""
        if self.box is None:
            return None
        r0, r1, c0, c1 = self.box
        g = self.img[r0:r1:step, c0:c1:step]
        a = np.where(self.seen[r0:r1:step, c0:c1:step] > 0, 255, 0).astype(np.uint8)
        rgba = np.dstack([g, g, g, a])
        half = self.size / 2
        extent = ((c0 - half) * RES_CM, (c1 - half) * RES_CM,
                  (r0 - half) * RES_CM, (r1 - half) * RES_CM)
        return rgba, extent

    def save_png(self, path):
        if self.box is None:
            return False
        r0, r1, c0, c1 = self.box
        # forward up, left on the left: flip rows (x up) and columns (+y left)
        cv2.imwrite(path, self.img[r0:r1, c0:c1][::-1, ::-1])
        return True
