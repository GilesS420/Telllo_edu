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
The centre is the middle of the cross bar (the "-"), halfway between the side
bars. Close above the H, when it no longer fits in the image, find_center()
still finds that middle from the cross bar and the inner edges of the side bars.

Run this file to test it on saved images or a video:
    python -m drone.helipad dataset/
"""

import cv2
import numpy as np

from . import config as cfg
from .downcam import DownCam, cm_per_pixel


N = 48   # size of the straightened blob


def _h_score(m):
    """
    How well a straightened N x N 0/1 mask looks like an H: (score, transposed),
    score 0 = perfect, transposed = the side bars are horizontal in m. None = no H.
    """
    e, g0, g1 = int(0.1 * N), int(0.35 * N), int(0.65 * N)
    q, c0, c1 = int(0.25 * N), int(0.3 * N), int(0.7 * N)
    best = None
    for transposed, mm in ((False, m), (True, m.T)):   # bars vertical or horizontal
        bars = min(mm[:, :e].mean(), mm[:, -e:].mean())   # two full side bars
        rows = mm[:, g0:g1].mean(axis=1)                    # middle columns, per row
        cross = rows[c0:c1].max()                           # a filled cross bar ...
        gaps = max(rows[:q].mean(), rows[-q:].mean())       # ... with empty gaps around it
        if bars < 0.65 or cross < 0.8 or gaps > 0.2:
            continue
        score = 1.0 - (min(bars, cross) - gaps)
        if best is None or score < best[0]:
            best = (score, transposed)
    return best


def _runs(flags):
    """[(start, end)] of the runs of True values (end inclusive)."""
    d = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _cross_centre(mm):
    """
    Middle of the cross bar in a straightened H with vertical side bars:
    (x, y) halfway between the inner edges of the side bars, in the middle of
    the cross bar. None if that structure isn't clear.
    """
    n = mm.shape[1]
    bars = _runs(mm.mean(axis=0) > 0.6)
    left = [r for r in bars if r[0] < n * 0.3]
    right = [r for r in bars if r[1] > n * 0.7]
    if not left or not right:
        return None
    lo, hi = left[0][1] + 1, right[-1][0] - 1          # inner edges of the side bars
    if hi - lo < 2:
        return None
    rows = mm[:, lo:hi + 1].mean(axis=1)
    cross = _runs(rows > 0.5)
    if not cross:
        return None
    mid = mm.shape[0] / 2
    a, b = min(cross, key=lambda r: abs((r[0] + r[1]) / 2 - mid))
    return (lo + hi + 1) / 2, (a + b + 1) / 2


class Helipad:
    def __init__(self, cx, cy, size_px, score, contour, bar_dir=None, dark=True):
        self.cx, self.cy = cx, cy      # centre (middle of the cross bar), pixels
        self.size_px = size_px         # longest side of the H; None = H bigger than the image
        self.score = score             # difference with an ideal H, 0 = perfect
        self.contour = contour         # None for a centre found from the cross bar only
        self.bar_dir = bar_dir         # unit vector (x, y) along the side bars, in the image
        self.dark = dark               # True: dark H on light paper


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
        if best is not None:
            # dark H on light paper or the other way round (the mask it came from
            # doesn't say: a light mask also holds the H as a hole)
            x, y, w, h = cv2.boundingRect(best.contour)
            inside = np.zeros((h, w), np.uint8)
            cv2.drawContours(inside, [best.contour - (x, y)], -1, 1, -1)
            roi = blur[y:y + h, x:x + w]
            if inside.any() and (inside == 0).any():
                best.dark = roi[inside > 0].mean() < roi[inside == 0].mean()
        return best

    def find_center(self, gray, bar_dir, dark=True, size_px=None):
        """
        Centre of an H that is too big for the image (close above it): the middle
        of the cross bar, halfway between the inner edges of the two side bars.
        bar_dir/dark come from the last full detection (the H doesn't turn while
        the drone holds its heading). The structure closest to the image centre
        is used. size_px: expected size of the H in this image, if known; then
        only the part of the side bars next to the middle is used (the H may
        still fit the image lengthwise but not sideways).
        Returns a Helipad without contour, or None.
        """
        H, W = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (blur < thr) if dark else (blur > thr)
        ux, uy = bar_dir                       # along the side bars
        vx, vy = -uy, ux                       # across them
        ox, oy = W / 2, H / 2
        yy, xx = np.mgrid[0:H, 0:W]
        u = (xx - ox) * ux + (yy - oy) * uy
        v = (xx - ox) * vx + (yy - oy) * vy
        # across the bars: the side bars are (almost) fully ink along their length
        near_mid = np.abs(u) <= 0.35 * size_px if size_px else np.ones_like(ink)
        if not near_mid.any():
            return None
        vi = np.round(v[near_mid]).astype(np.int32)
        v0 = vi.min()
        cnt = np.bincount(vi - v0)
        frac = np.bincount(vi - v0, ink[near_mid].astype(np.float64)) / np.maximum(cnt, 1)
        frac[cnt < 0.3 * cnt.max()] = 0.0      # image corners: too few pixels to judge
        bars = _runs(frac > 0.6)
        best = None
        for (a0, a1), (b0, b1) in zip(bars, bars[1:]):
            lo, hi = a1 + 1 + v0, b0 - 1 + v0   # inner edges, v coordinates
            if hi - lo < 6:
                continue
            if best is None or abs((lo + hi) / 2) < abs(sum(best) / 2):
                best = (lo, hi)
        if best is None:
            return None
        lo, hi = best
        w = hi - lo + 1
        # along the bars, between them: the cross bar, with light gaps on both sides
        sel = (v >= lo + 0.15 * w) & (v <= hi - 0.15 * w)
        if not sel.any():
            return None
        ui = np.round(u[sel]).astype(np.int32)
        u0 = ui.min()
        cnt = np.bincount(ui - u0)
        frac = np.bincount(ui - u0, ink[sel].astype(np.float64)) / np.maximum(cnt, 1)
        cross = [(a, b) for a, b in _runs(frac > 0.6)
                 if a >= 2 and b <= len(frac) - 3 and 0.15 * w <= b - a + 1 <= 1.5 * w
                 and frac[max(a - 3, 0):a].mean() < 0.4 and frac[b + 1:b + 4].mean() < 0.4]
        if not cross:
            return None
        a, b = min(cross, key=lambda r: abs((r[0] + r[1]) / 2 + u0))
        uc, vc = (a + b) / 2 + u0, (lo + hi) / 2
        cx, cy = ox + uc * ux + vc * vx, oy + uc * uy + vc * vy
        if not (0 <= cx < W and 0 <= cy < H):
            return None
        return Helipad(cx, cy, None, 0.0, None, bar_dir, dark)

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
        straight = cv2.warpAffine(blob, warp, (N, N), flags=cv2.INTER_NEAREST)
        res = _h_score(straight)
        if res is None or res[0] > cfg.HELIPAD_MAX_SCORE:
            return None
        score, transposed = res
        # side bars run along box[1] -> box[0] (warped y axis), or along the x axis
        d = box[2] - box[1] if transposed else box[0] - box[1]
        bar_dir = tuple(d / max(np.hypot(*d), 1e-6))
        # centre: middle of the cross bar, mapped back from the straightened blob
        centre = _cross_centre(straight.T if transposed else straight)
        if centre is not None:
            px, py = centre[::-1] if transposed else centre
            back = cv2.invertAffineTransform(warp)
            cx, cy = back @ np.array([px, py, 1.0]) + (x - 1, y - 1)
        else:
            m = cv2.moments(c)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        return Helipad(float(cx), float(cy), max(rw, rh), score, c, bar_dir)


def draw_helipad(vis, pad):
    """Draw a found H on a BGR image (in place)."""
    if pad is None:
        return vis
    if pad.contour is not None:
        cv2.drawContours(vis, [pad.contour], -1, (0, 220, 0), 2)
    cx, cy = int(pad.cx), int(pad.cy)
    if pad.bar_dir is not None:   # the cross bar ("-") the drone aims at
        vx, vy = -pad.bar_dir[1] * 20, pad.bar_dir[0] * 20
        cv2.line(vis, (int(cx - vx), int(cy - vy)), (int(cx + vx), int(cy + vy)), (0, 220, 0), 2)
    cv2.drawMarker(vis, (cx, cy), (0, 220, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(vis, "H" if pad.contour is not None else "-", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
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
