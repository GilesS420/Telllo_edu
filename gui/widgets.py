"""
Small reusable widgets drawn on tk.Canvas: they are cheap to redraw, so they
can animate at 30 fps next to the (slower) matplotlib views.

Every instrument has set(...) for the newest value and tick() that moves the
drawing a bit closer to it, so the needles glide instead of jumping.
"""

import math
import time
import tkinter as tk
from tkinter import ttk

from .theme import C, F, level_color


def card(parent, title=None, padding=(10, 8)):
    """A dark card with an optional small title. Returns (outer frame, header, body)."""
    outer = ttk.Frame(parent, style="Card.TFrame", padding=padding)
    header = None
    if title:
        header = ttk.Frame(outer, style="Card.TFrame")
        header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(header, text=title.upper(), style="Title.TLabel").pack(side=tk.LEFT)
    body = ttk.Frame(outer, style="Card.TFrame")
    body.pack(fill=tk.BOTH, expand=True)
    return outer, header, body


def _approach(cur, target, k=0.25):
    if cur is None or target is None:
        return target
    return cur + (target - cur) * k


def _approach_angle(cur, target, k=0.25):
    if cur is None or target is None:
        return target
    d = (target - cur + 180) % 360 - 180
    return cur + d * k


class Segmented(ttk.Frame):
    """Row of buttons of which one is active (like a segmented control)."""

    def __init__(self, parent, options, command, value=None, style="Card.TFrame"):
        super().__init__(parent, style=style)
        self.command = command
        self.buttons = {}
        for name in options:
            b = ttk.Button(self, text=name, style="Seg.TButton", width=len(name) + 2,
                           command=lambda n=name: self.set(n, notify=True))
            b.pack(side=tk.LEFT, padx=(0, 2))
            self.buttons[name] = b
        self.value = None
        if value is not None:
            self.set(value)

    def set(self, value, notify=False):
        self.value = value
        for name, b in self.buttons.items():
            b.configure(style="SegOn.TButton" if name == value else "Seg.TButton")
        if notify:
            self.command(value)


class StatTile(ttk.Frame):
    """Small label + big value, for a sensor reading."""

    def __init__(self, parent, label, unit="", width=8):
        super().__init__(parent, style="Card.TFrame", padding=(8, 5))
        ttk.Label(self, text=label, style="Muted.TLabel").pack(anchor=tk.W)
        row = ttk.Frame(self, style="Card.TFrame")
        row.pack(anchor=tk.W)
        self.value = tk.Label(row, text="–", bg=C["card"], fg=C["text"], font=F["big"],
                              width=width, anchor=tk.W)
        self.value.pack(side=tk.LEFT)
        self.unit = ttk.Label(row, text=unit, style="Muted.TLabel")
        self.unit.pack(side=tk.LEFT, anchor=tk.S, pady=(0, 3))
        self._last = None

    def set(self, text, color=None):
        key = (text, color)
        if key != self._last:
            self._last = key
            self.value.configure(text=text, fg=color or C["text"])


class BatteryBar(tk.Canvas):
    def __init__(self, parent, width=74, height=24, bg=C["panel"]):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self.w, self.h = width, height
        self.level = None
        self.shown = None

    def set(self, level):
        self.level = level

    def tick(self):
        if self.level is None:
            target = None
        else:
            target = float(self.level)
        self.shown = _approach(self.shown, target, 0.2)
        self.delete("all")
        w, h = self.w - 6, self.h
        self.create_rectangle(1, 3, w, h - 3, outline=C["muted"], width=1.5)
        self.create_rectangle(w + 1, h / 2 - 4, w + 4, h / 2 + 4, fill=C["muted"], outline="")
        if self.shown is None:
            self.create_text(w / 2, h / 2, text="?", fill=C["muted"], font=F["bold"])
            return
        frac = max(0.0, min(1.0, self.shown / 100))
        color = level_color(min(1.0, frac / 0.6))
        self.create_rectangle(3, 5, 3 + (w - 5) * frac, h - 5, fill=color, outline="")
        self.create_text(w / 2, h / 2, text=f"{self.level:.0f}%", fill="white",
                         font=F["bold"])


class AttitudeIndicator(tk.Canvas):
    """Artificial horizon: pitch and roll of the drone."""

    PX_PER_DEG = 2.4

    def __init__(self, parent, size=150, bg=C["card"]):
        super().__init__(parent, width=size, height=size + 16, bg=bg, highlightthickness=0)
        self.size = size
        self.target = (0.0, 0.0)
        self.pitch = self.roll = 0.0
        self.valid = False
        n = 96
        self._circle = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
                        for i in range(n)]

    def set(self, pitch, roll):
        self.valid = pitch is not None and roll is not None
        if self.valid:
            self.target = (float(pitch), float(roll))

    def tick(self):
        self.pitch = _approach(self.pitch, self.target[0], 0.3)
        self.roll = _approach(self.roll, self.target[1], 0.3)
        s = self.size
        cx = cy = s / 2
        r = s / 2 - 8
        self.delete("all")
        a = math.radians(-self.roll)                 # horizon turns opposite to the roll
        nx, ny = -math.sin(a), math.cos(a)           # normal pointing to the ground (screen)
        off = self.pitch * self.PX_PER_DEG          # nose up -> horizon moves down
        # sky = full circle, ground = circle clipped by the horizon (half plane)
        pts = [(cx + r * x, cy + r * y) for x, y in self._circle]
        self.create_polygon(*sum(pts, ()), fill=C["sky"], outline="")
        ground = _clip(pts, (cx + nx * off, cy + ny * off), (nx, ny))
        if len(ground) >= 3:
            self.create_polygon(*sum(ground, ()), fill=C["ground"], outline="")
        # horizon + pitch ladder
        tx, ty = math.cos(a), math.sin(a)            # along the horizon
        for deg in range(-30, 31, 10):
            d = off - deg * self.PX_PER_DEG
            half = r * 0.9 if deg == 0 else (22 if deg % 20 == 0 else 12)
            px, py = cx + nx * d, cy + ny * d
            if abs(d) > r - 6 and deg != 0:
                continue
            self.create_line(px - tx * half, py - ty * half, px + tx * half, py + ty * half,
                             fill="white", width=2 if deg == 0 else 1)
            if deg and deg % 20 == 0:
                self.create_text(px + tx * (half + 9), py + ty * (half + 9), text=str(abs(deg)),
                                 fill="white", font=F["tiny"])
        # bezel, roll scale and pointer
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=C["border"], width=3)
        for deg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            t = math.radians(deg - 90)
            r0 = r - (9 if deg % 30 == 0 else 5)
            self.create_line(cx + r0 * math.cos(t), cy + r0 * math.sin(t),
                             cx + r * math.cos(t), cy + r * math.sin(t), fill="white")
        t = math.radians(-self.roll - 90)
        tip = (cx + (r - 11) * math.cos(t), cy + (r - 11) * math.sin(t))
        bl = (cx + (r - 20) * math.cos(t - 0.07), cy + (r - 20) * math.sin(t - 0.07))
        br = (cx + (r - 20) * math.cos(t + 0.07), cy + (r - 20) * math.sin(t + 0.07))
        self.create_polygon(*tip, *bl, *br, fill=C["warn"], outline="")
        # fixed aircraft symbol
        y = cy
        self.create_line(cx - 34, y, cx - 12, y, cx - 6, y + 6, fill=C["warn"], width=3)
        self.create_line(cx + 34, y, cx + 12, y, cx + 6, y + 6, fill=C["warn"], width=3)
        self.create_oval(cx - 2, y - 2, cx + 2, y + 2, fill=C["warn"], outline="")
        label = (f"P {self.target[0]:+.0f}°   R {self.target[1]:+.0f}°" if self.valid
                 else "geen data")
        self.create_text(cx, s + 14, text=label, fill=C["muted"], font=F["tiny"], anchor=tk.S)


def _clip(poly, point, normal):
    """Part of a convex polygon on the side the normal points to (Sutherland-Hodgman)."""
    px, py = point
    nx, ny = normal
    side = lambda q: (q[0] - px) * nx + (q[1] - py) * ny
    out = []
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        sc, sp = side(cur), side(prev)
        if (sc >= 0) != (sp >= 0):
            t = sp / (sp - sc)
            out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
        if sc >= 0:
            out.append(cur)
    return out


class Compass(tk.Canvas):
    """
    Heading in the mission frame, drawn like the top view: x (forward) points up,
    y (left) points left. The notch on the ring is the heading being held.
    """

    def __init__(self, parent, size=150, bg=C["card"]):
        super().__init__(parent, width=size, height=size + 16, bg=bg, highlightthickness=0)
        self.size = size
        self.yaw = None
        self.shown = 0.0
        self.ref = 0.0

    def set(self, yaw, ref):
        self.yaw, self.ref = yaw, ref

    @staticmethod
    def _dir(deg):
        t = math.radians(deg)
        return -math.sin(t), -math.cos(t)          # screen dx, dy (x forward = up)

    def tick(self):
        self.shown = _approach_angle(self.shown, self.yaw, 0.3) or 0.0
        s = self.size
        cx = cy = s / 2
        r = s / 2 - 8
        self.delete("all")
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=C["card2"], outline=C["border"],
                         width=2)
        for deg in range(0, 360, 10):
            dx, dy = self._dir(deg)
            r0 = r - (10 if deg % 90 == 0 else 6 if deg % 30 == 0 else 3)
            self.create_line(cx + dx * r0, cy + dy * r0, cx + dx * r, cy + dy * r,
                             fill=C["muted"] if deg % 30 else C["text"])
        for deg, text in ((0, "x"), (90, "y"), (180, "−x"), (270, "−y")):
            dx, dy = self._dir(deg)
            self.create_text(cx + dx * (r - 19), cy + dy * (r - 19), text=text,
                             fill=C["muted"], font=F["tiny"])
        dx, dy = self._dir(self.ref)
        self.create_line(cx + dx * (r - 2), cy + dy * (r - 2), cx + dx * (r + 6),
                         cy + dy * (r + 6), fill=C["ok"], width=4)
        if self.yaw is None:
            self.create_text(cx, cy, text="–", fill=C["muted"], font=F["big"])
            return
        dx, dy = self._dir(self.shown)
        px, py = -dy, dx
        tip = (cx + dx * (r - 14), cy + dy * (r - 14))
        tail = (cx - dx * 22, cy - dy * 22)
        self.create_polygon(*tip, cx + px * 11 - dx * 16, cy + py * 11 - dy * 16, *tail,
                            cx - px * 11 - dx * 16, cy - py * 11 - dy * 16,
                            fill=C["danger"], outline="white", width=1)
        self.create_text(cx, s + 14, text=f"{self.yaw:+.0f}°  (houdt {self.ref:+.0f}°)",
                         fill=C["muted"], font=F["tiny"], anchor=tk.S)


class HeightGauge(tk.Canvas):
    """
    Side view of the drone above the ground: the ground block comes from the
    terrain analysis, the drone sits at its barometer altitude, the bracket is
    the ToF distance between both.
    """

    def __init__(self, parent, width=104, height=150, bg=C["card"]):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self.w, self.h = width, height
        self.vals = {}
        self.shown = {}
        self.top = 150.0

    def set(self, alt, tof, ground):
        self.vals = {"alt": alt, "tof": tof, "ground": ground}

    def tick(self):
        for k, v in self.vals.items():
            self.shown[k] = _approach(self.shown.get(k), v, 0.3)
        alt, tof, ground = (self.shown.get(k) for k in ("alt", "tof", "ground"))
        if alt is None and tof is not None:
            alt = tof + (ground or 0)
        want = max(150.0, (alt or 0) + 40, (ground or 0) + 60)
        self.top = _approach(self.top, want, 0.1)
        self.delete("all")
        x0, x1 = 30, 62
        y_of = lambda v: self.h - 14 - (v + 20) / (self.top + 20) * (self.h - 24)
        # scale
        step = 50 if self.top < 300 else 100
        for v in range(0, int(self.top) + 1, step):
            y = y_of(v)
            self.create_line(x0 - 4, y, x0, y, fill=C["muted"])
            self.create_text(x0 - 6, y, text=str(v), anchor=tk.E, fill=C["muted"],
                             font=F["tiny"])
        self.create_line(x0, y_of(-20), x0, y_of(self.top), fill=C["border"])
        # floor + ground block
        self.create_line(x0, y_of(0), x1 + 30, y_of(0), fill=C["muted"], dash=(2, 2))
        g = ground or 0.0
        self.create_rectangle(x0 + 1, y_of(min(g, 0)), x1 + 30, y_of(-20), fill="#3a2c1f",
                              outline="")
        if ground is not None:
            color = C["terrain_hi"] if abs(ground) >= 15 else "#5c4630"
            self.create_rectangle(x0 + 1, y_of(g), x1 + 30, y_of(min(g, 0)), fill=color,
                                  outline="")
        if alt is None:
            self.create_text((x0 + x1) / 2 + 10, self.h / 2, text="geen\ndata",
                             fill=C["muted"], font=F["tiny"], justify=tk.CENTER)
            return
        ya = y_of(alt)
        # ToF bracket from the drone to the ground below it
        if tof is not None:
            yg = y_of(alt - tof)
            xm = x1 + 14
            self.create_line(xm, ya + 4, xm, yg, fill=C["cyan"], arrow=tk.LAST, width=1.5)
            self.create_text(xm + 3, (ya + yg) / 2, text=f"{self.vals['tof']:.0f}",
                             anchor=tk.W, fill=C["cyan"], font=F["tiny"])
        # drone
        cx = (x0 + x1) / 2 + 2
        self.create_line(cx - 14, ya, cx + 14, ya, fill=C["text"], width=3)
        for dx in (-14, 14):
            self.create_oval(cx + dx - 5, ya - 3, cx + dx + 5, ya, outline=C["text"])
        self.create_text(cx, ya - 9, text=f"{self.vals['alt'] or alt:.0f}", fill=C["text"],
                         font=F["tiny"])


class LineChart(tk.Canvas):
    """Fast multi-series time plot (newest data on the right)."""

    def __init__(self, parent, title, unit, series, fixed=None, symmetric=False,
                 width=360, height=150, bg=C["card"]):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self.title, self.unit, self.series = title, unit, series
        self.fixed, self.symmetric = fixed, symmetric
        self.bind("<Configure>", lambda e: self._size(e.width, e.height))
        self.W, self.H = width, height
        self._data, self._window = None, 60

    def _size(self, w, h):
        self.W, self.H = w, h
        if self._data:
            self.draw(self._data, self._window)

    def draw(self, data, window):
        self._data, self._window = data, window
        self.delete("all")
        W, H = self.W, self.H
        L, R, T, B = 40, 10, 24, 18
        self.create_text(10, 11, text=f"{self.title}  ({self.unit})", anchor=tk.W, fill=C["text"],
                         font=F["title"])
        # legend with current values (right aligned)
        x = W - R
        for key, label, color in reversed(self.series):
            vals = [v for v in data.get(key, []) if v is not None]
            text = f"{label} {vals[-1]:.1f}" if vals else f"{label} –"
            item = self.create_text(x, 11, text=text, anchor=tk.E, fill=color, font=F["tiny"])
            x = self.bbox(item)[0] - 10
        ts = data.get("t", [])
        vals = [v for key, _, _ in self.series for v in data.get(key, []) if v is not None]
        if self.fixed:
            lo, hi = self.fixed
        elif vals:
            lo, hi = min(vals), max(vals)
            if self.symmetric:
                m = max(abs(lo), abs(hi), 1e-6)
                lo, hi = -m, m
        else:
            lo, hi = 0, 1
        if hi - lo < 1e-6:
            lo, hi = lo - 1, hi + 1
        pad = (hi - lo) * 0.1
        lo, hi = lo - (0 if self.fixed else pad), hi + (0 if self.fixed else pad)
        px = lambda t: L + (t + window) / window * (W - L - R)
        py = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
        for v in _nice_ticks(lo, hi, 4):
            y = py(v)
            self.create_line(L, y, W - R, y, fill=C["border"], dash=(1, 3) if v else ())
            self.create_text(L - 5, y, text=_fmt(v), anchor=tk.E, fill=C["muted"],
                             font=F["tiny"])
        for s in range(0, int(window) + 1, max(10, int(window) // 4 // 10 * 10)):
            x = px(-s)
            self.create_text(x, H - 3, text=f"{-s}s" if s else "nu", anchor=tk.S,
                             fill=C["faint"], font=F["tiny"])
        if not ts:
            self.create_text(W / 2, H / 2, text="nog geen data", fill=C["faint"], font=F["small"])
            return
        step = max(1, len(ts) // max(1, int(W - L - R)))
        for key, _, color in self.series:
            seg = []
            for t, v in zip(ts[::step], data.get(key, [])[::step]):
                if v is None:
                    if len(seg) >= 4:
                        self.create_line(*seg, fill=color, width=1.6)
                    seg = []
                    continue
                seg += [px(t), py(min(hi, max(lo, v)))]
            if len(seg) >= 4:
                self.create_line(*seg, fill=color, width=1.6)


def _nice_ticks(lo, hi, n):
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v):
    return f"{v:.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.1f}"


class Toast:
    """Short message in the top bar instead of a pop-up window."""

    COLORS = {"info": C["accent"], "ok": C["ok"], "warn": C["warn"], "error": C["danger"]}

    def __init__(self, label):
        self.label = label
        self._until = 0

    def show(self, text, kind="info", seconds=4):
        self.label.configure(text=text, fg=self.COLORS.get(kind, C["text"]))
        self._until = time.time() + seconds

    def tick(self):
        if self._until and time.time() > self._until:
            self._until = 0
            self.label.configure(text="")
