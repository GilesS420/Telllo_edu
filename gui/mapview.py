"""
Top-view map with the terrain analysis. Drawn like the floor seen from above:
x (forward) points up, y (left) points left.

Mouse:
    click            add a waypoint there (height from the z field)
    drag a point     move it (snaps to 5 cm)
    right-click      remove the point under the mouse
    scroll           zoom, "Passend" in the toolbar fits everything again

Below the map: the height profile of the ground along the flown track.
"""

import math

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from drone import config as cfg

from .map3d import TERRAIN_CMAP, TERRAIN_NORM
from .theme import C, mpl_style

SNAP_CM = 5
PICK_PX = 12


class MapView:
    def __init__(self, parent, on_add, on_move, on_delete, on_select, on_hover):
        self.on_add, self.on_move, self.on_delete = on_add, on_move, on_delete
        self.on_select, self.on_hover = on_select, on_hover
        self.fig = Figure(figsize=(7, 5), dpi=100)
        gs = self.fig.add_gridspec(2, 2, height_ratios=(4.2, 1), width_ratios=(30, 1),
                                   hspace=0.3, wspace=0.04, left=0.11, right=0.9,
                                   top=0.97, bottom=0.09)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.cax = self.fig.add_subplot(gs[0, 1])
        self.pax = self.fig.add_subplot(gs[1, :])
        mpl_style(self.fig, self.ax, self.cax, self.pax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.configure(bg=C["panel"], highlightthickness=0)
        for ev, fn in (("button_press_event", self._press), ("motion_notify_event", self._motion),
                       ("button_release_event", self._release), ("scroll_event", self._scroll)):
            self.canvas.mpl_connect(ev, fn)
        self._drag = None
        self._waypoints = []
        self._zoom = None           # None = fit to content, else (cx, cy, span)
        self._fit = None
        self._terrain_key = None
        self._labels = []
        self._label_key = None
        self._puddle_patches = []
        self._puddle_key = None
        self._init_map()

    # ---------------------------------------------------------------- setup
    def _init_map(self):
        ax = self.ax
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_xaxis()                     # y (left) to the left
        ax.set_xlabel("y links (cm)", fontsize=8)
        ax.set_ylabel("x vooruit (cm)", fontsize=8)
        ax.grid(True, color=C["border"], linewidth=0.6)
        ax.set_axisbelow(True)
        self.image = ax.imshow([[float("nan")]], cmap=TERRAIN_CMAP, norm=TERRAIN_NORM,
                               origin="lower", extent=(0, 1, 0, 1), interpolation="nearest",
                               zorder=1, alpha=0.95)
        cb = self.fig.colorbar(self.image, cax=self.cax)
        cb.set_label("terreinhoogte (cm)", color=C["muted"], fontsize=8)
        cb.outline.set_edgecolor(C["border"])
        self.cax.tick_params(colors=C["muted"], labelsize=7)
        # axis cross at the start
        ax.annotate("", xy=(0, 40), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=2), zorder=6)
        ax.annotate("", xy=(40, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#22c55e", lw=2), zorder=6)
        ax.text(0, 48, "x", color="#ef4444", ha="center", fontsize=9, fontweight="bold")
        ax.text(48, 0, "y", color="#22c55e", va="center", ha="right", fontsize=9,
                fontweight="bold")
        g = cfg.GEOFENCE
        ax.plot([g["y"][0], g["y"][1], g["y"][1], g["y"][0], g["y"][0]],
                [g["x"][0], g["x"][0], g["x"][1], g["x"][1], g["x"][0]],
                "--", color=C["danger"], linewidth=1, alpha=0.6, zorder=2)
        P = lambda *a, **k: ax.plot([], [], *a, **k)[0]
        self.trail = P("-", color=C["trail"], linewidth=1.3, alpha=0.9, zorder=3)
        self.mission = P("-", color=C["accent"], linewidth=2.2, alpha=0.8, zorder=3)
        self.reached = P("o", color=C["ok"], markersize=9, linestyle="none", zorder=4)
        self.draft = P("o--", color=C["draft"], markersize=7, linewidth=1.5, zorder=5)
        self.selected = P("o", color=C["draft"], markersize=15, markerfacecolor="none",
                          markeredgewidth=2, linestyle="none", zorder=5)
        self.footprint = Polygon([[0, 0]], closed=True, fill=True, facecolor=C["puddle"],
                                 alpha=0.08, edgecolor=C["puddle"], linewidth=0.8, zorder=2)
        ax.add_patch(self.footprint)
        self.drone = Polygon([[0, 0]], closed=True, facecolor=C["danger"], edgecolor="white",
                             linewidth=1, zorder=7)
        ax.add_patch(self.drone)
        self.hover_txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, color=C["text"],
                                 fontsize=8, zorder=9, va="top",
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg"],
                                           edgecolor="none", alpha=0.8))
        # profile
        pax = self.pax
        pax.set_xlabel("afstand langs het gevlogen spoor (cm)", fontsize=8)
        pax.set_ylabel("grond (cm)", fontsize=8)
        pax.grid(True, color=C["border"], linewidth=0.5)
        self.profile_line, = pax.plot([], [], color=C["terrain_hi"], linewidth=1.2)
        self._profile_fill = None

    # ---------------------------------------------------------------- mouse
    def _xy(self, event):
        """Mouse -> mission frame (x, y) in cm, or None outside the map."""
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        return event.ydata, event.xdata          # plot x = mission y, plot y = mission x

    def _pick(self, event):
        best, dist = None, PICK_PX
        for i, p in enumerate(self._waypoints):
            px, py = self.ax.transData.transform((p[1], p[0]))
            d = math.hypot(px - event.x, py - event.y)
            if d < dist:
                best, dist = i, d
        return best

    @staticmethod
    def _snap(v):
        return round(v / SNAP_CM) * SNAP_CM

    def _press(self, event):
        xy = self._xy(event)
        if xy is None:
            return
        i = self._pick(event)
        if event.button == 3:
            if i is not None:
                self.on_delete(i)
            return
        if event.button != 1:
            return
        if i is not None:
            self._drag = i
            self.on_select(i)
        else:
            self.on_add(self._snap(xy[0]), self._snap(xy[1]))

    def _motion(self, event):
        xy = self._xy(event)
        if xy is None:
            self.on_hover(None)
            return
        x, y = self._snap(xy[0]), self._snap(xy[1])
        if self._drag is not None:
            self.on_move(self._drag, x, y, final=False)
        self.on_hover((xy[0], xy[1], self._pick(event)))
        self.widget.configure(cursor="hand2" if self._pick(event) is not None
                              or self._drag is not None else "crosshair")

    def _release(self, event):
        if self._drag is None:
            return
        xy = self._xy(event)
        i, self._drag = self._drag, None
        if xy is not None:
            self.on_move(i, self._snap(xy[0]), self._snap(xy[1]), final=True)
        else:
            self.on_move(i, None, None, final=True)

    def _scroll(self, event):
        if event.inaxes is not self.ax or self._fit is None:
            return
        cx, cy, span = self._zoom or self._fit
        f = 0.8 if event.button == "up" else 1.25
        # zoom around the mouse position
        mx, my = event.xdata, event.ydata
        self._zoom = (mx + (cx - mx) * f, my + (cy - my) * f, max(60, min(2000, span * f)))

    def fit(self):
        self._zoom = None

    # ---------------------------------------------------------------- drawing
    def _terrain(self, cells, version):
        if version == self._terrain_key:
            return
        self._terrain_key = version
        if not cells:
            self.image.set_data([[float("nan")]])
            self.image.set_extent((0, 1, 0, 1))
            return
        cell = cfg.TERRAIN_CELL_CM
        ii = [round(c[0] / cell - 0.5) for c in cells]
        jj = [round(c[1] / cell - 0.5) for c in cells]
        i0, j0 = min(ii), min(jj)
        rows, cols = max(ii) - i0 + 1, max(jj) - j0 + 1
        grid = [[float("nan")] * cols for _ in range(rows)]
        for (x, y, e, _, n), i, j in zip(cells, ii, jj):
            if n >= cfg.TERRAIN_MIN_SAMPLES:
                grid[i - i0][j - j0] = 0.0 if abs(e) < 4 else e
        # rows = x (vertical), cols = y (horizontal)
        self.image.set_data(grid)
        self.image.set_extent((j0 * cell, (j0 + cols) * cell, i0 * cell, (i0 + rows) * cell))

    def _profile(self, profile):
        pax = self.pax
        if self._profile_fill is not None:
            self._profile_fill.remove()
            self._profile_fill = None
        if not profile:
            self.profile_line.set_data([], [])
            return
        d = [p[0] for p in profile]
        e = [p[1] for p in profile]
        # light smoothing: the barometer is noisy
        k = 5
        sm = [sum(e[max(0, i - k):i + k + 1]) / len(e[max(0, i - k):i + k + 1])
              for i in range(len(e))]
        self.profile_line.set_data(d, sm)
        self._profile_fill = pax.fill_between(d, -100, sm, color="#7a5532", alpha=0.5,
                                              linewidth=0)
        pax.set_xlim(0, max(d[-1], 100))
        lo, hi = min(min(sm), -10), max(max(sm), 40)
        pax.set_ylim(lo - 5, hi + 5)

    def update(self, s):
        x, y, z, yaw = s["pose"]
        wps = s["waypoints"]
        self._waypoints = wps
        sw = lambda pts: ([p[1] for p in pts], [p[0] for p in pts])
        self.draft.set_data(*sw(wps))
        sel = s.get("selected")
        self.selected.set_data(*sw([wps[sel]] if sel is not None and sel < len(wps) else []))
        mission = s["mission"]
        self.mission.set_data(*sw(mission if mission != wps else []))
        self.reached.set_data(*sw(mission[:s["reached"]]))
        self.trail.set_data(*sw(s["trail"]))
        # drone: arrow-shaped triangle pointing along the nose
        a = math.radians(yaw)
        f = (math.cos(a), math.sin(a))
        l = (-f[1], f[0])
        tri = [(x + f[0] * 22, y + f[1] * 22), (x - f[0] * 12 + l[0] * 12, y - f[1] * 12 + l[1] * 12),
               (x - f[0] * 5, y - f[1] * 5), (x - f[0] * 12 - l[0] * 12, y - f[1] * 12 - l[1] * 12)]
        self.drone.set_xy([(py, px) for px, py in tri])
        fp = s.get("footprint")
        self.footprint.set_xy([(py, px) for px, py in fp] if fp else [(y, x)])
        self.footprint.set_visible(bool(fp))
        self._terrain(s["terrain"], s["terrain_version"])

        # puddles as circles with their real size
        pkey = tuple((p["id"], p["x"], p["y"], p["area_cm2"]) for p in s["puddles"])
        if pkey != self._puddle_key:
            self._puddle_key = pkey
            for patch in self._puddle_patches:
                patch.remove()
            self._puddle_patches = []
            for p in s["puddles"]:
                r = max(8.0, math.sqrt(max(p["area_cm2"], 1) / math.pi))
                c = Circle((p["y"], p["x"]), r, facecolor=C["puddle"], alpha=0.35,
                           edgecolor=C["puddle"], linewidth=1.5, zorder=4)
                self.ax.add_patch(c)
                self._puddle_patches.append(c)

        labels = [((p[1], p[0]), f" {i + 1}", C["draft"]) for i, p in enumerate(wps)]
        labels += [((p["y"], p["x"]), f" plas {p['id']}", C["puddle"]) for p in s["puddles"]]
        for o in s.get("obstacles", []):
            labels.append(((o["y"], o["x"]), f"{o['height']:+.0f} cm", C["warn"]))
        key = tuple((tuple(round(v) for v in pt), t) for pt, t, _ in labels)
        if key != self._label_key:
            self._label_key = key
            for t in self._labels:
                t.remove()
            self._labels = [self.ax.text(*pt, text, color=c, fontsize=8, zorder=8,
                                         fontweight="bold")
                            for pt, text, c in labels]

        # limits: fit everything (or the user's zoom), equal scale
        pts = [(p[1], p[0]) for p in wps + mission + s["trail"][-600:]] + [(y, x), (0, 0)]
        pts += [(p["y"], p["x"]) for p in s["puddles"]]
        pts += [(c[1], c[0]) for c in s["terrain"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 250) + 80
        self._fit = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, span)
        cx, cy, span = self._zoom or self._fit
        self.ax.set_xlim(cx + span / 2, cx - span / 2)    # inverted: +y on the left
        self.ax.set_ylim(cy - span / 2, cy + span / 2)
        self._profile(s["profile"])
        self.canvas.draw_idle()

    def set_hover_text(self, text):
        self.hover_txt.set_text(text)
