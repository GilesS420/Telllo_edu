"""
3D view of the mission: axis cross at the start, the planned path, the flown
trail, a small drone model (turns with yaw, tilts with pitch/roll), the H
landing pad seen this flight and the measured terrain as coloured tiles.

Rotate by dragging, zoom with the scroll wheel. The axis limits glide to new
values instead of jumping, and x and y always have the same scale.
"""

import math
import tkinter as tk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from drone import config as cfg

from .theme import C, mpl_style

matplotlib.rcParams["axes3d.grid"] = True

# Viewing angles (mission frame: x = forward, y = left, z = up)
VIEWS = {
    "3D": (24, 205),          # from behind-right of the start position
    "Boven": (90, 180),       # top view: forward = up, left = left
    "Zijkant": (0, -90),      # side view from the right: forward = right
    "Achter": (8, 180),       # from behind, looking forward
}

# terrain colours: dips blue, floor grey, obstacles brown -> orange
TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain", [(0.0, "#2563eb"), (0.36, "#334155"), (0.44, "#3b4757"),
                (0.6, "#a16207"), (1.0, "#fbbf24")])
TERRAIN_NORM = Normalize(-40, 60)

DRONE_ARM = 16      # cm, drawn a bit larger than the real Tello for visibility


def _rgba(hex_color, alpha=1.0):
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (alpha,)


class Map3D:
    def __init__(self, parent):
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        mpl_style(self.fig, self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.configure(bg=C["panel"], highlightthickness=0)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.zoom = 1.0
        self.view = "3D"
        self.show_terrain = True
        self._lim = None            # current (cx, cy, span, top), glides to the target
        self._labels = []
        self._label_key = None
        self._terrain_art = None
        self._terrain_key = None
        self._style_axes()
        self._init_artists()
        self.set_view("3D", draw=False)

    # ------------------------------------------------------------- setup
    def _style_axes(self):
        ax = self.ax
        pane = _rgba(C["card"], 1.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(pane)
            axis._axinfo["grid"].update(color=_rgba(C["border"], 0.9), linewidth=0.6)
            axis.line.set_color(C["border"])
            axis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        ax.zaxis.set_pane_color(_rgba("#18212b", 1.0))   # the floor
        ax.tick_params(colors=C["muted"], labelsize=7, pad=0)
        for axis, text in ((ax.xaxis, "x vooruit (cm)"), (ax.yaxis, "y links (cm)"),
                           (ax.zaxis, "z hoogte (cm)")):
            axis.set_label_text(text, color=C["muted"], fontsize=8)
            axis.labelpad = 2

    def _init_artists(self):
        ax = self.ax
        line = lambda *a, **k: ax.plot([], [], [], *a, **k)[0]
        # axis cross at the start: x red, y green, z blue (like most 3D tools)
        L = 45
        for (dx, dy, dz), color, text in (((L, 0, 0), "#ef4444", "x"), ((0, L, 0), "#22c55e", "y"),
                                          ((0, 0, L), "#3b82f6", "z")):
            ax.quiver(0, 0, 0, dx, dy, dz, color=color, linewidth=2, arrow_length_ratio=0.22)
            ax.text(dx * 1.18, dy * 1.18, dz * 1.18, text, color=color, fontsize=9,
                    fontweight="bold", ha="center", va="center")
        ax.plot([0], [0], [0], marker="s", color=C["text"], markersize=5, linestyle="none")
        self.terrain_floor = None
        self.draft = line("o--", color=C["draft"], markersize=5, linewidth=1.4)
        self.selected = line("o", color=C["draft"], markersize=13, markerfacecolor="none",
                             markeredgewidth=2, linestyle="none")
        self.connector = line(":", color=C["draft"], alpha=0.6)
        self.mission = line("-", color=C["accent"], linewidth=2.2)
        self.mission_pts = line("o", color=C["accent"], markersize=5, linestyle="none")
        self.reached = line("o", color=C["ok"], markersize=8, linestyle="none")
        self.trail = line("-", color=C["trail"], linewidth=1.4, alpha=0.9)
        self.drop = line("--", color=C["muted"], linewidth=0.8)
        self.shadow = line("o", color="black", alpha=0.45, markersize=9, linestyle="none")
        self.arms = [line("-", color=C["text"], linewidth=2.5) for _ in range(2)]
        self.rotors = [line("-", color=C["danger"], linewidth=1.5) for _ in range(4)]
        self.nose = line("-", color=C["danger"], linewidth=3)
        self.pad = line("s", color=C["cyan"], markersize=11, alpha=0.75, linestyle="none")
        self.footprint = line("-", color=C["cyan"], linewidth=0.8, alpha=0.5)

    # ------------------------------------------------------------- view
    def set_view(self, name, draw=True):
        self.view = name
        elev, azim = VIEWS[name]
        self.ax.view_init(elev=elev, azim=azim)
        # hide the axis that points at the viewer in the 2D views (only clutters)
        for axis, label, hidden in ((self.ax.zaxis, "z hoogte (cm)", name == "Boven"),
                                    (self.ax.yaxis, "y links (cm)", name == "Zijkant"),
                                    (self.ax.xaxis, "x vooruit (cm)", name == "Achter")):
            axis.set_label_text("" if hidden else label)
            axis.set_major_formatter(NullFormatter() if hidden else ScalarFormatter())
        if draw:
            self.canvas.draw_idle()

    def reset_zoom(self):
        self.zoom = 1.0

    def _on_scroll(self, event):
        self.zoom = min(4.0, max(0.25, self.zoom * (0.85 if event.button == "up" else 1.18)))

    def _limits(self, points):
        xs = [0.0] + [p[0] for p in points]
        ys = [0.0] + [p[1] for p in points]
        zs = [0.0] + [p[2] for p in points]
        m = 50
        lo_x, hi_x = min(xs) - m, max(xs) + m
        lo_y, hi_y = min(ys) - m, max(ys) + m
        span = max(hi_x - lo_x, hi_y - lo_y, 200) * self.zoom
        top = max(max(zs) + m, 150)
        target = ((lo_x + hi_x) / 2, (lo_y + hi_y) / 2, span, top)
        if self._lim is None:
            self._lim = target
        else:   # glide: no jumping axes while flying
            self._lim = tuple(c + (t - c) * 0.25 for c, t in zip(self._lim, target))
        cx, cy, span, top = self._lim
        self.ax.set_xlim(cx - span / 2, cx + span / 2)
        self.ax.set_ylim(cy - span / 2, cy + span / 2)
        self.ax.set_zlim(min(0, self._zmin), top)
        self.ax.set_box_aspect((1, 1, max((top - min(0, self._zmin)) / span, 0.2)))

    # ------------------------------------------------------------- drawing
    @staticmethod
    def _set(artist, pts):
        if pts:
            xs, ys, zs = zip(*pts)
            artist.set_data_3d(list(xs), list(ys), list(zs))
        else:
            artist.set_data_3d([], [], [])

    def _drone(self, x, y, z, yaw, pitch, roll):
        """Quadcopter X-frame, rotated by yaw and tilted by pitch/roll."""
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        p, r = math.radians(pitch or 0), math.radians(roll or 0)

        def world(fwd, left, up=0.0):
            up = up + fwd * math.sin(p) + left * math.sin(r)   # nose up = +pitch
            return (x + fwd * cy - left * sy, y + fwd * sy + left * cy, z + up)

        tips = []
        for k, ang in enumerate((45, 135, 225, 315)):
            a = math.radians(ang)
            tips.append((DRONE_ARM * math.cos(a), DRONE_ARM * math.sin(a)))
        self._set(self.arms[0], [world(*tips[0]), world(*tips[2])])
        self._set(self.arms[1], [world(*tips[1]), world(*tips[3])])
        for rotor, (f, l) in zip(self.rotors, tips):
            ring = [world(f + 7 * math.cos(t), l + 7 * math.sin(t), 1.5)
                    for t in (i * math.pi / 6 for i in range(13))]
            self._set(rotor, ring)
            rotor.set_color(C["danger"] if f > 0 else C["muted"])   # front rotors red
        self._set(self.nose, [world(0, 0), world(DRONE_ARM * 1.6, 0)])

    def _terrain(self, cells, version):
        key = (version, self.show_terrain)
        if key == self._terrain_key:
            return
        self._terrain_key = key
        if self._terrain_art is not None:
            self._terrain_art.remove()
            self._terrain_art = None
        cells = [c for c in cells if c[4] >= cfg.TERRAIN_MIN_SAMPLES]
        self._zmin = min([0.0] + [c[2] for c in cells])
        if not self.show_terrain or not cells:
            return
        h = cfg.TERRAIN_CELL_CM / 2
        quads, colors = [], []
        for x, y, e, _, _ in cells:
            e = 0.0 if abs(e) < 4 else e        # flatten sensor noise on the floor
            quads.append([(x - h, y - h, e), (x + h, y - h, e), (x + h, y + h, e),
                          (x - h, y + h, e)])
            colors.append(TERRAIN_CMAP(TERRAIN_NORM(e)))
        self._terrain_art = Poly3DCollection(quads, facecolors=colors, edgecolors=C["panel"],
                                             linewidths=0.3, alpha=0.9)
        self.ax.add_collection3d(self._terrain_art)

    _zmin = 0.0

    def update(self, s):
        """s: dict with everything to draw (built by the main window)."""
        x, y, z, yaw = s["pose"]
        wps, mission = s["waypoints"], s["mission"]
        self._set(self.draft, wps)
        sel = s.get("selected")
        self._set(self.selected, [wps[sel]] if sel is not None and sel < len(wps) else [])
        self._set(self.connector, [(x, y, z), wps[0]] if wps and not s["airborne"] else [])
        show_mission = mission and mission != wps
        self._set(self.mission, mission)
        self._set(self.mission_pts, mission if show_mission else [])
        self._set(self.reached, mission[:s["reached"]])
        self._set(self.trail, s["trail"])
        ground = s.get("ground_under") or 0.0
        self._set(self.drop, [(x, y, ground), (x, y, z)] if z > ground + 1 else [])
        self._set(self.shadow, [(x, y, ground)] if z > ground + 1 else [])
        self._drone(x, y, z, yaw, s.get("pitch"), s.get("roll"))
        pad = s.get("helipad")
        pad_pts = [(pad[0], pad[1], 0)] if pad is not None else []
        self._set(self.pad, pad_pts)
        fp = s.get("footprint")
        self._set(self.footprint, [(px, py, ground) for px, py in fp] if fp else [])
        self._terrain(s["terrain"], s["terrain_version"])

        # labels: waypoint numbers (coordinates only for the selected one) + the H
        labels = []
        for i, p in enumerate(wps):
            text = f" {i + 1}" + (f"  ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})" if i == sel else "")
            labels.append((p, text, C["draft"]))
        if show_mission:
            for i, p in enumerate(mission):
                labels.append((p, f" M{i + 1}", C["accent"]))
        for p in pad_pts:
            labels.append((p, " H", C["cyan"]))
        key = tuple((tuple(round(v) for v in pt), t) for pt, t, _ in labels)
        if key != self._label_key:
            self._label_key = key
            for t in self._labels:
                t.remove()
            self._labels = [self.ax.text(*pt, text, color=c, fontsize=8)
                            for pt, text, c in labels]

        self._limits(wps + mission + s["trail"][-400:] + [(x, y, z)]
                     + pad_pts)
        self.canvas.draw_idle()


def legend(parent):
    """Colour legend as a row of tk labels (no matplotlib legend over the plot)."""
    row = tk.Frame(parent, bg=C["panel"])
    for color, text in ((C["draft"], "gepland pad"), (C["accent"], "actieve missie"),
                        (C["ok"], "bereikt"), (C["trail"], "gevlogen"),
                        (C["danger"], "drone (neus)"), (C["cyan"], "H (landingsplaats)"),
                        ("#a16207", "terrein hoger"), ("#2563eb", "terrein lager")):
        tk.Label(row, text="●", fg=color, bg=C["panel"]).pack(side=tk.LEFT)
        tk.Label(row, text=text, fg=C["muted"], bg=C["panel"],
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(0, 10))
    return row
