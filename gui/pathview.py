"""
Path view: the flown path seen from above, with where the drone took off (S),
the H landing pads and the puddles it found. Drawn like the floor seen from
above, same as the map: x (forward) points up, y (left) points left.

Every takeoff starts a new mission frame (the drone itself is (0, 0) again).
FlightLog chains those flights: the pose where the drone stood before the
takeoff (DroneApp.frame_link) says where the new frame lies in the old one, so
all flights since the last Reset share the frame of the first flight (S1).
A Reset (drone moved by hand) starts a new path.

Scroll = zoom, "Passend" in the toolbar fits everything again.
"""

import math

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from .theme import C, mpl_style

SAME_PLACE_CM = 30      # H sightings / pads closer than this are the same pad
ARROW_EVERY_CM = 60     # direction arrows along the path


def to_world(T, x, y):
    """Point of a flight frame -> path frame. T = (x, y, yaw deg) of the frame's origin."""
    a = math.radians(T[2])
    c, s = math.cos(a), math.sin(a)
    return T[0] + x * c - y * s, T[1] + x * s + y * c


class Flight:
    """One mission frame: its place in the path frame and what was seen in it."""

    def __init__(self, frame_id, T):
        self.frame_id = frame_id
        self.T = T
        self.paths = []         # one polyline [(x, y), ...] per takeoff (own frame)
        self.starts = []        # takeoff spots (own frame)
        self.pads = []          # H places (own frame)
        self.puddles = []       # (id, x, y, w cm, h cm) (own frame)
        self.mission = []       # planned waypoints (own frame)

    @property
    def used(self):
        return bool(self.starts)


class FlightLog:
    """Everything flown since the last Reset, chained into the frame of the first flight."""

    def __init__(self):
        self.flights = []
        self._airborne = False

    def clear(self):
        """Forget the path; in the air, it starts again where the drone is now."""
        cur = self.flights[-1] if self.flights else None
        self.flights = []
        if cur is not None:
            fresh = Flight(cur.frame_id, (0.0, 0.0, 0.0))
            if self._airborne and cur.paths and cur.paths[-1]:
                here = cur.paths[-1][-1]
                fresh.starts.append(here)
                fresh.paths.append([here])
            self.flights.append(fresh)

    def update(self, app, slow=True):
        """Call often: follows the drone. slow=True also reads H's and puddles."""
        airborne = app.airborne     # read before frame_id: a takeoff resets the frame first
        frame_id = app.frame_id
        cur = self.flights[-1] if self.flights else None
        if cur is None or cur.frame_id != frame_id:
            link = app.frame_link
            if cur is not None and not cur.used:
                self.flights.pop()   # never flew in that frame (startup, aborted takeoff)
            if link is None:
                self.flights = []    # Reset: the drone may have been moved, new path
            T = (0.0, 0.0, 0.0)      # the first flight is the path frame
            if link is not None and cur is not None and self.flights:
                x, y = to_world(cur.T, link[0], link[1])
                T = (x, y, cur.T[2] + link[3])
            cur = Flight(frame_id, T)
            self.flights.append(cur)
            self._airborne = False

        x, y, _, _ = app.pose.get()
        if airborne and not self._airborne:
            cur.starts.append((x, y))
            cur.paths.append([(x, y)])
        self._airborne = airborne
        if airborne and cur.paths:
            path = cur.paths[-1]
            if not path or math.hypot(path[-1][0] - x, path[-1][1] - y) > 2:
                path.append((x, y))
        if slow:
            for px, py in app.seen_helipads():
                near = [i for i, p in enumerate(cur.pads)
                        if math.hypot(p[0] - px, p[1] - py) < SAME_PLACE_CM]
                if near:
                    cur.pads[near[0]] = (px, py)
                else:
                    cur.pads.append((px, py))
            cur.puddles = [(p.id, p.x, p.y, p.w_cm, p.h_cm)
                           for p in app.puddle_detector.puddles()]
            if app.current_waypoints:
                cur.mission = [w[:2] for w in app.current_waypoints]

    def to_world(self, x, y):
        """Point of the current flight frame -> path frame."""
        return to_world(self.flights[-1].T, x, y) if self.flights else (x, y)

    def scene(self):
        """Everything in the path frame, with labels, ready to draw or save."""
        used = [f for f in self.flights if f.used]
        many = len(used) > 1
        out = {"paths": [], "starts": [], "pads": [], "puddles": [], "mission": [],
               "flights": len(used)}
        for k, f in enumerate(used, 1):
            out["paths"] += [[to_world(f.T, *p) for p in path] for path in f.paths]
            for x, y in f.starts:
                out["starts"].append({"label": f"S{len(out['starts']) + 1}", "flight": k,
                                      "x": x, "y": y} | _xy(f.T, x, y))
            for x, y in f.pads:
                wx, wy = to_world(f.T, x, y)
                if not any(math.hypot(p["x"] - wx, p["y"] - wy) < SAME_PLACE_CM
                           for p in out["pads"]):
                    out["pads"].append({"label": f"H{len(out['pads']) + 1}", "flight": k,
                                        "x": wx, "y": wy})
            for pid, x, y, w, h in f.puddles:
                out["puddles"].append({"label": f"V{k}-P{pid}" if many else f"P{pid}",
                                       "flight": k, "id": pid, "w_cm": w, "h_cm": h}
                                      | _xy(f.T, x, y))
        if self.flights:
            f = self.flights[-1]
            out["mission"] = [to_world(f.T, *p) for p in f.mission]
        out["length_cm"] = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                               for path in out["paths"] for a, b in zip(path, path[1:]))
        return out


def _xy(T, x, y):
    wx, wy = to_world(T, x, y)
    return {"x": wx, "y": wy}


class PathView:
    def __init__(self, parent):
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.09)
        mpl_style(self.fig, self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.configure(bg=C["panel"], highlightthickness=0)
        self.canvas.mpl_connect("scroll_event", self._scroll)
        self._zoom = None           # None = fit to content, else (cx, cy, span)
        self._fit = None
        self._paths = []
        self._arrows = None
        self._marks = []
        self._mark_key = None
        self._init()

    def _init(self):
        ax = self.ax
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y links (cm)", fontsize=8)
        ax.set_ylabel("x vooruit (cm)", fontsize=8)
        ax.grid(True, color=C["border"], linewidth=0.6)
        ax.set_axisbelow(True)
        # axis cross at S1
        ax.annotate("", xy=(0, 30), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=1.5), zorder=2)
        ax.annotate("", xy=(30, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#22c55e", lw=1.5), zorder=2)
        P = lambda *a, **k: ax.plot([], [], *a, **k)[0]
        self.mission = P("--", color=C["faint"], linewidth=1, zorder=2)
        self.drone = Polygon([[0, 0]], closed=True, facecolor=C["danger"], edgecolor="white",
                             linewidth=1, zorder=9)
        ax.add_patch(self.drone)

    def _scroll(self, event):
        if event.inaxes is not self.ax or self._fit is None:
            return
        cx, cy, span = self._zoom or self._fit
        f = 0.8 if event.button == "up" else 1.25
        mx, my = event.xdata, event.ydata
        self._zoom = (mx + (cx - mx) * f, my + (cy - my) * f, max(60, min(3000, span * f)))

    def fit(self):
        self._zoom = None

    def _draw_paths(self, paths):
        # one line per takeoff; recreated only when a takeoff adds one
        while len(self._paths) > len(paths):
            self._paths.pop().remove()
        while len(self._paths) < len(paths):
            self._paths += self.ax.plot([], [], "-", color=C["trail"], linewidth=2.2,
                                        alpha=0.9, zorder=4, solid_capstyle="round")
        for line, path in zip(self._paths, paths):
            line.set_data([p[1] for p in path], [p[0] for p in path])
        # small arrows every ARROW_EVERY_CM: which way it flew
        xs, ys, us, vs = [], [], [], []
        for path in paths:
            run = 0.0
            for a, b in zip(path, path[1:]):
                d = math.hypot(b[0] - a[0], b[1] - a[1])
                run += d
                if run >= ARROW_EVERY_CM and d > 0:
                    run = 0.0
                    xs.append(b[1])
                    ys.append(b[0])
                    us.append((b[1] - a[1]) / d)
                    vs.append((b[0] - a[0]) / d)
        if self._arrows is not None:
            self._arrows.remove()
            self._arrows = None
        if xs:
            self._arrows = self.ax.quiver(xs, ys, us, vs, color=C["trail"], zorder=5,
                                          pivot="mid", scale=40, width=0.006,
                                          headwidth=4, headlength=4, headaxislength=3.5)

    def _draw_marks(self, s):
        key = (tuple((round(p["x"]), round(p["y"]), p["label"]) for p in s["starts"]),
               tuple((round(p["x"]), round(p["y"]), p["label"]) for p in s["pads"]),
               tuple((round(p["x"]), round(p["y"]), round(p["w_cm"] + p["h_cm"]), p["label"])
                     for p in s["puddles"]))
        if key == self._mark_key:
            return
        self._mark_key = key
        for m in self._marks:
            m.remove()
        ax, marks = self.ax, []
        for p in s["puddles"]:
            r = max((p["w_cm"] + p["h_cm"]) / 4, 8)
            marks.append(ax.add_patch(Circle((p["y"], p["x"]), r, facecolor=C["water"],
                                             alpha=0.4, edgecolor=C["water"], linewidth=1.5,
                                             zorder=6)))
            marks.append(ax.plot(p["y"], p["x"], "o", color=C["water"], markersize=4,
                                 zorder=7)[0])
            marks.append(ax.text(p["y"] - r * 0.7, p["x"] + r * 0.7,
                                 f"{p['label']}\n({p['x']:.0f}, {p['y']:.0f})",
                                 color=C["water"], fontsize=8, fontweight="bold", zorder=8))
        for p in s["pads"]:
            marks.append(ax.add_patch(Circle((p["y"], p["x"]), 15, facecolor=C["cyan"],
                                             alpha=0.25, edgecolor=C["cyan"], linewidth=1.5,
                                             zorder=6)))
            marks.append(ax.text(p["y"], p["x"], "H", color=C["cyan"], fontsize=11,
                                 fontweight="bold", ha="center", va="center", zorder=8))
            marks.append(ax.text(p["y"] - 17, p["x"] - 17, f"{p['label']}", color=C["cyan"],
                                 fontsize=8, va="top", zorder=8))
        for p in s["starts"]:
            marks.append(ax.plot(p["y"], p["x"], "s", markersize=12, markerfacecolor="none",
                                 markeredgecolor=C["ok"], markeredgewidth=2, zorder=7)[0])
            marks.append(ax.text(p["y"] - 8, p["x"] - 8, p["label"], color=C["ok"], fontsize=8,
                                 fontweight="bold", va="top", zorder=8))
        self._marks = marks

    def update(self, s, pose=None):
        """s: FlightLog.scene(); pose: (x, y, yaw) of the drone in the path frame, or None."""
        self._draw_paths(s["paths"])
        self._draw_marks(s)
        self.mission.set_data([p[1] for p in s["mission"]], [p[0] for p in s["mission"]])
        pts = [p for path in s["paths"] for p in path[-800:]] + s["mission"] + [(0, 0)]
        pts += [(p["x"], p["y"]) for p in s["starts"] + s["pads"] + s["puddles"]]
        if pose is not None:
            x, y, yaw = pose
            a = math.radians(yaw)
            f = (math.cos(a), math.sin(a))
            l = (-f[1], f[0])
            tri = [(x + f[0] * 18, y + f[1] * 18),
                   (x - f[0] * 10 + l[0] * 10, y - f[1] * 10 + l[1] * 10),
                   (x - f[0] * 4, y - f[1] * 4),
                   (x - f[0] * 10 - l[0] * 10, y - f[1] * 10 - l[1] * 10)]
            self.drone.set_xy([(py, px) for px, py in tri])
            pts.append((x, y))
        self.drone.set_visible(pose is not None)
        # limits: fit everything (or the user's zoom), equal scale, +y on the left
        xs, ys = [p[1] for p in pts], [p[0] for p in pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 200) + 80
        self._fit = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, span)
        cx, cy, span = self._zoom or self._fit
        self.ax.set_xlim(cx + span / 2, cx - span / 2)
        self.ax.set_ylim(cy - span / 2, cy + span / 2)
        self.canvas.draw_idle()

    def save_png(self, path):
        self.fig.savefig(path, dpi=150, facecolor=self.fig.get_facecolor())


def legend(parent):
    """Colour legend as a row of tk labels (no matplotlib legend over the plot)."""
    import tkinter as tk
    row = tk.Frame(parent, bg=C["panel"])
    for sym, color, text in (("━", C["trail"], "gevlogen pad (pijl = richting)"),
                             ("■", C["ok"], "S = startlocatie"),
                             ("●", C["cyan"], "H = landingsplatform"),
                             ("●", C["water"], "P = plas"),
                             ("▲", C["danger"], "drone"),
                             ("┅", C["faint"], "geplande missie")):
        tk.Label(row, text=sym, fg=color, bg=C["panel"]).pack(side=tk.LEFT)
        tk.Label(row, text=text, fg=C["muted"], bg=C["panel"],
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(0, 10))
    return row
