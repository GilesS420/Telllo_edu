"""
Graphical mission control for the Tello EDU.

    python tello_gui.py          # real drone
    python tello_gui.py --sim    # simulator, no drone needed

* Enter waypoints (x, y, z in cm) and edit the list
* 3D grid with the planned path, numbered points with coordinates, the flown
  trail, the current drone position and the puddles found
* Start mission / go to point / take off / land / emergency stop
* Live downward camera image with detections, puddle list and log

Uses DroneApp (drone/app.py), so Jetson missions still work while the GUI is
open (they are drawn in the grid too). Everything runs offline (tkinter +
matplotlib), so it also works on the Tello Wi-Fi. Missions are saved/loaded in
json_flights/.
"""

import argparse
import json
import math
import os
import queue
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib
from PIL import Image, ImageTk

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from drone import config as cfg  # noqa: E402
from drone.navigator import validate_waypoints  # noqa: E402
from drone.app import DroneApp  # noqa: E402

REFRESH_MS = 50

# GUI label -> config value
NAV_MODES = {"Stap (stopt bij elk punt)": "go", "Vloeiend (zonder stoppen)": "rc"}
FINE_MODES = {"Automatisch": "auto", "Op elk punt": "all", "Alleen laatste punt": "last",
              "Uit": "off"}
FLIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_flights")

# Viewing angles (mission frame: x = forward, y = left, z = up)
VIEWS = {
    "3D": (25, 205),          # from behind-right of the start position
    "Boven": (90, 180),       # top view: forward = up, left = left
    "Zijkant": (0, -90),      # side view from the right: forward = right
}


class LogRedirect:
    """Copies everything printed (by any thread) to a queue for the log window."""

    def __init__(self, original):
        self.original = original
        self.queue = queue.Queue()

    def write(self, text):
        self.queue.put(text)
        if self.original:
            self.original.write(text)

    def flush(self):
        if self.original:
            self.original.flush()


class TelloGUI:
    def __init__(self, root, app, log):
        self.root = root
        self.app = app
        self.log = log
        self.waypoints = []          # editable draft path [(x, y, z), ...]
        self.trail = []              # flown positions
        self.trail_mission = None
        self.mission_count = 0
        self._labels = []
        self._label_key = None
        self._limits_key = None
        self._puddle_key = None
        self._camera_photo = None

        root.title("Tello EDU – missiebesturing")
        root.geometry("1450x880")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_left(ttk.Frame(root, padding=8))
        self._build_right(ttk.Frame(root, padding=(0, 8, 8, 8)))
        root.bind("<Return>", lambda e: self.add_waypoint())
        self.refresh()

    # ================================================================ layout
    def _build_left(self, left):
        left.pack(side=tk.LEFT, fill=tk.Y)

        # --- status
        box = ttk.LabelFrame(left, text="Status", padding=6)
        box.pack(fill=tk.X)
        self.status_vars = {}
        for key, label in (("state", "Toestand"), ("battery", "Batterij"), ("pos", "Positie"),
                           ("heading", "Richting"), ("odo", "Odometrie"),
                           ("mission", "Missie"), ("puddles", "Plassen")):
            row = ttk.Frame(box)
            row.pack(fill=tk.X)
            ttk.Label(row, text=f"{label}:", width=10).pack(side=tk.LEFT)
            var = tk.StringVar(value="–")
            ttk.Label(row, textvariable=var, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
            self.status_vars[key] = var

        # --- waypoint entry
        box = ttk.LabelFrame(left, text="Waypoint (cm)   x = vooruit, y = links, z = hoogte",
                             padding=6)
        box.pack(fill=tk.X, pady=(4, 0))
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        self.entries = {}
        for name, default in (("x", "100"), ("y", "0"), ("z", "80")):
            ttk.Label(row, text=name).pack(side=tk.LEFT)
            e = ttk.Entry(row, width=7, justify=tk.RIGHT)
            e.insert(0, default)
            e.pack(side=tk.LEFT, padx=(2, 8))
            self.entries[name] = e
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(row, text="➕ Toevoegen (Enter)", command=self.add_waypoint).pack(side=tk.LEFT)
        ttk.Button(row, text="Bijwerken", command=self.update_waypoint).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Huidige positie", command=self.fill_current_pos).pack(side=tk.LEFT)

        # --- waypoint list
        box = ttk.LabelFrame(left, text="Pad", padding=6)
        box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.tree = ttk.Treeview(box, columns=("n", "x", "y", "z"), show="headings", height=5)
        for col, text, w in (("n", "#", 40), ("x", "x", 70), ("y", "y", 70), ("z", "z", 70)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=tk.E)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.load_selected_into_entries())
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(6, 0))
        for text, cmd in (("▲", lambda: self.move_selected(-1)), ("▼", lambda: self.move_selected(1)),
                          ("Verwijder", self.delete_selected), ("Wis alles", self.clear_waypoints),
                          ("Opslaan", self.save_mission), ("Laden", self.load_mission)):
            ttk.Button(row, text=text, command=cmd, width=len(text) + 2).pack(side=tk.LEFT, padx=1)

        # --- mission settings
        box = ttk.LabelFrame(left, text="Missie", padding=6)
        box.pack(fill=tk.X, pady=(4, 0))
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Snelheid (cm/s)").pack(side=tk.LEFT)
        self.speed = tk.IntVar(value=cfg.DEFAULT_SPEED)
        ttk.Label(row, textvariable=self.speed, width=4).pack(side=tk.RIGHT)
        ttk.Scale(row, from_=10, to=100, variable=self.speed,
                  command=lambda v: self.speed.set(int(float(v)))).pack(side=tk.LEFT, fill=tk.X,
                                                                        expand=True, padx=6)
        for label, attr, options, default in (
                ("Vliegmodus", "nav_mode", NAV_MODES, cfg.NAV_MODE),
                ("Nauwkeurig positioneren", "fine", FINE_MODES, cfg.FINE_POSITION)):
            row = ttk.Frame(box)
            row.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            names = list(options)
            var = tk.StringVar(value=next(k for k, v in options.items() if v == default))
            ttk.Combobox(row, textvariable=var, values=names, state="readonly",
                         width=22).pack(side=tk.RIGHT)
            setattr(self, attr, var)
        self.land_at_end = tk.BooleanVar(value=cfg.LAND_AT_END)
        ttk.Checkbutton(box, text="Landen na het laatste punt",
                        variable=self.land_at_end).pack(anchor=tk.W, pady=(4, 0))

        # --- camera mode
        box = ttk.LabelFrame(left, text="Camera", padding=6)
        box.pack(fill=tk.X, pady=(8, 0))
        self.camera_mode_var = tk.StringVar(value="onder")
        ttk.Label(box, text="Actieve camera:").pack(anchor=tk.W)
        ttk.Label(box, textvariable=self.camera_mode_var,
                  font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Voor", command=lambda: self.set_camera_mode(False)).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(row, text="Onder", command=lambda: self.set_camera_mode(True)).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # --- flight buttons
        box = ttk.LabelFrame(left, text="Vliegen", padding=6)
        box.pack(fill=tk.X, pady=(4, 0))
        grid = ttk.Frame(box)
        grid.pack(fill=tk.X)
        buttons = (
            ("▶ Start missie", self.start_mission, "#2e7d32", 0, 0),
            ("Ga naar geselecteerd punt", self.goto_selected, "#1565c0", 0, 1),
            ("Opstijgen", lambda: self.app.submit({"type": "takeoff"}), "#555555", 1, 0),
            ("Landen (L)", self.land, "#ef6c00", 1, 1),
        )
        for text, cmd, color, r, c in buttons:
            tk.Button(grid, text=text, command=cmd, bg=color, fg="white", activebackground=color,
                      relief=tk.FLAT, pady=4).grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        tk.Button(box, text="⚠ NOODSTOP – motoren uit (X)", command=self.emergency, bg="#c62828",
                  fg="white", activebackground="#b71c1c", relief=tk.FLAT, pady=4,
                  font=("TkDefaultFont", 10, "bold")).pack(fill=tk.X, padx=2, pady=(4, 2))
        self.record_btn = ttk.Button(box, text="🎥 Opnemen (dataset)", command=self.toggle_record)
        self.record_btn.pack(fill=tk.X, padx=2, pady=(4, 0))
        self.root.bind("<KeyPress-l>", lambda e: self._key(self.land, e))
        self.root.bind("<KeyPress-x>", lambda e: self._key(self.emergency, e))

    def _build_right(self, right):
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- 3D plot
        bar = ttk.Frame(right)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Aanzicht:").pack(side=tk.LEFT)
        for name in VIEWS:
            ttk.Button(bar, text=name, command=lambda n=name: self.set_view(n)).pack(side=tk.LEFT,
                                                                                    padx=2)
        ttk.Button(bar, text="Wis spoor", command=self.clear_trail).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Label(bar, text="(sleep met de muis om te draaien)", foreground="#777").pack(
            side=tk.LEFT, padx=8)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._init_plot()

        # --- bottom: camera, puddles, log
        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.X, pady=(6, 0))
        cam = ttk.LabelFrame(bottom, text="Onderste camera", padding=4)
        cam.pack(side=tk.LEFT)
        self.cam_label = tk.Label(cam, width=320, height=240, bg="black",
                      text="Camera wordt geladen...",
                      fg="white", wraplength=300, justify=tk.CENTER)
        self.cam_label.pack()
        self.camera_btn = ttk.Button(cam, text="Camera: onder", command=self.toggle_camera_mode)
        self.camera_btn.pack(fill=tk.X, pady=(4, 0))

        box = ttk.LabelFrame(bottom, text="Gevonden plassen", padding=4)
        box.pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.puddle_tree = ttk.Treeview(box, columns=("id", "x", "y", "area"), show="headings",
                                        height=10)
        for col, text, w in (("id", "#", 35), ("x", "x", 60), ("y", "y", 60), ("area", "cm²", 60)):
            self.puddle_tree.heading(col, text=text)
            self.puddle_tree.column(col, width=w, anchor=tk.E)
        self.puddle_tree.pack(fill=tk.Y, expand=True)

        box = ttk.LabelFrame(bottom, text="Log", padding=4)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(box, height=14, width=40, state=tk.DISABLED, wrap=tk.WORD,
                                font=("TkFixedFont", 9))
        scroll = ttk.Scrollbar(box, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ================================================================== plot
    def _init_plot(self):
        ax = self.ax
        ax.set_xlabel("x vooruit (cm)")
        ax.set_ylabel("y links (cm)")
        ax.set_zlabel("z hoogte (cm)")
        ax.plot([0], [0], [0], marker="s", color="black", markersize=8, linestyle="none")
        ax.text(0, 0, 0, "  start", color="black")
        self.draft_line, = ax.plot([], [], [], "o--", color="#ef6c00", markersize=7,
                                   label="ingevoerd pad")
        self.connector, = ax.plot([], [], [], ":", color="#ef6c00", alpha=0.6)
        self.mission_line, = ax.plot([], [], [], "o-", color="#1565c0", linewidth=2,
                                     markersize=6, label="actieve missie")
        self.reached_pts, = ax.plot([], [], [], "o", color="#2e7d32", markersize=9,
                                    linestyle="none", label="bereikt")
        self.trail_line, = ax.plot([], [], [], "-", color="#c62828", linewidth=1.5,
                                   label="gevlogen")
        self.drop_line, = ax.plot([], [], [], "--", color="#555555", linewidth=1)
        self.heading_line, = ax.plot([], [], [], "-", color="#c62828", linewidth=3)
        self.drone_pt, = ax.plot([], [], [], marker="X", color="#c62828", markersize=14,
                                 linestyle="none", label="drone")
        self.puddle_pts, = ax.plot([], [], [], "o", color="#00acc1", markersize=12, alpha=0.7,
                                   linestyle="none", label="plas")
        ax.legend(loc="upper left", fontsize=8)
        self.set_view("3D", draw=False)
        self._update_limits(force=True)

    def set_view(self, name, draw=True):
        elev, azim = VIEWS[name]
        self.ax.view_init(elev=elev, azim=azim)
        # hide the axis that points at the viewer in the 2D views (it only clutters)
        for axis, label, hidden in ((self.ax.zaxis, "z hoogte (cm)", name == "Boven"),
                                    (self.ax.yaxis, "y links (cm)", name == "Zijkant")):
            axis.set_label_text("" if hidden else label)
            axis.set_major_formatter(NullFormatter() if hidden else ScalarFormatter())
        if draw:
            self.canvas.draw_idle()

    def _update_limits(self, points=(), force=False):
        xs = [0.0] + [p[0] for p in points]
        ys = [0.0] + [p[1] for p in points]
        zs = [0.0] + [p[2] for p in points]
        margin = 40
        lo_x, hi_x = min(xs) - margin, max(xs) + margin
        lo_y, hi_y = min(ys) - margin, max(ys) + margin
        hi_z = max(max(zs) + margin, 150)
        # equal scale on x and y, at least 2 x 2 m
        span = max(hi_x - lo_x, hi_y - lo_y, 200)
        cx, cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
        key = tuple(int(round(v / 50)) for v in (cx, cy, span, hi_z))
        if key == self._limits_key and not force:
            return
        self._limits_key = key
        self.ax.set_xlim(cx - span / 2, cx + span / 2)
        self.ax.set_ylim(cy - span / 2, cy + span / 2)
        self.ax.set_zlim(0, hi_z)
        self.ax.set_box_aspect((1, 1, max(hi_z / span, 0.25)))

    @staticmethod
    def _set(line, pts):
        if pts:
            xs, ys, zs = zip(*pts)
            line.set_data_3d(list(xs), list(ys), list(zs))
        else:
            line.set_data_3d([], [], [])

    def update_plot(self, pose, puddles):
        x, y, z, _ = pose
        mission = list(self.app.current_waypoints)
        self._set(self.draft_line, self.waypoints)
        # dotted line "from here to point 1", only while planning
        show_connector = self.waypoints and not self.app.airborne
        self._set(self.connector, [(x, y, z), self.waypoints[0]] if show_connector else [])
        self._set(self.mission_line, mission)
        self._set(self.reached_pts, mission[:self.app.waypoints_reached])
        self._set(self.trail_line, self.trail)
        self._set(self.drone_pt, [(x, y, z)])
        self._set(self.drop_line, [(x, y, 0), (x, y, z)] if z > 0 else [])
        a = math.radians(pose[3])  # nose direction
        self._set(self.heading_line, [(x, y, z), (x + 30 * math.cos(a), y + 30 * math.sin(a), z)])
        self._set(self.puddle_pts, [(p["x"], p["y"], 0) for p in puddles])

        # Labels: numbered points with coordinates + puddle ids (only rebuilt on change)
        labels = []
        for i, p in enumerate(self.waypoints):
            labels.append((p, f" {i + 1} ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})", "#ef6c00"))
        if mission != self.waypoints:
            for i, p in enumerate(mission):
                labels.append((p, f" M{i + 1} ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})", "#1565c0"))
        for p in puddles:
            labels.append(((p["x"], p["y"], 0), f" plas #{p['id']}", "#00838f"))
        key = tuple((tuple(round(v) for v in pt), t) for pt, t, _ in labels)
        if key != self._label_key:
            self._label_key = key
            for t in self._labels:
                t.remove()
            self._labels = [self.ax.text(*pt, text, color=c, fontsize=8) for pt, text, c in labels]

        self._update_limits(self.waypoints + mission + self.trail + [(x, y, z)]
                            + [(p["x"], p["y"], 0) for p in puddles])
        self.canvas.draw_idle()

    # ============================================================= refresh
    def refresh(self):
        try:
            self._refresh()
        finally:
            self.root.after(REFRESH_MS, self.refresh)

    def _refresh(self):
        app = self.app
        # log
        chunks = []
        while not self.log.queue.empty():
            chunks.append(self.log.queue.get_nowait())
        if chunks:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, "".join(chunks))
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        # status
        pose = app.pose.get()
        x, y, z, yaw = pose
        try:
            battery = app.tello.get_battery()
        except Exception:
            battery = "?"
        puddles = app.tracker.confirmed()
        self.status_vars["state"].set(app.state + ("  ● REC" if app.recording else ""))
        self.status_vars["battery"].set(f"{battery}%")
        self.status_vars["pos"].set(f"x={x:.0f}  y={y:.0f}  z={z:.0f} cm")
        self.status_vars["heading"].set(f"{yaw:+.0f}°   (vasthouden op {app.pose.reference_yaw:+.0f}°)")
        if not app.airborne:
            self.status_vars["odo"].set("– (op de grond)")
        elif app.pose.measured:
            self.status_vars["odo"].set(f"✔ meet de vloer (kwaliteit {app.pose.quality:.2f})")
        else:
            self.status_vars["odo"].set("✖ geen beeld, rekent met commando's")
        # new mission: reset the flown trail
        if app.mission_id != self.trail_mission:
            self.trail_mission = app.mission_id
            self.trail = []
        if app.mission_id is None:
            self.status_vars["mission"].set("–")
        else:
            self.status_vars["mission"].set(f"{app.mission_id}  ({app.waypoints_reached}/"
                                            f"{len(app.current_waypoints)} punten)")
        self.status_vars["puddles"].set(str(len(puddles)))

        if app.airborne and (not self.trail or
                             max(abs(a - b) for a, b in zip(self.trail[-1], (x, y, z))) > 2):
            self.trail.append((x, y, z))

        self.update_plot(pose, puddles)
        self.update_puddles(puddles)
        self.update_camera()
        self.camera_mode_var.set("onder" if self.app.downvision_enabled else "voor")

    def update_puddles(self, puddles):
        key = tuple((p["id"], p["x"], p["y"]) for p in puddles)
        if key == self._puddle_key:
            return
        self._puddle_key = key
        self.puddle_tree.delete(*self.puddle_tree.get_children())
        for p in puddles:
            self.puddle_tree.insert("", tk.END, values=(p["id"], f"{p['x']:.0f}", f"{p['y']:.0f}",
                                                        f"{p['area_cm2']:.0f}"))

    def update_camera(self):
        frame = self.app.frame_reader.frame if hasattr(self.app, "frame_reader") else None
        self.camera_btn.configure(text="Camera: onder" if self.app.downvision_enabled
                                  else "Camera: voor")
        self.camera_mode_var.set("onder" if self.app.downvision_enabled else "voor")
        if frame is None:
            self.cam_label.configure(text="Geen camerabeeld", image="")
            return

        vis = frame.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        elif not cfg.FRAME_IS_RGB:
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        vis = cv2.resize(vis, (720, 480))
        battery = self.app.tello.get_battery()
        cv2.putText(vis, f"Battery: {battery}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(vis, f"Downvision: {'ON' if self.app.downvision_enabled else 'OFF'}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, "GUI Camera", (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        img = Image.fromarray(vis)
        self._camera_photo = ImageTk.PhotoImage(img)
        self.cam_label.configure(image=self._camera_photo, text="")

    def toggle_camera_mode(self):
        try:
            self.app.toggle_downvision()
        except Exception as e:
            messagebox.showerror("Camera", str(e))
            return
        self.camera_btn.configure(text="Camera: onder" if self.app.downvision_enabled
                                  else "Camera: voor")

    def set_camera_mode(self, use_downvision):
        try:
            self.app.set_downvision(use_downvision)
        except Exception as e:
            messagebox.showerror("Camera", str(e))
            return
        self.camera_btn.configure(text="Camera: onder" if self.app.downvision_enabled
                                  else "Camera: voor")
        self.camera_mode_var.set("onder" if self.app.downvision_enabled else "voor")

    # ========================================================== waypoints
    def _read_entries(self):
        try:
            point = tuple(float(self.entries[k].get().replace(",", ".")) for k in "xyz")
        except ValueError:
            messagebox.showerror("Ongeldige waarde", "x, y en z moeten getallen zijn (cm).")
            return None
        try:
            validate_waypoints([point])
        except ValueError as e:
            messagebox.showerror("Buiten de geofence", str(e).replace("waypoint 0: ", ""))
            return None
        return point

    def _selected_index(self):
        sel = self.tree.selection()
        return self.tree.index(sel[0]) if sel else None

    def _refresh_tree(self, select=None):
        self.tree.delete(*self.tree.get_children())
        for i, (x, y, z) in enumerate(self.waypoints):
            self.tree.insert("", tk.END, values=(i + 1, f"{x:.0f}", f"{y:.0f}", f"{z:.0f}"))
        if select is not None and 0 <= select < len(self.waypoints):
            item = self.tree.get_children()[select]
            self.tree.selection_set(item)
            self.tree.see(item)

    def add_waypoint(self):
        point = self._read_entries()
        if point:
            self.waypoints.append(point)
            self._refresh_tree()

    def update_waypoint(self):
        i = self._selected_index()
        point = self._read_entries()
        if i is not None and point:
            self.waypoints[i] = point
            self._refresh_tree(select=i)

    def load_selected_into_entries(self):
        i = self._selected_index()
        if i is None:
            return
        for k, v in zip("xyz", self.waypoints[i]):
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, f"{v:.0f}")

    def fill_current_pos(self):
        for k, v in zip("xyz", self.app.pose.get()[:3]):
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, f"{v:.0f}")

    def move_selected(self, delta):
        i = self._selected_index()
        if i is None or not 0 <= i + delta < len(self.waypoints):
            return
        wp = self.waypoints
        wp[i], wp[i + delta] = wp[i + delta], wp[i]
        self._refresh_tree(select=i + delta)

    def delete_selected(self):
        i = self._selected_index()
        if i is not None:
            del self.waypoints[i]
            self._refresh_tree(select=min(i, len(self.waypoints) - 1))

    def clear_waypoints(self):
        if self.waypoints and messagebox.askyesno("Wis alles", "Alle waypoints wissen?"):
            self.waypoints = []
            self._refresh_tree()

    def save_mission(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", initialdir=FLIGHTS_DIR,
                                            filetypes=[("Missie", "*.json")])
        if path:
            with open(path, "w") as f:
                json.dump(self._mission(self.waypoints, self.land_at_end.get()), f, indent=2)

    def load_mission(self):
        path = filedialog.askopenfilename(initialdir=FLIGHTS_DIR, filetypes=[("Missie", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.waypoints = validate_waypoints(data["waypoints"] if isinstance(data, dict) else data)
            if isinstance(data, dict):
                self.speed.set(int(data.get("speed", self.speed.get())))
                self.land_at_end.set(bool(data.get("land_at_end", self.land_at_end.get())))
                for attr, options in (("nav_mode", NAV_MODES), ("fine", FINE_MODES)):
                    label = next((k for k, v in options.items() if v == data.get(attr)), None)
                    if label:
                        getattr(self, attr).set(label)
        except (ValueError, KeyError, OSError) as e:
            messagebox.showerror("Laden mislukt", str(e))
            return
        self._refresh_tree()

    # ============================================================== flying
    def _mission(self, waypoints, land_at_end):
        self.mission_count += 1
        return {"type": "mission", "id": f"gui-{self.mission_count}", "speed": self.speed.get(),
                "land_at_end": land_at_end, "nav_mode": NAV_MODES[self.nav_mode.get()],
                "fine": FINE_MODES[self.fine.get()], "waypoints": [list(p) for p in waypoints]}

    def start_mission(self):
        if not self.waypoints:
            messagebox.showinfo("Geen pad", "Voeg eerst waypoints toe.")
            return
        self.app.submit(self._mission(self.waypoints, self.land_at_end.get()))

    def goto_selected(self):
        i = self._selected_index()
        if i is None:
            messagebox.showinfo("Geen punt", "Selecteer eerst een punt in de lijst.")
            return
        self.app.submit(self._mission([self.waypoints[i]], land_at_end=False))

    def land(self):
        self.app.submit({"type": "land"})

    def emergency(self):
        self.app.emergency_stop()

    def toggle_record(self):
        self.app.toggle_recording()
        self.record_btn.configure(text="⏹ Stop opnemen" if self.app.recording
                                  else "🎥 Opnemen (dataset)")

    def clear_trail(self):
        self.trail = []

    def _key(self, action, event):
        # don't trigger flight keys while typing coordinates
        if not isinstance(event.widget, (tk.Entry, ttk.Entry)):
            action()

    def on_close(self):
        if self.app.airborne and not messagebox.askyesno(
                "Afsluiten", "De drone vliegt nog. Landen en afsluiten?"):
            return
        self.root.title("Landen en afsluiten...")
        self.root.update()
        self.app.shutdown()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="use the simulator instead of a drone")
    parser.add_argument("--record", metavar="DIR", help="save downward frames to DIR (dataset)")
    args = parser.parse_args()

    log = LogRedirect(sys.stdout)
    sys.stdout = log

    root = tk.Tk()
    if args.sim:
        from drone.sim import FakeTello
        tello = FakeTello()
    else:
        from djitellopy import Tello
        Tello.RESPONSE_TIMEOUT = cfg.RESPONSE_TIMEOUT
        tello = Tello()

    app = DroneApp(tello, record_dir=args.record)
    try:
        app.start()
    except Exception as e:
        root.withdraw()
        messagebox.showerror("Geen verbinding met de drone",
                             f"{e}\n\nZit de laptop op de Wi-Fi van de Tello?\n"
                             "Testen zonder drone: python tello_gui.py --sim")
        app.link.close()
        return
    TelloGUI(root, app, log)
    root.mainloop()


if __name__ == "__main__":
    main()
