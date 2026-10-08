"""
Graphical mission control for the Tello EDU.

    python tello_gui.py          # real drone
    python tello_gui.py --sim    # simulator, no drone needed

Layout
    top bar      state, battery, flight time, Jetson link, odometry, messages
    left         plan the path: waypoints, raster pattern, mission settings, fly
    middle       tabs: 3D view | map & terrain (click to plan) | sensor graphs
    right        camera, instruments (horizon, compass, height), sensor values,
                 puddles found

Uses DroneApp (drone/app.py), so Jetson missions keep working while the GUI is
open (they are drawn too). Everything runs offline (tkinter + matplotlib), so
it also works on the Tello Wi-Fi. Missions are saved/loaded in json_flights/.

Keys: Enter add point, Del remove point, Ctrl+Z undo, Ctrl+S save, Ctrl+O load,
Ctrl+1/2/3 switch tab, L land, X emergency stop (motors off).
"""

import argparse
import json
import math
import os
import queue
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib
from PIL import Image, ImageTk

matplotlib.use("TkAgg")

from drone import config as cfg  # noqa: E402
from drone.app import DroneApp  # noqa: E402
from drone.navigator import validate_waypoints  # noqa: E402
from gui import theme  # noqa: E402
from gui.map3d import VIEWS, Map3D, legend  # noqa: E402
from gui.mapview import MapView  # noqa: E402
from gui.theme import C, F, tk_button  # noqa: E402
from gui.widgets import (AttitudeIndicator, BatteryBar, Compass, HeightGauge,  # noqa: E402
                         LineChart, Segmented, StatTile, Toast, card)

TICK_MS = 33                    # instruments, camera, top bar (~30 fps)
PLOT_3D_S = 0.10                # matplotlib views are slower: redraw less often
PLOT_MAP_S = 0.15
CHART_S = 0.25

# GUI label -> config value
NAV_MODES = {"Stap (stopt bij elk punt)": "go", "Vloeiend (zonder stoppen)": "rc"}
FINE_MODES = {"Automatisch": "auto", "Op elk punt": "all", "Alleen laatste punt": "last",
              "Uit": "off"}
STATES = {"idle": ("AAN DE GROND", C["faint"]), "taking_off": ("OPSTIJGEN", C["warn"]),
          "flying": ("VLIEGT", C["ok"]), "landing": ("LANDEN", C["warn"]),
          "error": ("FOUT", C["danger"])}
MAP_HELP = "klik = punt · slepen = verplaatsen · rechtsklik = wissen · scroll = zoom"
WINDOWS = {"30 s": 30, "1 min": 60, "5 min": 300}
FLIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_flights")


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


def fmt(v, digits=0, unit=""):
    return "–" if v is None else f"{v:.{digits}f}{unit}"


class TelloGUI:
    def __init__(self, root, app, log, sim=False):
        self.root, self.app, self.log, self.sim = root, app, log, sim
        self.waypoints = []          # editable draft path [(x, y, z), ...]
        self.history = []            # undo stack of earlier waypoint lists
        self.sel = None              # selected waypoint index
        self.trail = []              # flown positions
        self.trail_mission = None
        self.mission_count = 0
        self.show = None             # displayed pose, glides towards the real pose
        self.takeoff_time = None
        self._dragging = False
        self._next = {"3d": 0, "map": 0, "chart": 0, "side": 0}
        self._tick_n = 0
        self._puddle_key = None
        self._camera_photo = None
        self._was_airborne = False
        self._summary = None

        theme.apply(root)
        root.title("Tello EDU – missiebesturing")
        root.geometry("1500x930")
        root.minsize(1240, 760)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_topbar()
        body = ttk.Frame(root, style="Bg.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style="Bg.TFrame", width=330)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        left.pack_propagate(False)
        center = ttk.Frame(body, style="Bg.TFrame")
        center.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(body, style="Bg.TFrame", width=384)
        right.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        right.pack_propagate(False)
        self._build_left(left)
        self._build_center(center)
        self._build_right(right)
        self._bind_keys()
        self._validate_entries()
        self.toast.show("Klik op de kaart om punten toe te voegen, of vul x, y, z in.", "info", 8)
        self.tick()

    # ================================================================ layout
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=C["bg"], height=48)
        bar.pack(fill=tk.X, padx=8, pady=6)
        tk.Label(bar, text="TELLO EDU", bg=C["bg"], fg=C["text"], font=F["huge"]).pack(
            side=tk.LEFT)
        tk.Label(bar, text="missiebesturing", bg=C["bg"], fg=C["muted"], font=F["base"]).pack(
            side=tk.LEFT, padx=(6, 14), pady=(6, 0))
        if self.sim:
            tk.Label(bar, text=" SIMULATOR ", bg=C["s4"], fg="#140a26", font=F["title"]).pack(
                side=tk.LEFT, padx=(0, 10))
        self.state_pill = tk.Label(bar, text="", bg=C["faint"], fg="#0b0f14", font=F["title"],
                                   padx=10, pady=3)
        self.state_pill.pack(side=tk.LEFT, padx=(0, 12))
        self.battery = BatteryBar(bar, bg=C["bg"])
        self.battery.pack(side=tk.LEFT, padx=(0, 14))
        self.top_vars = {}
        for key in ("time", "height", "odo", "jetson"):
            lbl = tk.Label(bar, text="", bg=C["bg"], fg=C["muted"], font=F["base"])
            lbl.pack(side=tk.LEFT, padx=(0, 16))
            self.top_vars[key] = lbl
        self.rec_pill = tk.Label(bar, text=" ● REC ", bg=C["danger"], fg="white",
                                 font=F["title"])
        msg = tk.Label(bar, text="", bg=C["bg"], fg=C["accent"], font=F["base"], anchor=tk.E)
        msg.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.toast = Toast(msg)

    def _build_left(self, left):
        # --- flight buttons (packed first at the bottom so they never scroll away)
        outer, _, box = card(left, "Vliegen")
        outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        grid = ttk.Frame(box, style="Card.TFrame")
        grid.pack(fill=tk.X)
        for text, cmd, color, r, c in (
                ("▶  Start missie", self.start_mission, "#238453", 0, 0),
                ("Ga naar punt", self.goto_selected, "#1f6fb5", 0, 1),
                ("Opstijgen", lambda: self.app.submit({"type": "takeoff"}), "#3a4757", 1, 0),
                ("Landen  (L)", self.land, "#c2701b", 1, 1)):
            tk_button(grid, text, cmd, color).grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        tk_button(grid, "Landen op de H  (H)", self.helipad_land, "#2f7d6d").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        grid.columnconfigure(0, weight=1, uniform="b")
        grid.columnconfigure(1, weight=1, uniform="b")
        tk_button(box, "⚠  NOODSTOP – motoren uit  (X)", self.emergency, "#c62f37",
                  font=F["bold"]).pack(fill=tk.X, padx=2, pady=(4, 0))

        # --- mission settings
        outer, _, box = card(left, "Missie-instellingen")
        outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X)
        ttk.Label(row, text="Snelheid", style="Card.TLabel").pack(side=tk.LEFT)
        self.speed = tk.IntVar(value=cfg.DEFAULT_SPEED)
        self.speed_lbl = ttk.Label(row, text="", style="Card.TLabel", width=9, anchor=tk.E)
        self.speed_lbl.pack(side=tk.RIGHT)
        tk.Scale(row, from_=10, to=100, variable=self.speed, orient=tk.HORIZONTAL,
                 showvalue=False, bg=C["accent"], troughcolor=C["card2"], bd=0,
                 highlightthickness=0, sliderrelief=tk.FLAT, sliderlength=14, width=10,
                 activebackground="#6cc2ff",
                 command=lambda v: self._speed_changed(v)).pack(side=tk.LEFT, fill=tk.X,
                                                                expand=True, padx=8)
        self._speed_changed(self.speed.get())
        for label, attr, options, default in (
                ("Vliegmodus", "nav_mode", NAV_MODES, cfg.NAV_MODE),
                ("Nauwkeurig positioneren", "fine", FINE_MODES, cfg.FINE_POSITION)):
            row = ttk.Frame(box, style="Card.TFrame")
            row.pack(fill=tk.X, pady=(6, 0))
            ttk.Label(row, text=label, style="Card.TLabel").pack(side=tk.LEFT)
            var = tk.StringVar(value=next(k for k, v in options.items() if v == default))
            cb = ttk.Combobox(row, textvariable=var, values=list(options), state="readonly",
                              width=21)
            cb.pack(side=tk.RIGHT)
            cb.bind("<<ComboboxSelected>>", lambda e: self._path_stats())
            setattr(self, attr, var)
        self.land_at_end = tk.BooleanVar(value=cfg.LAND_AT_END)
        ttk.Checkbutton(box, text="Landen na het laatste punt",
                        variable=self.land_at_end).pack(anchor=tk.W, pady=(6, 0))
        self.level = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Zelfde hoogte houden (vloer niet volgen)",
                        variable=self.level).pack(anchor=tk.W, pady=(4, 0))
        self.helipad_search = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Landen op gevonden H (anders 1e punt)",
                        variable=self.helipad_search).pack(anchor=tk.W, pady=(4, 0))

        # --- waypoint entry
        outer, _, box = card(left, "Waypoint (cm)")
        outer.pack(fill=tk.X)
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X)
        self.entries = {}
        for name, default, hint in (("x", "100", "vooruit"), ("y", "0", "links"),
                                    ("z", "80", "hoogte")):
            col = ttk.Frame(row, style="Card.TFrame")
            col.pack(side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True)
            ttk.Label(col, text=f"{name}  {hint}", style="Muted.TLabel").pack(anchor=tk.W)
            lo, hi = cfg.GEOFENCE[name]
            e = ttk.Spinbox(col, from_=lo, to=hi, increment=10, width=6, justify=tk.RIGHT,
                            command=self._validate_entries)
            e.set(default)
            e.pack(fill=tk.X)
            e.bind("<KeyRelease>", lambda ev: self._validate_entries())
            self.entries[name] = e
        self.entry_hint = ttk.Label(box, text="", style="Error.TLabel")
        self.entry_hint.pack(anchor=tk.W, pady=(2, 0))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(2, 0))
        for c, (text, cmd) in enumerate((("＋ Toevoegen", self.add_waypoint),
                                         ("Bijwerken", self.update_waypoint),
                                         ("Drone-positie", self.fill_current_pos))):
            ttk.Button(row, text=text, command=cmd, style="Small.TButton").grid(
                row=0, column=c, sticky="ew", padx=(0 if c == 0 else 3, 0))
            row.columnconfigure(c, weight=1, uniform="w")

        # --- waypoint list
        outer, header, box = card(left, "Pad")
        outer.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.path_stats = ttk.Label(header, text="", style="Muted.TLabel")
        self.path_stats.pack(side=tk.RIGHT)
        self.tree = ttk.Treeview(box, columns=("n", "x", "y", "z"), show="headings", height=4,
                                 selectmode="browse")
        for col, text, w in (("n", "#", 36), ("x", "x", 70), ("y", "y", 70), ("z", "z", 70)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=tk.E)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._tree_selected())
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(6, 0))
        for text, cmd, tip in (("▲", lambda: self.move_selected(-1), None),
                               ("▼", lambda: self.move_selected(1), None),
                               ("✕", self.delete_selected, None),
                               ("↶", self.undo, None),
                               ("Raster…", self.pattern_dialog, None),
                               ("Wis", self.clear_waypoints, None)):
            ttk.Button(row, text=text, command=cmd, style="Small.TButton",
                       width=len(text) + 1).pack(side=tk.LEFT, padx=(0, 2))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="Opslaan", command=self.save_mission, style="Small.TButton").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(row, text="Laden", command=self.load_mission, style="Small.TButton").pack(
            side=tk.LEFT, fill=tk.X, expand=True)

    def _build_center(self, center):
        self.tabs = ttk.Notebook(center)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        # --- 3D
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  3D-weergave  ")
        bar = ttk.Frame(tab, padding=(8, 6))
        bar.pack(fill=tk.X)
        self.view_seg = Segmented(bar, list(VIEWS), self._set_view, "3D", style="TFrame")
        self.view_seg.pack(side=tk.LEFT)
        ttk.Button(bar, text="Zoom passend", style="Small.TButton",
                   command=lambda: self.map3d.reset_zoom()).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(bar, text="Wis spoor", style="Small.TButton",
                   command=self.clear_trail).pack(side=tk.LEFT, padx=2)
        self.terrain3d = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Terrein", variable=self.terrain3d, style="TCheckbutton",
                        command=self._terrain3d_changed).pack(side=tk.LEFT, padx=8)
        self.map3d = Map3D(tab)
        self.map3d.widget.pack(fill=tk.BOTH, expand=True)
        legend(tab).pack(fill=tk.X, padx=8, pady=(0, 4))

        # --- map & terrain
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  Kaart & terrein  ")
        bar = ttk.Frame(tab, padding=(8, 6))
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Passend", style="Small.TButton",
                   command=lambda: self.mapview.fit()).pack(side=tk.LEFT)
        ttk.Button(bar, text="Herijken (grond hier = 0)", style="Small.TButton",
                   command=self.rereference).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Button(bar, text="Wis terrein", style="Small.TButton",
                   command=self.clear_terrain).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Terrein → CSV", style="Small.TButton",
                   command=self.export_terrain).pack(side=tk.LEFT, padx=2)
        split = ttk.Frame(tab)
        split.pack(fill=tk.BOTH, expand=True)
        info_outer, _, info = card(split, "Terreinanalyse")
        info_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))
        self.terrain_info = tk.Label(info, text="", bg=C["card"], fg=C["text"],
                                     font=F["small"], justify=tk.LEFT, anchor=tk.NW,
                                     width=26, wraplength=200)
        self.terrain_info.pack(fill=tk.BOTH, expand=True)
        self.mapview = MapView(split, self._map_add, self._map_move, self._map_delete,
                               self._map_select, self._map_hover)
        self.mapview.widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.mapview.set_hover_text(MAP_HELP)

        # --- sensors
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="  Sensoren  ")
        bar = ttk.Frame(tab, padding=(8, 6))
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Venster", foreground=C["muted"]).pack(side=tk.LEFT, padx=(0, 6))
        self.window_seg = Segmented(bar, list(WINDOWS), lambda v: None, "1 min", style="TFrame")
        self.window_seg.pack(side=tk.LEFT)
        ttk.Button(bar, text="Telemetrie → CSV", style="Small.TButton",
                   command=self.export_telemetry).pack(side=tk.RIGHT)
        grid = ttk.Frame(tab, padding=(8, 0, 8, 8))
        grid.pack(fill=tk.BOTH, expand=True)
        self.charts = [
            LineChart(grid, "Hoogte", "cm", [("tof", "ToF", C["s1"]), ("alt", "baro", C["s2"]),
                                             ("z", "positie z", C["s3"])]),
            LineChart(grid, "Terrein onder de drone (baro − ToF)", "cm",
                      [("ground", "grond", C["terrain_hi"])]),
            LineChart(grid, "Snelheid (drone-assen)", "cm/s",
                      [("vx", "vx", C["s1"]), ("vy", "vy", C["s2"]), ("vz", "vz", C["s3"])],
                      symmetric=True),
            LineChart(grid, "Houding", "°", [("pitch", "pitch", C["s1"]),
                                             ("roll", "roll", C["s2"])], symmetric=True),
            LineChart(grid, "Versnelling", "g", [("ax", "ax", C["s1"]), ("ay", "ay", C["s2"]),
                                                 ("az", "az", C["s3"])]),
            LineChart(grid, "Batterij / temperatuur", "% / °C",
                      [("battery", "bat", C["ok"]), ("temp", "temp", C["danger"])]),
        ]
        for i, ch in enumerate(self.charts):
            ch.grid(row=i // 2, column=i % 2, sticky="nsew", padx=3, pady=3)
        for c in range(2):
            grid.columnconfigure(c, weight=1, uniform="c")
        for r in range(3):
            grid.rowconfigure(r, weight=1, uniform="r")

        # --- log (below the tabs)
        outer, header, box = card(center, "Log", padding=(10, 6))
        outer.pack(fill=tk.X, pady=(8, 0))
        self.log_text = tk.Text(box, height=6, state=tk.DISABLED, wrap=tk.WORD, bg=C["card"],
                                fg=C["muted"], font=F["mono"], relief=tk.FLAT, bd=0,
                                highlightthickness=0, insertbackground=C["text"])
        for tag, color in (("err", C["danger"]), ("warn", C["warn"]), ("ok", C["ok"]),
                           ("puddle", C["puddle"]), ("wp", C["accent"])):
            self.log_text.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(box, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_right(self, right):
        # --- camera
        outer, header, box = card(right, "Camera")
        outer.pack(fill=tk.X)
        self.cam_seg = Segmented(header, ["Voor", "Onder"],
                                 lambda v: self.set_camera_mode(v == "Onder"), "Onder")
        self.cam_seg.pack(side=tk.RIGHT)
        self.cam_w, self.cam_h = 360, 240
        self.cam = tk.Canvas(box, width=self.cam_w, height=self.cam_h, bg="#05080b",
                             highlightthickness=0)
        self.cam.pack()
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(6, 0))
        self.show_dets = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Detecties tonen", variable=self.show_dets).pack(side=tk.LEFT)
        self.record_btn = ttk.Button(row, text="● Opnemen (dataset)", style="Small.TButton",
                                     command=self.toggle_record)
        self.record_btn.pack(side=tk.RIGHT)

        # --- instruments
        outer, _, box = card(right, "Vlucht")
        outer.pack(fill=tk.X, pady=(8, 0))
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack()
        self.attitude = AttitudeIndicator(row, size=124)
        self.attitude.pack(side=tk.LEFT)
        self.compass = Compass(row, size=124)
        self.compass.pack(side=tk.LEFT, padx=4)
        self.height = HeightGauge(row, width=96, height=140)
        self.height.pack(side=tk.LEFT)
        tiles = ttk.Frame(box, style="Card.TFrame")
        tiles.pack(fill=tk.X, pady=(6, 0))
        self.tiles = {}
        for i, (key, label, unit) in enumerate((
                ("tof", "ToF (afstand grond)", "cm"), ("alt", "Barometer-hoogte", "cm"),
                ("ground", "Terrein onder drone", "cm"), ("speed", "Snelheid", "cm/s"),
                ("temp", "Temperatuur", "°C"), ("odo", "Odometrie", ""))):
            t = StatTile(tiles, label, unit, width=6)
            t.grid(row=i // 2, column=i % 2, sticky="w")
            self.tiles[key] = t
        tiles.columnconfigure(0, weight=1)
        tiles.columnconfigure(1, weight=1)

        # --- puddles
        outer, header, box = card(right, "Gevonden plassen")
        outer.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.puddle_count = ttk.Label(header, text="0", style="Muted.TLabel")
        self.puddle_count.pack(side=tk.RIGHT)
        self.puddle_tree = ttk.Treeview(box, columns=("id", "x", "y", "area", "hits"),
                                        show="headings", height=3, selectmode="browse")
        for col, text, w in (("id", "#", 30), ("x", "x", 60), ("y", "y", 60),
                             ("area", "cm²", 60), ("hits", "gezien", 60)):
            self.puddle_tree.heading(col, text=text)
            self.puddle_tree.column(col, width=w, anchor=tk.E)
        self.puddle_tree.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(row, text="Vlieg naar plas", style="Small.TButton",
                   command=self.goto_puddle).pack(side=tk.LEFT)
        ttk.Button(row, text="Wis lijst", style="Small.TButton",
                   command=lambda: self.app.submit({"type": "reset_puddles"})).pack(
            side=tk.RIGHT)

    def _bind_keys(self):
        r = self.root
        r.bind("<Return>", lambda e: self._key(self.add_waypoint, e, typing_ok=True))
        r.bind("<Delete>", lambda e: self._key(self.delete_selected, e))
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-s>", lambda e: self.save_mission())
        r.bind("<Control-o>", lambda e: self.load_mission())
        for i in range(3):
            r.bind(f"<Control-Key-{i + 1}>", lambda e, i=i: self.tabs.select(i))
        r.bind("<KeyPress-l>", lambda e: self._key(self.land, e))
        r.bind("<KeyPress-h>", lambda e: self._key(self.helipad_land, e))
        r.bind("<KeyPress-x>", lambda e: self._key(self.emergency, e))

    def _key(self, action, event, typing_ok=False):
        # don't trigger flight keys while typing coordinates
        if typing_ok or not isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            action()

    # ================================================================ refresh
    def tick(self):
        try:
            self._tick()
        except Exception as e:      # keep the GUI alive whatever happens
            print(f"⚠️  GUI: {e}")
        finally:
            self.root.after(TICK_MS, self.tick)

    def _tick(self):
        app = self.app
        now = time.time()
        self._tick_n += 1
        self._drain_log()
        self.toast.tick()

        # displayed pose glides towards the measured pose (smooth animation)
        pose = app.pose.get()
        if self.show is None:
            self.show = list(pose)
        else:
            for i in range(3):
                self.show[i] += (pose[i] - self.show[i]) * 0.35
            self.show[3] += ((pose[3] - self.show[3] + 180) % 360 - 180) * 0.35
        tel = app.telemetry.latest if hasattr(app, "telemetry") else {}

        # flight time + trail
        if app.airborne and not self._was_airborne:
            self.takeoff_time = now
        self._was_airborne = app.airborne
        if app.mission_id != self.trail_mission:
            self.trail_mission = app.mission_id
            self.trail = []
        x, y, z, _ = pose
        if app.airborne and (not self.trail or
                             max(abs(a - b) for a, b in zip(self.trail[-1], (x, y, z))) > 2):
            self.trail.append((x, y, z))

        self._topbar(tel, now)
        self.attitude.set(tel.get("pitch"), tel.get("roll"))
        self.compass.set(self.show[3] if app.pose else None, app.pose.reference_yaw)
        self.height.set(tel.get("alt") if app.airborne else 0.0,
                        tel.get("tof") if app.airborne else None, tel.get("ground"))
        for w in (self.attitude, self.compass, self.height, self.battery):
            w.tick()
        if self._tick_n % 2 == 0:
            self.update_camera(tel)

        if now >= self._next["side"]:
            self._next["side"] = now + 0.2
            self._tiles(tel)
            self.update_puddles(app.tracker.confirmed())

        tab = self.tabs.index(self.tabs.select())
        if tab == 0 and now >= self._next["3d"]:
            self._next["3d"] = now + PLOT_3D_S
            self.map3d.update(self._scene(tel))
        elif tab == 1 and now >= self._next["map"]:
            self._next["map"] = now + PLOT_MAP_S
            self._update_map(tel)
        elif tab == 2 and now >= self._next["chart"]:
            self._next["chart"] = now + CHART_S
            self._update_charts()

    def _drain_log(self):
        lines = []
        while not self.log.queue.empty():
            lines.append(self.log.queue.get_nowait())
        if not lines:
            return
        self.log_text.configure(state=tk.NORMAL)
        for chunk in "".join(lines).splitlines(keepends=True):
            tag = ("err" if "❌" in chunk or "🚨" in chunk else
                   "warn" if "⚠" in chunk or "🛑" in chunk else
                   "puddle" if "💧" in chunk else
                   "wp" if "📍" in chunk or "🗺" in chunk else
                   "ok" if "✅" in chunk or "🏁" in chunk else None)
            self.log_text.insert(tk.END, chunk, tag or ())
            if tag in ("err", "warn"):
                self.toast.show(chunk.strip(), "error" if tag == "err" else "warn", 6)
            elif tag == "puddle":
                self.toast.show(chunk.strip(), "info", 5)
        self.log_text.delete("1.0", "end-400l")       # keep the last 400 lines
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _topbar(self, tel, now):
        app = self.app
        text, color = STATES.get(app.state, (app.state.upper(), C["faint"]))
        self.state_pill.configure(text=text, bg=color)
        battery = tel.get("battery")
        if battery is None:
            try:
                battery = app.tello.get_battery()
            except Exception:
                battery = None
        self.battery.set(battery)
        v = self.top_vars
        if app.airborne and self.takeoff_time:
            s = int(now - self.takeoff_time)
            v["time"].configure(text=f"⏱ {s // 60:02d}:{s % 60:02d}", fg=C["text"])
        else:
            v["time"].configure(text="⏱ --:--", fg=C["muted"])
        v["height"].configure(text=f"↕ {self.show[2]:.0f} cm", fg=C["text"] if app.airborne
                              else C["muted"])
        if not app.airborne:
            v["odo"].configure(text="odometrie –", fg=C["muted"])
        elif app.pose.measured:
            v["odo"].configure(text=f"odometrie ✔ {app.pose.quality:.2f}", fg=C["ok"])
        else:
            v["odo"].configure(text="odometrie ✖ (commando's)", fg=C["warn"])
        rx = getattr(app.link, "last_rx", None)
        if rx is None:
            v["jetson"].configure(text="Jetson: geen contact", fg=C["muted"])
        elif now - rx < 3:
            v["jetson"].configure(text="Jetson ● verbonden", fg=C["ok"])
        else:
            v["jetson"].configure(text=f"Jetson: {now - rx:.0f} s stil", fg=C["warn"])
        if app.recording:
            if not self.rec_pill.winfo_ismapped():
                self.rec_pill.pack(side=tk.LEFT, padx=(0, 12), before=v["time"])
            self.rec_pill.configure(bg=C["danger"] if int(now * 2) % 2 else "#7a1f25")
        elif self.rec_pill.winfo_ismapped():
            self.rec_pill.pack_forget()

    def _tiles(self, tel):
        app = self.app
        t = self.tiles
        t["tof"].set(fmt(tel.get("tof")))
        t["alt"].set(fmt(tel.get("alt")))
        g = tel.get("ground")
        t["ground"].set(fmt(g, 0) if g is None else f"{g:+.0f}",
                        C["warn"] if g is not None and abs(g) >= cfg.TERRAIN_OBSTACLE_CM else None)
        vs = [tel.get(k) for k in ("vx", "vy", "vz")]
        t["speed"].set("–" if None in vs else f"{math.sqrt(sum(v * v for v in vs)):.0f}")
        temp = tel.get("temp")
        t["temp"].set(fmt(temp), C["danger"] if temp is not None and temp >= 85 else None)
        if not app.airborne:
            t["odo"].set("–")
        else:
            q = app.pose.quality
            t["odo"].set(f"{q:.2f}" if app.pose.measured else "kwijt",
                         C["ok"] if app.pose.measured else C["warn"])

    def _footprint(self, tel):
        """Corners of the ground area the downward camera sees right now."""
        h = tel.get("tof")
        if not self.app.airborne or not h or not self.app.downvision_enabled:
            return None
        x, y, _, yaw = self.show
        half_w = h * math.tan(math.radians(cfg.CAM_HFOV_DEG) / 2)
        half_l = half_w * 0.75
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        pts = []
        for f, l in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
            pts.append((x + f * c - l * s, y + f * s + l * c))
        return pts + [pts[0]]

    def _scene(self, tel):
        terrain = self.app.telemetry.terrain if hasattr(self.app, "telemetry") else None
        x, y = self.show[0], self.show[1]
        return {
            "pose": tuple(self.show), "pitch": tel.get("pitch"), "roll": tel.get("roll"),
            "waypoints": list(self.waypoints), "selected": self.sel,
            "mission": list(self.app.current_waypoints), "reached": self.app.waypoints_reached,
            "trail": self.trail, "airborne": self.app.airborne,
            "puddles": self.app.tracker.confirmed(),
            "terrain": terrain.snapshot() if terrain and self.terrain3d.get() else [],
            "terrain_version": (terrain.version if terrain else 0, self.terrain3d.get()),
            "ground_under": terrain.elevation_at(x, y) if terrain else None,
            "footprint": self._footprint(tel),
        }

    def _update_map(self, tel):
        terrain = self.app.telemetry.terrain
        s = self._scene(tel)
        s["terrain"] = terrain.snapshot()
        s["terrain_version"] = terrain.version
        s["profile"] = terrain.profile_copy()
        if self._summary is None or self._summary[0] != terrain.version:
            self._summary = (terrain.version, terrain.summary())
            self._terrain_text(self._summary[1], terrain)
        s["obstacles"] = (self._summary[1] or {}).get("obstacles", [])
        self.mapview.update(s)

    def _terrain_text(self, summ, terrain):
        if not terrain.calibrated:
            self.terrain_info.configure(
                text="Nog geen metingen.\n\nVlieg over het gebied: de drone vergelijkt zijn "
                     "barometer-hoogte met de afstand tot de grond (ToF). Waar de ToF kleiner "
                     "wordt terwijl de hoogte gelijk blijft, ligt iets hogers.\n\nTip: gebruik "
                     "Raster… om het gebied systematisch af te vliegen.")
            return
        if summ is None:
            self.terrain_info.configure(text="Bezig met meten…")
            return
        lines = [f"Gemeten oppervlak   {summ['area_m2']:.1f} m²  ({summ['cells']} vakjes)",
                 f"Laagste / hoogste    {summ['min']:+.0f} / {summ['max']:+.0f} cm",
                 f"Ruwheid (std)         {summ['roughness']:.0f} cm", ""]
        if summ["obstacles"]:
            lines.append(f"Obstakels (> {cfg.TERRAIN_OBSTACLE_CM} cm):")
            for o in summ["obstacles"][:6]:
                lines.append(f"  {o['height']:+.0f} cm bij ({o['x']:.0f}, {o['y']:.0f})"
                             f"  ~{o['size_cm2'] / 10000:.2f} m²")
        else:
            lines.append("Geen obstakels gevonden.")
        if summ["dips"]:
            lines.append("")
            lines.append("Lager dan de vloer:")
            for o in summ["dips"][:4]:
                lines.append(f"  {o['height']:+.0f} cm bij ({o['x']:.0f}, {o['y']:.0f})")
        lines += ["", "Nauwkeurigheid: ±10–20 cm (barometer). Herijk boven de vloer als de "
                      "kaart langzaam verschuift."]
        self.terrain_info.configure(text="\n".join(lines))

    def _update_charts(self):
        window = WINDOWS[self.window_seg.value]
        keys = {k for ch in self.charts for k, _, _ in ch.series}
        data = self.app.telemetry.series(keys, window)
        for ch in self.charts:
            ch.draw(data, window)

    def update_puddles(self, puddles):
        key = tuple((p["id"], p["x"], p["y"], p.get("hits")) for p in puddles)
        if key == self._puddle_key:
            return
        self._puddle_key = key
        sel = self.puddle_tree.selection()
        self.puddle_tree.delete(*self.puddle_tree.get_children())
        for p in puddles:
            self.puddle_tree.insert("", tk.END, iid=str(p["id"]), values=(
                p["id"], f"{p['x']:.0f}", f"{p['y']:.0f}", f"{p['area_cm2']:.0f}",
                p.get("hits", "")))
        if sel and self.puddle_tree.exists(sel[0]):
            self.puddle_tree.selection_set(sel[0])
        self.puddle_count.configure(text=str(len(puddles)))

    def update_camera(self, tel):
        app = self.app
        frame = app.frame_reader.frame if hasattr(app, "frame_reader") else None
        vis = app.latest_vis() if (self.show_dets.get() and app.downvision_enabled) else None
        self.cam.delete("all")
        if frame is None and vis is None:
            self.cam.create_text(self.cam_w / 2, self.cam_h / 2, text="Geen camerabeeld",
                                 fill=C["muted"], font=F["base"])
            return
        if vis is not None:
            img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 2:
            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            img = frame if cfg.FRAME_IS_RGB else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        f = min(self.cam_w / w, self.cam_h / h)
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                         interpolation=cv2.INTER_AREA)
        self._camera_photo = ImageTk.PhotoImage(Image.fromarray(img))
        cx, cy = self.cam_w / 2, self.cam_h / 2
        self.cam.create_image(cx, cy, image=self._camera_photo)
        x0, y0 = cx - img.shape[1] / 2, cy - img.shape[0] / 2
        x1, y1 = x0 + img.shape[1], y0 + img.shape[0]
        # HUD: crosshair, mode, height
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.cam.create_line(cx + dx * 6, cy + dy * 6, cx + dx * 16, cy + dy * 16,
                                 fill="#ffffff", width=1)
        mode = "ONDER" if app.downvision_enabled else "VOOR"
        self.cam.create_rectangle(x0 + 4, y0 + 4, x0 + 58, y0 + 22, fill="#05080b", outline="")
        self.cam.create_text(x0 + 8, y0 + 6, text=mode, anchor=tk.NW, fill=C["puddle"],
                             font=F["title"])
        if tel.get("tof") and app.airborne:
            self.cam.create_text(x0 + 8, y1 - 6, anchor=tk.SW, fill="white", font=F["small"],
                                 text=f"ToF {tel['tof']:.0f} cm")
        if app.recording:
            self.cam.create_oval(x1 - 18, y0 + 8, x1 - 8, y0 + 18, fill=C["danger"],
                                 outline="")

    # ================================================================ camera
    def set_camera_mode(self, use_downvision):
        try:
            self.app.set_downvision(use_downvision)
        except Exception as e:
            self.toast.show(f"Camera wisselen mislukt: {e}", "error")
        self.cam_seg.set("Onder" if self.app.downvision_enabled else "Voor")

    def toggle_record(self):
        self.app.toggle_recording()
        self.record_btn.configure(text="■ Stop opnemen" if self.app.recording
                                  else "● Opnemen (dataset)")

    # ============================================================= waypoints
    def _speed_changed(self, v):
        self.speed.set(int(float(v)))
        self.speed_lbl.configure(text=f"{self.speed.get()} cm/s")
        if hasattr(self, "path_stats"):
            self._path_stats()

    def _read_entries(self, quiet=False):
        try:
            point = tuple(float(self.entries[k].get().replace(",", ".")) for k in "xyz")
        except ValueError:
            return None, "x, y en z moeten getallen zijn (cm)"
        try:
            validate_waypoints([point])
        except ValueError as e:
            return None, "Buiten de geofence: " + str(e).replace("waypoint 0: ", "")
        return point, None

    def _validate_entries(self):
        point, err = self._read_entries()
        for k, e in self.entries.items():
            try:
                v = float(e.get().replace(",", "."))
                lo, hi = cfg.GEOFENCE[k]
                ok = lo <= v <= hi
            except ValueError:
                ok = False
            e.configure(style="TSpinbox" if ok else "Bad.TSpinbox")
        self.entry_hint.configure(text=err or "")
        return point

    def _push(self):
        self.history.append(list(self.waypoints))
        del self.history[:-50]

    def _changed(self, select=None):
        self.sel = select if select is not None and 0 <= select < len(self.waypoints) else None
        self.tree.delete(*self.tree.get_children())
        for i, (x, y, z) in enumerate(self.waypoints):
            self.tree.insert("", tk.END, iid=str(i), values=(i + 1, f"{x:.0f}", f"{y:.0f}",
                                                             f"{z:.0f}"))
        if self.sel is not None:
            self.tree.selection_set(str(self.sel))
            self.tree.see(str(self.sel))
        self._path_stats()
        self._next["3d"] = self._next["map"] = 0      # redraw now

    def _path_stats(self):
        if not self.waypoints:
            self.path_stats.configure(text="leeg")
            return
        pts = [tuple(self.app.pose.get()[:3])] + list(self.waypoints)
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        stops = len(self.waypoints) * (2.5 if NAV_MODES[self.nav_mode.get()] == "go" else 0.5)
        duration = length / max(self.speed.get(), 1) + stops + 8      # + takeoff/landing
        self.path_stats.configure(text=f"{len(self.waypoints)} punten · {length / 100:.1f} m · "
                                       f"±{duration:.0f} s")

    def _tree_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        self.sel = int(sel[0])
        for k, v in zip("xyz", self.waypoints[self.sel]):
            self.entries[k].set(f"{v:.0f}")
        self._validate_entries()
        self._next["3d"] = self._next["map"] = 0

    def add_waypoint(self):
        point = self._validate_entries()
        if point is None:
            self.toast.show(self.entry_hint.cget("text"), "warn")
            return
        self._push()
        self.waypoints.append(point)
        self._changed(select=len(self.waypoints) - 1)

    def update_waypoint(self):
        point = self._validate_entries()
        if self.sel is None:
            self.toast.show("Selecteer eerst een punt om bij te werken.", "warn")
            return
        if point:
            self._push()
            self.waypoints[self.sel] = point
            self._changed(select=self.sel)

    def fill_current_pos(self):
        for k, v in zip("xyz", self.app.pose.get()[:3]):
            self.entries[k].set(f"{v:.0f}")
        self._validate_entries()

    def move_selected(self, delta):
        i = self.sel
        if i is None or not 0 <= i + delta < len(self.waypoints):
            return
        self._push()
        wp = self.waypoints
        wp[i], wp[i + delta] = wp[i + delta], wp[i]
        self._changed(select=i + delta)

    def delete_selected(self):
        if self.sel is not None:
            self._push()
            del self.waypoints[self.sel]
            self._changed(select=min(self.sel, len(self.waypoints) - 1))

    def clear_waypoints(self):
        if self.waypoints:
            self._push()
            self.waypoints = []
            self._changed()
            self.toast.show("Pad gewist (Ctrl+Z om terug te zetten).", "info")

    def undo(self):
        if not self.history:
            self.toast.show("Niets om ongedaan te maken.", "info", 2)
            return
        self.waypoints = self.history.pop()
        self._changed(select=min(self.sel or 0, len(self.waypoints) - 1))

    # --- map callbacks
    def _map_z(self):
        try:
            z = float(self.entries["z"].get().replace(",", "."))
        except ValueError:
            z = 80.0
        lo, hi = cfg.GEOFENCE["z"]
        return min(max(z, lo), hi)

    def _valid(self, point):
        try:
            validate_waypoints([point])
            return True
        except ValueError as e:
            self.toast.show("Buiten de geofence: " + str(e).replace("waypoint 0: ", ""), "warn")
            return False

    def _map_add(self, x, y):
        point = (x, y, self._map_z())
        if self._valid(point):
            self._push()
            self.waypoints.append(point)
            self._changed(select=len(self.waypoints) - 1)

    def _map_move(self, i, x, y, final):
        if i >= len(self.waypoints):
            return
        if not self._dragging:
            self._push()
            self._dragging = True
        if x is not None:
            point = (x, y, self.waypoints[i][2])
            try:
                validate_waypoints([point])
                self.waypoints[i] = point
            except ValueError:
                pass
        if final:
            self._dragging = False
            self._changed(select=i)
            self._tree_selected()
        else:
            self._next["map"] = 0

    def _map_delete(self, i):
        self._push()
        del self.waypoints[i]
        self._changed(select=min(i, len(self.waypoints) - 1))

    def _map_select(self, i):
        self._changed(select=i)
        self._tree_selected()

    def _map_hover(self, info):
        if info is None:
            self.mapview.set_hover_text(MAP_HELP)
            return
        x, y, i = info
        g = self.app.telemetry.terrain.elevation_at(x, y)
        text = f"x {x:.0f}   y {y:.0f} cm"
        if g is not None:
            text += f"   ·   terrein {g:+.0f} cm"
        if i is not None:
            text += f"   ·   punt {i + 1} (slepen om te verplaatsen)"
        self.mapview.set_hover_text(text)

    # --- raster pattern
    def pattern_dialog(self):
        PatternDialog(self.root, self)

    def set_pattern(self, points):
        self._push()
        self.waypoints = points
        self._changed(select=0)
        self.toast.show(f"Rasterpad met {len(points)} punten gemaakt.", "ok")

    # --- files
    def save_mission(self):
        if not self.waypoints:
            self.toast.show("Er is nog geen pad om op te slaan.", "warn")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialdir=FLIGHTS_DIR,
                                            filetypes=[("Missie", "*.json")])
        if path:
            with open(path, "w") as f:
                json.dump(self._mission(self.waypoints, self.land_at_end.get()), f, indent=2)
            self.toast.show(f"Opgeslagen: {os.path.basename(path)}", "ok")

    def load_mission(self):
        path = filedialog.askopenfilename(initialdir=FLIGHTS_DIR, filetypes=[("Missie", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            points = validate_waypoints(data["waypoints"] if isinstance(data, dict) else data)
            if isinstance(data, dict):
                self.speed.set(int(data.get("speed", self.speed.get())))
                self._speed_changed(self.speed.get())
                self.land_at_end.set(bool(data.get("land_at_end", self.land_at_end.get())))
                self.level.set(bool(data.get("level", False)))
                self.helipad_search.set(bool(data.get("helipad_search", False)))
                for attr, options in (("nav_mode", NAV_MODES), ("fine", FINE_MODES)):
                    label = next((k for k, v in options.items() if v == data.get(attr)), None)
                    if label:
                        getattr(self, attr).set(label)
        except (ValueError, KeyError, OSError) as e:
            messagebox.showerror("Laden mislukt", str(e))
            return
        self._push()
        self.waypoints = [tuple(p) for p in points]
        self._changed(select=0)
        self.toast.show(f"Geladen: {os.path.basename(path)} ({len(points)} punten)", "ok")

    def export_telemetry(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=time.strftime("telemetrie_%H%M%S.csv"),
                                            filetypes=[("CSV", "*.csv")])
        if path:
            n = self.app.telemetry.export_csv(path)
            self.toast.show(f"{n} metingen opgeslagen in {os.path.basename(path)}", "ok")

    def export_terrain(self):
        cells = self.app.telemetry.terrain.snapshot()
        if not cells:
            self.toast.show("Nog geen terreinmetingen.", "warn")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=time.strftime("terrein_%H%M%S.csv"),
                                            filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w") as f:
                f.write("x_cm,y_cm,hoogte_cm,std_cm,metingen\n")
                for x, y, e, sd, n in sorted(cells):
                    f.write(f"{x:.0f},{y:.0f},{e:.1f},{sd:.1f},{n}\n")
            self.toast.show(f"Terrein ({len(cells)} vakjes) opgeslagen.", "ok")

    def rereference(self):
        if not self.app.airborne:
            self.toast.show("Herijken kan alleen in de lucht, boven de vloer.", "warn")
            return
        self.app.telemetry.terrain.rereference()
        self.toast.show("Grond onder de drone wordt het nieuwe nulniveau.", "ok")

    def clear_terrain(self):
        self.app.telemetry.terrain.reset()
        self._summary = None
        self.toast.show("Terreinkaart gewist.", "info")

    def _terrain3d_changed(self):
        self.map3d.show_terrain = self.terrain3d.get()
        self._next["3d"] = 0

    def _set_view(self, name):
        self.map3d.set_view(name)

    def clear_trail(self):
        self.trail = []

    # ================================================================ flying
    def _mission(self, waypoints, land_at_end):
        self.mission_count += 1
        return {"type": "mission", "id": f"gui-{self.mission_count}", "speed": self.speed.get(),
                "land_at_end": land_at_end, "nav_mode": NAV_MODES[self.nav_mode.get()],
                "fine": FINE_MODES[self.fine.get()], "level": self.level.get(),
                "helipad_search": self.helipad_search.get() and land_at_end,
                "waypoints": [list(p) for p in waypoints]}

    def start_mission(self):
        if not self.waypoints:
            self.toast.show("Voeg eerst waypoints toe (klik op de kaart of vul x, y, z in).",
                            "warn")
            return
        self.app.submit(self._mission(self.waypoints, self.land_at_end.get()))
        self.toast.show(f"Missie gestart: {len(self.waypoints)} punten.", "ok")

    def goto_selected(self):
        if self.sel is None:
            self.toast.show("Selecteer eerst een punt in de lijst of op de kaart.", "warn")
            return
        self.app.submit(self._mission([self.waypoints[self.sel]], land_at_end=False))

    def goto_puddle(self):
        sel = self.puddle_tree.selection()
        p = next((p for p in self.app.tracker.confirmed() if sel and str(p["id"]) == sel[0]),
                 None)
        if p is None:
            self.toast.show("Selecteer eerst een plas in de lijst.", "warn")
            return
        z = self.app.pose.get()[2] if self.app.airborne else self._map_z()
        point = (p["x"], p["y"], min(max(z, cfg.GEOFENCE["z"][0]), cfg.GEOFENCE["z"][1]))
        if self._valid(point):
            self.app.submit(self._mission([point], land_at_end=False))

    def land(self):
        self.app.submit({"type": "land"})

    def helipad_land(self):
        """Take off if needed, look for an H below the drone and land on it (no path)."""
        self.app.submit({"type": "helipad_land"})
        self.toast.show("Zoeken naar de H en erop landen…")

    def emergency(self):
        self.app.emergency_stop()

    def on_close(self):
        if self.app.airborne and not messagebox.askyesno(
                "Afsluiten", "De drone vliegt nog. Landen en afsluiten?"):
            return
        self.root.title("Landen en afsluiten...")
        self.root.update()
        self.app.shutdown()
        self.root.destroy()


class PatternDialog:
    """Raster ('grasmaaier') pattern to scan an area for puddles and terrain."""

    def __init__(self, root, gui):
        self.gui = gui
        top = self.top = tk.Toplevel(root, bg=C["card"], padx=14, pady=12)
        top.title("Rasterpad")
        top.transient(root)
        top.resizable(False, False)
        ttk.Label(top, text="Vlieg een gebied in banen af", style="Card.TLabel",
                  font=F["bold"]).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.vars = {}
        for r, (key, label, default) in enumerate((
                ("x0", "Start x (cm)", 0), ("y0", "Start y (cm)", 0),
                ("length", "Lengte vooruit (cm)", 300), ("width", "Breedte naar links (cm)", 200),
                ("spacing", "Afstand tussen banen (cm)", 0), ("z", "Hoogte (cm)", 80)), 1):
            ttk.Label(top, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w",
                                                                 pady=2)
            var = tk.StringVar(value=str(default))
            ttk.Spinbox(top, from_=-600, to=600, increment=10, width=8, textvariable=var,
                        justify=tk.RIGHT, command=self._hint).grid(row=r, column=1, pady=2)
            self.vars[key] = var
        self.vars["z"].trace_add("write", lambda *a: self._hint())
        self.hint = ttk.Label(top, text="", style="Muted.TLabel", wraplength=260)
        self.hint.grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 4))
        row = ttk.Frame(top, style="Card.TFrame")
        row.grid(row=9, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(row, text="Annuleren", command=top.destroy).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Maak pad", command=self.make).pack(side=tk.LEFT)
        self._hint(set_spacing=True)
        top.update_idletasks()
        top.geometry(f"+{root.winfo_rootx() + 120}+{root.winfo_rooty() + 120}")
        top.grab_set()

    def _values(self):
        return {k: float(v.get().replace(",", ".")) for k, v in self.vars.items()}

    def _footprint(self, z):
        return 2 * z * math.tan(math.radians(cfg.CAM_HFOV_DEG) / 2)

    def _hint(self, set_spacing=False):
        try:
            z = float(self.vars["z"].get())
        except ValueError:
            return
        fw = self._footprint(z)
        if set_spacing:
            self.vars["spacing"].set(str(int(fw * 0.75 // 5 * 5)))
        self.hint.configure(text=f"Op {z:.0f} cm ziet de camera ±{fw:.0f} cm breed. Een "
                                 f"tussenafstand tot ±{fw * 0.8:.0f} cm geeft overlap, zodat "
                                 f"geen plas gemist wordt.")

    def make(self):
        try:
            v = self._values()
        except ValueError:
            self.hint.configure(text="Vul overal een getal in.")
            return
        if v["spacing"] <= 0 or v["length"] == 0:
            self.hint.configure(text="Lengte en tussenafstand moeten groter dan 0 zijn.")
            return
        lanes = int(abs(v["width"]) // v["spacing"]) + 1
        sign = 1 if v["width"] >= 0 else -1
        pts = []
        for k in range(lanes):
            y = v["y0"] + sign * k * v["spacing"]
            xs = (v["x0"], v["x0"] + v["length"])
            if k % 2:
                xs = xs[::-1]
            pts += [(xs[0], y, v["z"]), (xs[1], y, v["z"])]
        try:
            validate_waypoints(pts)
        except ValueError as e:
            self.hint.configure(text=f"Buiten de geofence: {e}")
            return
        self.gui.set_pattern(pts)
        self.top.destroy()


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
        tello = FakeTello(camera_turn_deg=cfg.CAM_ROTATE_DEG)
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
    TelloGUI(root, app, log, sim=args.sim)
    root.mainloop()


if __name__ == "__main__":
    main()
