"""Colours, fonts and the ttk style of the mission control GUI (dark theme)."""

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

C = {
    "bg": "#0e1318",        # window
    "panel": "#151c24",     # columns
    "card": "#1b242e",      # cards
    "card2": "#222d39",     # inputs, hover
    "border": "#2b3745",
    "text": "#e6edf3",
    "muted": "#8b98a6",
    "faint": "#5b6876",
    "accent": "#4cb2ff",    # active mission, selection
    "ok": "#3ccf8e",
    "warn": "#f5a524",
    "danger": "#f0545c",
    "draft": "#ff9f43",     # planned path
    "trail": "#f472b6",     # flown trail
    "cyan": "#22d3ee",
    "terrain_hi": "#c08a4b",
    "sky": "#2f6fae",
    "ground": "#7a5532",
    # graph series
    "s1": "#4cb2ff", "s2": "#f5a524", "s3": "#3ccf8e", "s4": "#b48cff", "s5": "#f472b6",
}

F = {}  # fonts, filled by apply()


def _family(root):
    have = set(tkfont.families(root))
    for name in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Ubuntu", "DejaVu Sans"):
        if name in have:
            return name
    return tkfont.nametofont("TkDefaultFont").actual("family")


def _mono(root):
    have = set(tkfont.families(root))
    for name in ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"):
        if name in have:
            return name
    return tkfont.nametofont("TkFixedFont").actual("family")


def apply(root):
    fam, mono = _family(root), _mono(root)
    F.update({
        "base": (fam, 10), "small": (fam, 9), "tiny": (fam, 8),
        "bold": (fam, 10, "bold"), "title": (fam, 9, "bold"),
        "big": (fam, 15, "bold"), "huge": (fam, 18, "bold"),
        "mono": (mono, 9),
    })
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(family=fam, size=10)
    root.configure(bg=C["bg"])
    root.option_add("*TCombobox*Listbox.background", C["card2"])
    root.option_add("*TCombobox*Listbox.foreground", C["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=C["panel"], foreground=C["text"], fieldbackground=C["card2"],
                bordercolor=C["border"], darkcolor=C["card"], lightcolor=C["card"],
                troughcolor=C["card2"], focuscolor=C["accent"], insertcolor=C["text"],
                selectbackground=C["accent"], selectforeground="#06121c", font=F["base"])
    s.configure("TFrame", background=C["panel"])
    s.configure("Bg.TFrame", background=C["bg"])
    s.configure("Card.TFrame", background=C["card"])
    s.configure("TLabel", background=C["panel"], foreground=C["text"])
    s.configure("Card.TLabel", background=C["card"])
    s.configure("Muted.TLabel", background=C["card"], foreground=C["muted"], font=F["small"])
    s.configure("Title.TLabel", background=C["card"], foreground=C["muted"], font=F["title"])
    s.configure("Error.TLabel", background=C["card"], foreground=C["danger"], font=F["small"])
    s.configure("Value.TLabel", background=C["card"], font=F["big"])

    s.configure("TButton", background=C["card2"], foreground=C["text"], borderwidth=0,
                padding=(10, 5), relief="flat")
    s.map("TButton", background=[("pressed", C["border"]), ("active", "#2a3746"),
                                 ("disabled", C["card"])],
          foreground=[("disabled", C["faint"])])
    s.configure("Small.TButton", padding=(6, 3), font=F["small"])
    s.configure("Seg.TButton", padding=(8, 3), font=F["small"])
    s.configure("SegOn.TButton", padding=(8, 3), font=F["small"], background=C["accent"],
                foreground="#06121c")
    s.map("SegOn.TButton", background=[("active", "#6cc2ff")])

    s.configure("TEntry", padding=4, fieldbackground=C["card2"], foreground=C["text"])
    s.configure("Bad.TEntry", fieldbackground="#3a1f25", bordercolor=C["danger"])
    s.configure("TSpinbox", padding=3, fieldbackground=C["card2"], foreground=C["text"],
                arrowcolor=C["muted"], arrowsize=10)
    s.configure("Bad.TSpinbox", fieldbackground="#3a1f25", bordercolor=C["danger"])
    s.configure("TCombobox", padding=3, fieldbackground=C["card2"], foreground=C["text"],
                arrowcolor=C["muted"], background=C["card2"])
    s.map("TCombobox", fieldbackground=[("readonly", C["card2"])],
          foreground=[("readonly", C["text"])], selectbackground=[("readonly", C["card2"])],
          selectforeground=[("readonly", C["text"])])
    s.configure("TCheckbutton", background=C["card"], foreground=C["text"])
    s.map("TCheckbutton", background=[("active", C["card"])],
          indicatorcolor=[("selected", C["accent"]), ("!selected", C["card2"])])
    s.configure("Horizontal.TScale", background=C["accent"], troughcolor=C["card2"])

    s.configure("Treeview", background=C["card"], fieldbackground=C["card"],
                foreground=C["text"], rowheight=22, borderwidth=0)
    s.map("Treeview", background=[("selected", C["accent"])],
          foreground=[("selected", "#06121c")])
    s.configure("Treeview.Heading", background=C["card2"], foreground=C["muted"],
                font=F["title"], relief="flat", padding=(4, 3))
    s.map("Treeview.Heading", background=[("active", C["border"])])

    s.configure("TNotebook", background=C["bg"], borderwidth=0, tabmargins=(0, 0, 0, 0))
    s.configure("TNotebook.Tab", background=C["bg"], foreground=C["muted"], padding=(14, 6),
                borderwidth=0, font=F["bold"])
    s.map("TNotebook.Tab", background=[("selected", C["panel"])],
          foreground=[("selected", C["text"]), ("active", C["text"])])
    s.configure("Vertical.TScrollbar", background=C["card2"], troughcolor=C["card"],
                arrowcolor=C["muted"], borderwidth=0)
    return s


def mpl_style(fig, *axes):
    """Dark colours for matplotlib figures/axes."""
    fig.patch.set_facecolor(C["panel"])
    for ax in axes:
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        for side in getattr(ax, "spines", {}).values():
            side.set_color(C["border"])
        ax.xaxis.label.set_color(C["muted"])
        ax.yaxis.label.set_color(C["muted"])


def lerp_color(a, b, t):
    a = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def level_color(frac):
    """0 -> red, 0.5 -> orange, 1 -> green (battery etc.)."""
    if frac < 0.5:
        return lerp_color(C["danger"], C["warn"], frac / 0.5)
    return lerp_color(C["warn"], C["ok"], (frac - 0.5) / 0.5)


def tk_button(parent, text, command, color, fg="white", font=None, **kw):
    """Flat coloured button (ttk can't colour single buttons on every platform)."""
    hover = lerp_color(color, "#ffffff", 0.15)
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg, activebackground=hover,
                  activeforeground=fg, relief=tk.FLAT, bd=0, highlightthickness=0,
                  cursor="hand2", font=font or F["bold"], padx=8, pady=6, **kw)
    b.bind("<Enter>", lambda e: b.configure(bg=hover))
    b.bind("<Leave>", lambda e: b.configure(bg=color))
    return b
