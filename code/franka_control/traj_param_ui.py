"""
Trajectory parameter UI for trajectory_tracking.py.
Spawned as a subprocess; writes JSON parameter lines to stdout every 50 ms.

Exposes 9 sliders:
  - 1  global speed multiplier (scales all frequencies)
  - 7  per-joint amplitudes
  - 1  joint 4 bias (elbow offset)
"""

import json
import tkinter as tk
from tkinter import ttk

# Defaults must match the globals in trajectory_tracking.py
A_DEFAULT     = [0.80, 0.40, 0.50, 0.80, 0.50, 0.30, 0.50]
A_MAX         = [2.00, 1.50, 2.00, 2.00, 2.00, 2.00, 2.00]
SPEED_DEFAULT = 1.0
BIAS4_DEFAULT = -1.50

root = tk.Tk()
root.title("Franka — Trajectory Parameters")
root.resizable(False, False)

SLIDER_W = 280

def _section(label):
    ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=6, pady=(6, 0))
    ttk.Label(root, text=label, padding=(8, 3),
              font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

def _row(parent, label, lo, hi, default, fmt):
    """One labelled slider row. Returns the DoubleVar."""
    f = ttk.Frame(parent, padding=(8, 2))
    f.pack(fill=tk.X)
    ttk.Label(f, text=label,       width=10, anchor="w").grid(row=0, column=0)
    ttk.Label(f, text=f"{lo}",     width=5,  anchor="e").grid(row=0, column=1)
    var = tk.DoubleVar(value=default)
    ttk.Scale(f, from_=lo, to=hi, variable=var,
              orient=tk.HORIZONTAL, length=SLIDER_W).grid(row=0, column=2, padx=4)
    ttk.Label(f, text=f"{hi}",     width=5,  anchor="w").grid(row=0, column=3)
    val_lbl = ttk.Label(f, text=fmt(default), width=8, anchor="center")
    val_lbl.grid(row=0, column=4)
    def _upd(name, index, op, lbl=val_lbl, v=var):
        lbl.config(text=fmt(v.get()))
    var.trace_add("write", _upd)
    return var

# ── Speed ──────────────────────────────────────────────────────────────────────
_section("Global Speed  (scales all frequencies)")
speed_var = _row(root, "speed", 0.1, 5.0, SPEED_DEFAULT, lambda v: f"{v:.2f}×")

# ── Amplitudes ─────────────────────────────────────────────────────────────────
_section("Amplitudes  (rad)")
amp_vars = [
    _row(root, f"j{i+1}", 0.0, A_MAX[i], A_DEFAULT[i], lambda v: f"{v:.3f}")
    for i in range(7)
]

# ── Joint 4 bias ───────────────────────────────────────────────────────────────
_section("Joint 4 Bias  (elbow offset, rad)")
bias_var = _row(root, "bias", -3.0, 0.5, BIAS4_DEFAULT, lambda v: f"{v:+.3f}")

ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=6, pady=(6, 2))

# ── Periodic stdout write ──────────────────────────────────────────────────────
def send():
    print(json.dumps({
        "speed": speed_var.get(),
        "A":     [v.get() for v in amp_vars],
        "bias4": bias_var.get(),
    }), flush=True)
    root.after(50, send)

root.after(50, send)
root.mainloop()
