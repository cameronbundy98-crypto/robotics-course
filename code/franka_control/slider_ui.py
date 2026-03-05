"""
Slider UI for Franka PD control.
Spawned as a subprocess by pd_control_sliders.py.
Writes current joint target values to stdout as a JSON line every 50 ms.
"""

import json
import math
import sys
import tkinter as tk
from tkinter import ttk

JOINT_LIMITS = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718,  3.000),
    (-2.8973,  2.8973),
    (-0.8973,  0.8973),
    (-2.8973,  2.8973),
]

root = tk.Tk()
root.title("Franka — Joint Targets")
root.resizable(False, False)

SLIDER_W = 300

# ── Header ─────────────────────────────────────────────────────────────────────
hdr = ttk.Frame(root, padding=(8, 6, 8, 2))
hdr.pack(fill=tk.X)
ttk.Label(hdr, text="Joint",    width=8,           anchor="w").grid(row=0, column=0)
ttk.Label(hdr, text="Min",      width=6,           anchor="e").grid(row=0, column=1)
ttk.Label(hdr, text="Target",   width=SLIDER_W//8, anchor="center").grid(row=0, column=2)
ttk.Label(hdr, text="Max",      width=6,           anchor="w").grid(row=0, column=3)
ttk.Label(hdr, text="Value",    width=8,           anchor="center").grid(row=0, column=4)

ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=6)

# ── Sliders ────────────────────────────────────────────────────────────────────
slider_vars = []

for i, (lo, hi) in enumerate(JOINT_LIMITS):
    row = ttk.Frame(root, padding=(8, 2))
    row.pack(fill=tk.X)

    ttk.Label(row, text=f"Joint {i+1}", width=8, anchor="w").grid(row=0, column=0)
    ttk.Label(row, text=f"{math.degrees(lo):.0f}°", width=6, anchor="e").grid(row=0, column=1)

    var = tk.DoubleVar(value=0.0)
    ttk.Scale(row, from_=lo, to=hi, variable=var,
              orient=tk.HORIZONTAL, length=SLIDER_W).grid(row=0, column=2, padx=4)

    ttk.Label(row, text=f"{math.degrees(hi):.0f}°", width=6, anchor="w").grid(row=0, column=3)

    val_lbl = ttk.Label(row, text=" 0.0°", width=8, anchor="center")
    val_lbl.grid(row=0, column=4)

    def _update(name, index, op, lbl=val_lbl, v=var):
        lbl.config(text=f"{math.degrees(v.get()):+6.1f}°")

    var.trace_add("write", _update)
    slider_vars.append(var)

# ── Periodic stdout write ──────────────────────────────────────────────────────
def send():
    vals = [v.get() for v in slider_vars]
    print(json.dumps(vals), flush=True)
    root.after(50, send)

root.after(50, send)
root.mainloop()
