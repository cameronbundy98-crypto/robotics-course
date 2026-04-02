"""
Live task-space tracking plot.
Spawned as a subprocess by taskspace_tracking.py.
Reads JSON lines from stdin.

Left panel  — actual EE path vs reference circle (Y–Z plane)
Right panels — X, Y, Z position error components and error norm over time
"""

import json
import sys
import threading
from collections import deque

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np

MAXPTS  = 500     # history depth; controls how much of the time plots are visible

# ── Shared data ────────────────────────────────────────────────────────────────
times  = deque(maxlen=MAXPTS)
x_act  = deque(maxlen=MAXPTS)   # each entry: [x, y, z]
x_ref  = deque(maxlen=MAXPTS)
center = None
_lock  = threading.Lock()
stdin_open = True

def _read_stdin():
    global stdin_open, center
    for line in sys.stdin:
        try:
            d = json.loads(line)
            with _lock:
                times.append(d["t"])
                x_act.append(d["x"])
                x_ref.append(d["x_ref"])
                if center is None:
                    center = d["center"]
        except (json.JSONDecodeError, KeyError):
            pass
    stdin_open = False

threading.Thread(target=_read_stdin, daemon=True).start()

# ── Figure ─────────────────────────────────────────────────────────────────────
BG    = "#1a1a1a"
AXBG  = "#252525"
CACT  = "#4fc3f7"   # cyan  – actual
CREF  = "#ff8a65"   # coral – reference
CTXT  = "#cccccc"
CGRID = "#383838"

fig = plt.figure(figsize=(11, 7), facecolor=BG)
fig.suptitle("Task-Space Trajectory Tracking", color=CTXT, fontsize=12)

gs = fig.add_gridspec(4, 2, left=0.09, right=0.97, top=0.93, bottom=0.08,
                      hspace=0.55, wspace=0.38)

ax_traj = fig.add_subplot(gs[:, 0])   # left: 2-D EE path in Y–Z plane
ax_ex   = fig.add_subplot(gs[0, 1])
ax_ey   = fig.add_subplot(gs[1, 1], sharex=ax_ex)
ax_ez   = fig.add_subplot(gs[2, 1], sharex=ax_ex)
ax_en   = fig.add_subplot(gs[3, 1], sharex=ax_ex)
time_axes = [ax_ex, ax_ey, ax_ez, ax_en]

for ax in [ax_traj] + time_axes:
    ax.set_facecolor(AXBG)
    ax.tick_params(colors=CTXT, labelsize=8)
    ax.grid(True, color=CGRID, lw=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

# Left panel: trajectory
ax_traj.set_aspect("equal")
ax_traj.set_title("EE path (Y–Z plane)", color=CTXT, fontsize=9)
ax_traj.set_xlabel("Y  (m)", color=CTXT, fontsize=8)
ax_traj.set_ylabel("Z  (m)", color=CTXT, fontsize=8)
ref_circle_line, = ax_traj.plot([], [], color=CREF, lw=1.5, ls="--", label="reference")
act_trail,       = ax_traj.plot([], [], color=CACT, lw=1.5,          label="actual")
act_dot,         = ax_traj.plot([], [], "o", color=CACT, ms=7, zorder=5)
ref_dot,         = ax_traj.plot([], [], "o", color=CREF, ms=7, zorder=5)
ax_traj.legend(loc="upper right", fontsize=8, facecolor="#333",
               labelcolor=CTXT, edgecolor="#555")

# Right panels: error over time
for ax, lbl in zip(time_axes, ["eₓ (m)", "e_y (m)", "e_z (m)", "‖e‖ (m)"]):
    ax.set_ylabel(lbl, color=CTXT, fontsize=8, rotation=0,
                  ha="right", labelpad=48)
    # zero reference line for component plots
    if lbl != "‖e‖ (m)":
        ax.axhline(0, color="#555", lw=0.8, ls="--")
time_axes[-1].set_xlabel("time  (s)", color=CTXT, fontsize=8)

err_lines = [ax.plot([], [], color=CACT, lw=1.2)[0] for ax in time_axes]

# ── Plot loop ──────────────────────────────────────────────────────────────────
plt.show(block=False)
plt.pause(0.1)

_circle_drawn = False

try:
    while plt.fignum_exists(fig.number) and stdin_open:
        with _lock:
            n = len(times)
            if n < 2:
                plt.pause(0.05)
                continue
            t_snap  = np.array(list(times))
            xa_snap = np.array(list(x_act))    # (N, 3)
            xr_snap = np.array(list(x_ref))
            ctr     = np.array(center) if center is not None else None

        # Draw reference circle once, as soon as we know the centre
        if not _circle_drawn and ctr is not None:
            r = np.linalg.norm(xr_snap[0, 1:] - ctr[1:])   # infer from first point
            theta = np.linspace(0, 2 * np.pi, 300)
            ref_circle_line.set_data(ctr[1] + r * np.cos(theta),
                                     ctr[2] + r * np.sin(theta))
            pad = r + 0.04
            ax_traj.set_xlim(ctr[1] - pad, ctr[1] + pad)
            ax_traj.set_ylim(ctr[2] - pad, ctr[2] + pad)
            _circle_drawn = True

        # Trajectory panel
        act_trail.set_data(xa_snap[:, 1], xa_snap[:, 2])
        act_dot.set_data([xa_snap[-1, 1]], [xa_snap[-1, 2]])
        ref_dot.set_data([xr_snap[-1, 1]], [xr_snap[-1, 2]])

        # Time-domain error panels
        t_now = t_snap[-1]
        t_min = t_snap[0]    # oldest point in the deque; MAXPTS controls depth
        mask  = np.ones(len(t_snap), dtype=bool)
        t_win = t_snap[mask]
        err   = xa_snap[mask] - xr_snap[mask]   # (N, 3)
        enorm = np.linalg.norm(err, axis=1)

        for i in range(3):
            err_lines[i].set_data(t_win, err[:, i])
        err_lines[3].set_data(t_win, enorm)

        for i, ax in enumerate(time_axes):
            ax.set_xlim(t_min, t_now + 0.2)
            vals = err[:, i] if i < 3 else enorm
            if len(vals):
                lo, hi = vals.min(), vals.max()
                pad = max(0.002, 0.15 * (hi - lo))
                ax.set_ylim(lo - pad, hi + pad)

        fig.canvas.draw_idle()
        plt.pause(0.05)

except KeyboardInterrupt:
    pass
