"""
Live trajectory tracking plot.
Spawned as a subprocess by trajectory_tracking.py.
Reads JSON lines from stdin and shows a moving-window plot of
actual vs reference joint angles for all 7 joints.
"""

import json
import sys
import threading
from collections import deque

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np

N_JOINTS   = 7
WINDOW     = 10.0   # seconds of history visible
MAXPOINTS  = 600    # deque capacity (~30 s at 20 Hz)

# ── Shared data (written by reader thread, read by plot thread) ────────────────
times = deque(maxlen=MAXPOINTS)
q_act = [deque(maxlen=MAXPOINTS) for _ in range(N_JOINTS)]
q_ref = [deque(maxlen=MAXPOINTS) for _ in range(N_JOINTS)]
q_norm = deque(maxlen=MAXPOINTS)
stdin_open = True

def _read_stdin():
    global stdin_open
    for line in sys.stdin:
        try:
            d = json.loads(line)
            times.append(d["t"])
            for i in range(N_JOINTS):
                q_act[i].append(d["q"][i])
                q_ref[i].append(d["qr"][i])
            q_norm.append(np.linalg.norm(np.array(d["q"]) - np.array(d["qr"])))
        except (json.JSONDecodeError, KeyError):
            pass
    stdin_open = False

threading.Thread(target=_read_stdin, daemon=True).start()

# ── Figure ─────────────────────────────────────────────────────────────────────
BG       = "#1a1a1a"
AX_BG    = "#252525"
C_ACT    = "#4fc3f7"   # cyan  – actual
C_REF    = "#ff8a65"   # coral – reference
C_TEXT   = "#cccccc"
C_GRID   = "#383838"

fig, axes = plt.subplots(N_JOINTS+1, 1, figsize=(8, 9), sharex=True)
fig.patch.set_facecolor(BG)
fig.suptitle("Joint-Space Trajectory Tracking", color=C_TEXT, fontsize=11, y=0.995)

lines_act, lines_ref = [], []

for i, ax in enumerate(axes):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    ax.grid(True, color=C_GRID, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    if i<N_JOINTS:
        ax.set_ylabel(f"j{i+1} (rad)", color=C_TEXT, fontsize=7.5,
                      rotation=0, ha="right", labelpad=40)

        la, = ax.plot([], [], color=C_ACT, lw=1.3, label="actual")
        lr, = ax.plot([], [], color=C_REF, lw=1.3, ls="--", label="reference")
        lines_act.append(la)
        lines_ref.append(lr)
    else:
        ax.set_ylabel(f"norm (rad)", color=C_TEXT, fontsize=7.5,
                      rotation=0, ha="right", labelpad=40)
        la, = ax.plot([], [], color=C_ACT, lw=1.3, label="norm")
        lines_act.append(la)

    if i == 0:
        ax.legend(loc="upper right", fontsize=7,
                  facecolor="#333", labelcolor=C_TEXT, edgecolor="#555")

axes[-1].set_xlabel("time  (s)", color=C_TEXT, fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.993])

# ── Plot loop ──────────────────────────────────────────────────────────────────
plt.show(block=False)
plt.pause(0.1)

try:
    while plt.fignum_exists(fig.number) and stdin_open:
        if len(times) < 2:
            plt.pause(0.05)
            continue

        t_arr  = np.array(times)
        t_now  = t_arr[-1]
        t_min  = t_now - WINDOW
        mask   = t_arr >= t_min
        t_win  = t_arr[mask]

        for i in range(N_JOINTS):
            a = np.array(q_act[i])[mask]
            r = np.array(q_ref[i])[mask]
            lines_act[i].set_data(t_win, a)
            lines_ref[i].set_data(t_win, r)
            axes[i].set_xlim(t_min, t_now + 0.2)
            # y-limits: span of both signals + 10 % padding
            all_v = np.concatenate([a, r])
            if len(all_v):
                lo, hi = all_v.min(), all_v.max()
                pad = max(0.05, 0.12 * (hi - lo))
                axes[i].set_ylim(lo - pad, hi + pad)

        a = np.array(q_norm)[mask]
        lines_act[N_JOINTS].set_data(t_win, a)
        axes[N_JOINTS].set_xlim(t_min, t_now + 0.2)
        if len(a):
            lo, hi = a.min(), a.max()
            pad = max(0.005, 0.1 * (hi - lo))
            axes[N_JOINTS].set_ylim(0 - pad, hi + pad)
        fig.canvas.draw_idle()
        plt.pause(0.05)   # ~20 fps; also processes window events

except KeyboardInterrupt:
    pass
