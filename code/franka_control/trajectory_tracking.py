"""
Franka Joint-Space Trajectory Tracking
========================================
A controller tracks a periodic sinusoidal reference in joint space.
A live plot window (spawned as a subprocess) shows actual vs reference
joint angles over a moving time window.

Usage:
    mjpython trajectory_tracking.py pd
    mjpython trajectory_tracking.py pdplus
    mjpython trajectory_tracking.py invdyn
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Controller selection ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("controller", choices=["pd", "pdplus", "invdyn"],
                    help="pd: PD only | pdplus: PD + gravity | invdyn: inverse dynamics")
args = parser.parse_args()
CONTROLLER = args.controller
print(f"Controller: {CONTROLLER}")

# ── Model ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model   = mujoco.MjModel.from_xml_path(os.path.join(SCRIPT_DIR, "franka_scene.xml"))
data    = mujoco.MjData(model)
data_fk = mujoco.MjData(model)

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

# ── PD gains ──────────────────────────────────────────────────────────────────
Kp = np.array([200.0, 200.0, 200.0, 200.0,  50.0,  50.0,  50.0])*0.5
Kd = np.array([ 30.0,  30.0,  30.0,  30.0,  10.0,  10.0,  10.0])*0.5
ctrl_limits = model.actuator_ctrlrange.copy()

# ── Reference trajectory ───────────────────────────────────────────────────────
# Sinusoidal trajectory; joint 4 is offset to keep the elbow bent.
def reference(t):
    return np.array([
        0.8  * np.sin(0.50 * 5 * t),
        0.4  * np.sin(0.30 * 5 * t),
        0.5  * np.sin(0.40 * 5 * t + 0.50),
        0.8  * np.sin(0.25 * 5 * t)-1.5,   # stays in [-2.3, -0.7]
        0.5  * np.sin(0.60 * 5 * t + 1.00),
        0.3  * np.sin(0.40 * 5 * t + 0.30),
        0.5  * np.sin(0.35 * 5 * t + 0.70),
    ]), np.array([
        0.8  * 0.50 * 5 * np.cos(0.50 * 5 * t),
        0.4  * 0.30 * 5 * np.cos(0.30 * 5 * t),
        0.5  * 0.40 * 5 * np.cos(0.40 * 5 * t + 0.50),
        0.8  * 0.25 * 5 * np.cos(0.25 * 5 * t),   # stays in [-2.3, -0.7]
        0.5  * 0.60 * 5 * np.cos(0.60 * 5 * t + 1.00),
        0.3  * 0.40 * 5 * np.cos(0.40 * 5 * t + 0.30),
        0.5  * 0.35 * 5 * np.cos(0.35 * 5 * t + 0.70),
    ]), np.array([
        - 0.8  * 0.50 * 5 * 0.50 * 5 * np.sin(0.50 * 5 * t),
        - 0.4  * 0.30 * 5 * 0.30 * 5 * np.sin(0.30 * 5 * t),
        - 0.5  * 0.40 * 5 * 0.40 * 5 * np.sin(0.40 * 5 * t + 0.50),
        - 0.8  * 0.25 * 5 * 0.25 * 5 * np.sin(0.25 * 5 * t),   # stays in [-2.3, -0.7]
        - 0.5  * 0.60 * 5 * 0.60 * 5 * np.sin(0.60 * 5 * t + 1.00),
        - 0.3  * 0.40 * 5 * 0.40 * 5 * np.sin(0.40 * 5 * t + 0.30),
        - 0.5  * 0.35 * 5 * 0.35 * 5 * np.sin(0.35 * 5 * t + 0.70),
    ])

# ── Spawn live-plot subprocess ─────────────────────────────────────────────────
_bin = os.path.dirname(os.path.abspath(sys.executable))
_python3 = next(
    (p for p in [os.path.join(_bin, "python3"),
                 os.path.join(_bin, "python3.13")]
     if os.path.exists(p)),
    shutil.which("python3") or "python3",
)

plot_proc = subprocess.Popen(
    [_python3, os.path.join(SCRIPT_DIR, "trajectory_plot.py")],
    stdin=subprocess.PIPE,
    text=True,
    bufsize=1,
)

# ── Viewer overlay helpers ─────────────────────────────────────────────────────
def add_sphere(scene, slot, pos, rgba, radius=0.03):
    mujoco.mjv_initGeom(
        scene.geoms[slot],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.geoms[slot].size[0] = radius
    scene.geoms[slot].pos[:] = pos

# ── Initialise robot at trajectory start to avoid a sudden jerk ───────────────
data.qpos[:7], _ ,_= reference(0.0)
mujoco.mj_forward(model, data)

# ── Simulation loop ────────────────────────────────────────────────────────────
dt         = model.opt.timestep   # 0.002 s
PLOT_EVERY = 250                   # send data every 25 steps → ~20 Hz to plot

with mujoco.viewer.launch_passive(
    model, data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    try:
        step  = 0
        t_sim = 0.0

        while viewer.is_running():
            if plot_proc.poll() is not None:   # plot window was closed
                break

            step_start = time.time()

            # Calculate P error:
            q_ref, qd_ref, qdd = reference(t_sim)
            q     = data.qpos[:7].copy()
            qd    = data.qvel[:7].copy()

            data.qvel*=0 # zero the velocities
            mujoco.mj_forward(model,data) # propagate forward pass
            data.qvel = qd.copy() # restore velocities
            Gq = data.qfrc_bias # = C(q,qdot)+ G = G when qdot=0

            if CONTROLLER == "pd":
                tau = Kp * (q_ref - q) + Kd * (qd_ref - qd)
            elif CONTROLLER == "pdplus":
                tau = Kp * (q_ref - q) + Kd * (qd_ref - qd) + Gq
            else:  # invdyn
                data.qacc = Kp * (q_ref - q) + Kd * (qd_ref - qd) + qdd
                mujoco.mj_inverse(model, data)
                tau = data.qfrc_inverse

            data.ctrl[:] = np.clip(tau, ctrl_limits[:, 0], ctrl_limits[:, 1])
            mujoco.mj_step(model, data)
            t_sim += dt
            step  += 1

            # Viewer overlay: current EE (cyan) and reference EE (orange)
            data_fk.qpos[:7] = q_ref
            mujoco.mj_forward(model, data_fk)
            ee_ref_pos = data_fk.site_xpos[ee_id].copy()
            ee_cur_pos = data.site_xpos[ee_id].copy()

            with viewer.lock():
                add_sphere(viewer.user_scn, 0, ee_cur_pos, [0.0, 0.8, 1.0, 0.9])
                add_sphere(viewer.user_scn, 1, ee_ref_pos, [1.0, 0.5, 0.0, 0.9])
                viewer.user_scn.ngeom = 2
            viewer.sync()

            # Send snapshot to plot process
            if step % PLOT_EVERY == 0:
                msg = json.dumps({"t": round(t_sim, 4),
                                  "q":  q.tolist(),
                                  "qr": q_ref.tolist()})
                try:
                    print(msg, file=plot_proc.stdin, flush=True)
                except BrokenPipeError:
                    break

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        plot_proc.terminate()
