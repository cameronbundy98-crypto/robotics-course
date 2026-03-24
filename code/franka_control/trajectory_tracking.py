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
    mjpython trajectory_tracking.py pid
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Controller selection ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("controller", choices=["pd", "pdplus", "invdyn", "pid"],
                    help="pd: PD only | pdplus: PD + gravity | invdyn: inverse dynamics | pid: PD + integral")
args = parser.parse_args()
CONTROLLER = args.controller
print(f"Controller: {CONTROLLER}")

# ── Model ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model   = mujoco.MjModel.from_xml_path(os.path.join(SCRIPT_DIR, "franka_scene.xml"))
data    = mujoco.MjData(model)
data_fk = mujoco.MjData(model)

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

# ── PD / PID gains ────────────────────────────────────────────────────────────
Kp = np.array([200.0, 200.0, 200.0, 200.0,  50.0,  50.0,  50.0])*0.5
Kd = np.array([ 30.0,  30.0,  30.0,  30.0,  10.0,  10.0,  10.0])*0.5
Ki = 1.00*np.array([  5.0,   5.0,   5.0,   5.0,   2.0,   2.0,   2.0])
ctrl_limits = model.actuator_ctrlrange.copy()

# ── Reference trajectory parameters ───────────────────────────────────────────
#                j1     j2     j3     j4     j5     j6     j7
A    = np.array([0.80,  0.40,  0.50,  0.80,  0.50,  0.30,  0.50])  # amplitude (rad)
W    = np.array([2.50,  1.50,  2.00,  1.25,  3.00,  2.00,  1.75])  # angular frequency (rad/s)
W    = np.array([2.50,  1.50,  2.00,  1.25,  3.00,  2.00,  1.75])  # angular frequency (rad/s)
PHI  = np.array([0.00,  0.00,  0.50,  0.00,  1.00,  0.30,  0.70])  # phase (rad)
BIAS = np.array([0.00,  0.00,  0.00, -1.57,  0.00,  0.00,  0.00])  # constant offset (rad)

# ── Live parameters (updated by traj_param_ui.py slider subprocess) ───────────
_traj_lock = threading.Lock()
_live = {"A": A.copy(), "speed": 1.0, "bias4": float(BIAS[3])}

# Sinusoidal trajectory: q_i(t) = A_i*sin(W_i*t + PHI_i) + BIAS_i
# Derivatives follow analytically; joint 4 is offset to keep the elbow bent.
def reference(t):
    with _traj_lock:
        A_l = _live["A"].copy()
        W_l = W * _live["speed"]
        B_l = BIAS.copy()
        B_l[3] = _live["bias4"]
    theta = W_l * t + PHI
    return (
        A_l * np.sin(theta) + B_l,
        A_l * W_l * np.cos(theta),
       -A_l * W_l**2 * np.sin(theta),
    )

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

param_proc = subprocess.Popen(
    [_python3, os.path.join(SCRIPT_DIR, "traj_param_ui.py")],
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)

def _read_params():
    for line in param_proc.stdout:
        try:
            d = json.loads(line)
            with _traj_lock:
                _live["speed"] = d["speed"]
                _live["A"][:] = d["A"]
                _live["bias4"] = d["bias4"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

threading.Thread(target=_read_params, daemon=True).start()

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
        step        = 0
        t_sim       = 0.0
        integral_e  = np.zeros(7)   # accumulated position error for PID

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
            elif CONTROLLER == "pid":
                integral_e += (q_ref - q) * dt
                tau = Kp * (q_ref - q) + Ki * integral_e + Kd * (qd_ref - qd)
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
        param_proc.terminate()
