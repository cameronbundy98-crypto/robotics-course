"""
Franka PD Control with Interactive Joint Sliders
=================================================
Runs a PD controller for the Franka robot. A separate tkinter window
provides 7 sliders to set target joint angles in real time.

Two spheres are drawn in the MuJoCo viewer:
  - Cyan   : current end-effector position
  - Orange : target end-effector position (updates as sliders move)

Architecture
------------
  mjpython (main process) : MuJoCo viewer + simulation loop
  python3  (subprocess)   : tkinter slider UI  (slider_ui.py)
  communication           : subprocess stdout → JSON lines

The split is necessary because mjpython owns the macOS Cocoa main thread,
so no other GUI framework can run inside the same process.

Usage:
    mjpython pd_control_sliders.py
"""

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

# ── Model ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model   = mujoco.MjModel.from_xml_path(os.path.join(SCRIPT_DIR, "franka_scene.xml"))
data    = mujoco.MjData(model)
data_fk = mujoco.MjData(model)

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

# ── PD controller ─────────────────────────────────────────────────────────────
Kp = np.array([200.0, 200.0, 200.0, 200.0,  50.0,  50.0,  50.0])
Kd = np.array([ 30.0,  30.0,  30.0,  30.0,  10.0,  10.0,  10.0])
ctrl_limits = model.actuator_ctrlrange.copy()

# ── Find python3 (not mjpython) ───────────────────────────────────────────────
# mjpython lives alongside python3 in the venv's bin directory.
_bin = os.path.dirname(os.path.abspath(sys.executable))
_python3 = next(
    (p for p in [os.path.join(_bin, "python3"),
                 os.path.join(_bin, "python3.13")]
     if os.path.exists(p)),
    shutil.which("python3") or "python3",
)

# ── Spawn the slider UI as a regular python3 subprocess ───────────────────────
_ui_script = os.path.join(SCRIPT_DIR, "slider_ui.py")
ui_proc = subprocess.Popen(
    [_python3, _ui_script],
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)

# Shared joint-target array, updated by a reader thread
q_target = np.zeros(7)
_lock = threading.Lock()

def _read_ui():
    for line in ui_proc.stdout:
        try:
            vals = json.loads(line)
            with _lock:
                q_target[:] = vals
        except (json.JSONDecodeError, ValueError):
            pass

threading.Thread(target=_read_ui, daemon=True).start()

# ── Viewer overlay helper ──────────────────────────────────────────────────────
def add_sphere(scene, slot, pos, rgba, radius=0.04):
    mujoco.mjv_initGeom(
        scene.geoms[slot],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.geoms[slot].size[0] = radius
    scene.geoms[slot].pos[:] = pos

# ── Simulation loop ────────────────────────────────────────────────────────────
dt = model.opt.timestep

with mujoco.viewer.launch_passive(
    model, data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    try:
        while viewer.is_running():
            # Stop if the slider window was closed
            if ui_proc.poll() is not None:
                break

            step_start = time.time()

            with _lock:
                q_t = q_target.copy()

            # FK to get target EE position
            data_fk.qpos[:7] = q_t
            mujoco.mj_forward(model, data_fk)
            ee_target_pos = data_fk.site_xpos[ee_id].copy()

            # PD step
            q  = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            tau = Kp * (q_t - q) - Kd * qd
            data.ctrl[:] = np.clip(tau, ctrl_limits[:, 0], ctrl_limits[:, 1])
            mujoco.mj_step(model, data)

            ee_curr_pos = data.site_xpos[ee_id].copy()
            print(" target: ", ee_target_pos ,". Actual: ", ee_curr_pos )

            with viewer.lock():
                add_sphere(viewer.user_scn, 0, ee_curr_pos,   [0.0, 0.8, 1.0, 0.9])  # cyan
                add_sphere(viewer.user_scn, 1, ee_target_pos, [1.0, 0.5, 0.0, 0.9])  # orange
                viewer.user_scn.ngeom = 2
            viewer.sync()

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        ui_proc.terminate()
