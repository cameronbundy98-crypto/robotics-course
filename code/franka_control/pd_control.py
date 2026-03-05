"""
Franka PD Joint Control Demo
=============================
Runs a PD controller that drives the Franka robot from its home (zero) pose
to a target joint configuration.

Two spheres are overlaid on the scene via the viewer's user scene:
  - Cyan sphere   : current end-effector position (moves as the robot moves)
  - Orange sphere : target end-effector position  (fixed at the goal pose)

Usage:
    python3 pd_control.py
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Load model ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model = mujoco.MjModel.from_xml_path(os.path.join(SCRIPT_DIR, "franka_scene.xml"))
data  = mujoco.MjData(model)

# ── Target joint configuration ────────────────────────────────────────────────
# Joint limits:
#   j1 ∈ [-2.90,  2.90]   j2 ∈ [-1.76,  1.76]   j3 ∈ [-2.90,  2.90]
#   j4 ∈ [-3.07, -0.30]   j5 ∈ [-2.90,  2.90]   j6 ∈ [-0.90,  0.90]
#   j7 ∈ [-0.90,  0.90]
q_target = np.array([-2.89, -0.4, 0.0, -1.8, 0.0, 1.4, 0.0])
q_target = np.array([0.4  ,0.8  ,0.0 ,0.0  , 0.0, 0.0, 0.0 ])

# ── Compute target end-effector position via forward kinematics ───────────────
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

data_fk = mujoco.MjData(model)
data_fk.qpos[:7] = q_target
mujoco.mj_forward(model, data_fk)
ee_target_pos = data_fk.site_xpos[ee_id].copy()

print(f"Target joint angles : {q_target}")
print(f"Target EE position  : {ee_target_pos}")

# ── PD gains (torque limits: joints 1-4 → 87 Nm, joints 5-7 → 12 Nm) ─────────
Kp = np.array([200.0, 200.0, 200.0, 200.0,  50.0,  50.0,  50.0])
Kd = np.array([ 30.0,  30.0,  30.0,  30.0,  10.0,  10.0,  10.0])
ctrl_limits = model.actuator_ctrlrange.copy()  # (7, 2)

def pd_torques(q, qd):
    """Compute PD torques clipped to actuator limits."""
    tau = Kp * (q_target - q) - Kd * qd
    return np.clip(tau, ctrl_limits[:, 0], ctrl_limits[:, 1])

# ── Helper: write a sphere into a user-scene slot ─────────────────────────────
def add_sphere(scene, slot, pos, rgba, radius=0.04):
    mujoco.mjv_initGeom(
        scene.geoms[slot],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.geoms[slot].size[0] = radius
    scene.geoms[slot].pos[:] = pos

# ── Main simulation loop ───────────────────────────────────────────────────────
dt = model.opt.timestep  # 0.002 s

with mujoco.viewer.launch_passive(
    model, data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    try:
        while viewer.is_running():
            step_start = time.time()

            # Apply PD control
            q  = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            data.ctrl[:] = pd_torques(q, qd)

            mujoco.mj_step(model, data)

            # Get current end-effector position
            ee_curr_pos = data.site_xpos[ee_id].copy()
            print(" target: ", ee_target_pos ,". Actual: ", ee_curr_pos )

            # Draw both EE markers in the viewer overlay
            with viewer.lock():
                add_sphere(viewer.user_scn, 0, ee_curr_pos,   [0.0, 0.8, 1.0, 0.9])  # cyan
                add_sphere(viewer.user_scn, 1, ee_target_pos, [1.0, 0.5, 0.0, 0.9])  # orange
                viewer.user_scn.ngeom = 2

            viewer.sync()

            # Keep real-time pace
            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
