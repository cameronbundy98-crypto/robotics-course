"""
Franka Task-Space Trajectory Tracking
======================================
The end-effector tracks a Cartesian circle.  Two controllers are available:

  invdyn  — task-space inverse dynamics (acceleration-level, full dynamics)
  diffik     — differential IK (velocity-level, joint PD + gravity compensation)
  diffik_pos — differential IK, position-only (no velocity feedforward)

invdyn controller
-----------------
  τ = M(q)q̈_cmd + C(q,q̇) + G(q)       (solved by mj_inverse)

  q̈_cmd = J⁺(ẍ_des − J̇q̇) + N q̈_null

  ẍ_des  = ẍ_ref + Kp(x_ref − x) + Kd(ẋ_ref − ẋ)
  J̇q̇    ≈ (J_t − J_{t−Δt})/Δt · q̇          (finite difference)
  J⁺     = Jᵀ(JJᵀ + λI)⁻¹                   (damped pseudo-inverse)
  N      = I − J⁺J                           (null-space projector)
  q̈_null = Kp_ns(q_home − q) − Kd_ns q̇     (joint centering)

diffik controller
-----------------
  q̇_des = J⁺ ẋ_des + N Kv_ns(q_home − q)   (null-space velocity centering)
  q_des += q̇_des · Δt                        (integrate to joint positions)

  ẋ_des = ẋ_ref + Kp_v(x_ref − x)           (proportional + feedforward)
  τ = Kp_j(q_des − q) + Kd_j(q̇_des − q̇) + G(q)

diffik_pos controller
---------------------
  Same as diffik but no velocity feedforward — pure proportional IK:
  ẋ_des = Kp_v(x_ref − x)
  Equivalent to solving IK toward the instantaneous target and sending
  the result to a joint position controller.

Usage
-----
    mjpython taskspace_tracking.py invdyn
    mjpython taskspace_tracking.py diffik
    mjpython taskspace_tracking.py diffik_pos
    mjpython taskspace_tracking.py invdyn --no-bias
    mjpython taskspace_tracking.py invdyn --no-nullspace
    mjpython taskspace_tracking.py diffik --no-nullspace
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

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("controller", choices=["invdyn", "diffik", "diffik_pos"],
                    help="invdyn: task-space inverse dynamics | diffik: differential IK")
parser.add_argument("--no-bias", action="store_true",
                    help="(invdyn only) skip J̇q̇ bias term")
parser.add_argument("--no-nullspace", action="store_true",
                    help="Disable null-space joint-centering")
args = parser.parse_args()
CONTROLLER    = args.controller
USE_BIAS      = not args.no_bias
USE_NULLSPACE = not args.no_nullspace
print(f"Controller: {CONTROLLER}  |  Bias: {USE_BIAS}  |  Null-space: {USE_NULLSPACE}")

# ── Model ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model      = mujoco.MjModel.from_xml_path(os.path.join(SCRIPT_DIR, "franka_scene.xml"))
data       = mujoco.MjData(model)

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
nv    = model.nv   # 7 for the Franka arm (no floating base)

ctrl_limits = model.actuator_ctrlrange.copy()

# ── Home configuration ─────────────────────────────────────────────────────────
# Franka "ready" pose: elbow bent, clear of kinematic singularities
q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# ── Gains ─────────────────────────────────────────────────────────────────────

# invdyn: task-space PD on Cartesian acceleration error
Kp    = 400.0 * np.eye(3)   # N/m
Kd    =  40.0 * np.eye(3)   # N·s/m

# invdyn: null-space joint centering (acceleration level)
Kp_ns = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])   # 1/s²
Kd_ns = np.array([ 2.0,  2.0,  2.0,  2.0, 1.0, 1.0, 1.0])   # 1/s

# diffik: proportional gain on Cartesian position error → desired velocity
Kp_v  = 2.0 * np.eye(3)    # 1/s  (converts position error [m] to velocity [m/s])

# diffik: joint PD gains for tracking the integrated joint reference
Kp_j  = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 200.0, 200.0])   # N·m/rad
Kd_j  = np.array([ 40.0,  40.0,  40.0,  40.0,  20.0,  20.0,  20.0])   # N·m·s/rad

# diffik: null-space velocity centering
Kv_ns = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])   # 1/s

# Damped pseudo-inverse (shared): J⁺ = Jᵀ(JJᵀ + λI)⁻¹
LAMBDA = 1e-4

# ── Circle reference trajectory ────────────────────────────────────────────────
RADIUS = 0.12   # m
OMEGA  = 3.0    # rad/s  (~10.5 s per revolution)

# Place the circle centre at the home end-effector position.
# The circle lies in the Y–Z plane (lateral × vertical).
data.qpos[:nv] = q_home
mujoco.mj_forward(model, data)
x_center = data.site_xpos[ee_id].copy()
print(f"Circle centre (EE at home): {x_center.round(4)}")

def reference(t):
    """Returns (x_ref, xd_ref, xdd_ref) — position, velocity, acceleration."""
    c, s = np.cos(OMEGA * t), np.sin(OMEGA * t)

    x_ref = x_center.copy()
    x_ref[1] += RADIUS * c         # Y
    x_ref[2] += RADIUS * s         # Z

    xd_ref    = np.zeros(3)
    xd_ref[1] = -RADIUS * OMEGA * s
    xd_ref[2] =  RADIUS * OMEGA * c

    xdd_ref    = np.zeros(3)
    xdd_ref[1] = -RADIUS * OMEGA**2 * c
    xdd_ref[2] = -RADIUS * OMEGA**2 * s

    return x_ref, xd_ref, xdd_ref

# ── Spawn live-plot subprocess ─────────────────────────────────────────────────
_bin     = os.path.dirname(os.path.abspath(sys.executable))
_python3 = next(
    (p for p in [os.path.join(_bin, "python3"),
                 os.path.join(_bin, "python3.13")]
     if os.path.exists(p)),
    shutil.which("python3") or "python3",
)
plot_proc = subprocess.Popen(
    [_python3, os.path.join(SCRIPT_DIR, "taskspace_plot.py")],
    stdin=subprocess.PIPE, text=True, bufsize=1,
)

# ── Viewer overlay helpers ─────────────────────────────────────────────────────
def add_sphere(scene, slot, pos, rgba, radius=0.025):
    mujoco.mjv_initGeom(
        scene.geoms[slot],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.geoms[slot].size[0] = radius
    scene.geoms[slot].pos[:]  = pos

# ── Initialise robot at home pose ──────────────────────────────────────────────
data.qpos[:nv] = q_home
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# ── Pre-allocate arrays ────────────────────────────────────────────────────────
J_pos  = np.zeros((3, nv))   # positional Jacobian (filled by mj_jacSite)
J_rot  = np.zeros((3, nv))   # rotational Jacobian (required by mj_jacSite, unused)
J_prev = np.zeros((3, nv))   # previous-step Jacobian for finite-difference J̇

# diffik: integrated joint position reference (initialised to home)
q_des = q_home.copy()

# ── Simulation loop ────────────────────────────────────────────────────────────
dt         = model.opt.timestep   # 0.002 s
PLOT_EVERY = 25                   # → ~20 Hz to the plot process

with mujoco.viewer.launch_passive(
    model, data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    try:
        step  = 0
        t_sim = 0.0

        while viewer.is_running():
            if plot_proc.poll() is not None:
                break

            step_start = time.time()

            # ── Current state ──────────────────────────────────────────────
            q  = data.qpos[:nv].copy()
            qd = data.qvel[:nv].copy()
            x  = data.site_xpos[ee_id].copy()

            # ── Positional Jacobian  J ∈ ℝ^{3×7} ──────────────────────────
            mujoco.mj_jacSite(model, data, J_pos, J_rot, ee_id)
            J = J_pos.copy()

            # ── Damped pseudo-inverse  J⁺ = Jᵀ(JJᵀ + λI)⁻¹ ──────────────
            J_pinv = J.T @ np.linalg.solve(J @ J.T + LAMBDA * np.eye(3), np.eye(3))

            # ── Reference ─────────────────────────────────────────────────
            x_ref, xd_ref, xdd_ref = reference(t_sim)

            # ──────────────────────────────────────────────────────────────
            if CONTROLLER == "invdyn":

                # End-effector velocity
                xd = J @ qd

                # J̇q̇ via finite difference of J
                if USE_BIAS and step > 0:
                    bias = ((J - J_prev) / dt) @ qd
                else:
                    bias = np.zeros(3)

                # Desired Cartesian acceleration
                xdd_des = xdd_ref + Kp @ (x_ref - x) + Kd @ (xd_ref - xd)

                # Map to joint acceleration
                qacc_cmd = J_pinv @ (xdd_des - bias)

                # Null-space: joint centering at acceleration level
                if USE_NULLSPACE:
                    N        = np.eye(nv) - J_pinv @ J
                    qacc_ns  = Kp_ns * (q_home - q) - Kd_ns * qd
                    qacc_cmd = qacc_cmd + N @ qacc_ns

                # Inverse dynamics: find τ such that M q̈ + C + G = τ
                data.qacc[:nv] = qacc_cmd
                mujoco.mj_inverse(model, data)
                tau = data.qfrc_inverse[:nv].copy()

            else:  # diffik or diffik_pos

                # Desired end-effector velocity
                if CONTROLLER == "diffik":
                    xd_des = xd_ref + Kp_v @ (x_ref - x)  # feedforward + proportional
                else:  # diffik_pos: pure proportional, no feedforward
                    xd_des = Kp_v @ (x_ref - x)

                # Map to joint velocity
                qd_des = J_pinv @ xd_des

                # Null-space: velocity-level joint centering
                if USE_NULLSPACE:
                    N      = np.eye(nv) - J_pinv @ J
                    qd_des = qd_des + N @ (Kv_ns * (q_home - q))

                # Integrate to get the joint position reference
                q_des += qd_des * dt

                # Gravity compensation: evaluate G(q) with q̇ = 0
                qd_saved = data.qvel[:nv].copy()
                data.qvel[:nv] = 0.0
                mujoco.mj_forward(model, data)
                Gq = data.qfrc_bias[:nv].copy()
                data.qvel[:nv] = qd_saved
                mujoco.mj_forward(model, data)   # restore kinematics

                # Joint PD + gravity compensation
                tau = Kp_j * (q_des - q) + Kd_j * (qd_des - qd) + Gq

            # ── Apply torques ──────────────────────────────────────────────
            J_prev[:] = J   # always update for finite-difference (invdyn)
            data.ctrl[:] = np.clip(tau, ctrl_limits[:, 0], ctrl_limits[:, 1])
            mujoco.mj_step(model, data)
            t_sim += dt
            step  += 1

            # ── Viewer overlay: current EE (cyan) and target (orange) ──────
            with viewer.lock():
                add_sphere(viewer.user_scn, 0, x,     [0.0, 0.8, 1.0, 0.9])
                add_sphere(viewer.user_scn, 1, x_ref, [1.0, 0.5, 0.0, 0.9])
                viewer.user_scn.ngeom = 2
            viewer.sync()

            # ── Send data to plot process ──────────────────────────────────
            if step % PLOT_EVERY == 0:
                msg = json.dumps({
                    "t":      round(t_sim, 4),
                    "x":      x.tolist(),
                    "x_ref":  x_ref.tolist(),
                    "center": x_center.tolist(),
                })
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
