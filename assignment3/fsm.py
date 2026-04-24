from enum import Enum, auto
import mujoco
import numpy as np


class State(Enum):
    HOME = auto()
    GRASP_OPEN = auto()
    APPROACH = auto()
    GRASP_CLOSE = auto()
    LIFT = auto()


class ArmFSM:
    def __init__(self, model, data):
        self.state = State.HOME
        self.state_start_time = data.time

        self.n_arm = model.nu - 2
        self.site_id = model.site("grasp_site").id
        self.cube_body_id = model.body("cube_main").id

        self.q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853])
        self.q_grasp_bias = np.array([0.0, 0.35, 0.0, -1.95, 0.0, 2.25, 0.7853])

        self.gripper_open = np.array([0.04, -0.04])
        self.gripper_closed = np.array([0.0, 0.0])

        self.grasp_hold = self.q_home.copy()
        self.lift_start_z = data.body(self.cube_body_id).xpos[2]

        self.callback = None

    def transition(self, data=None):
        transitions = {
            State.HOME: State.GRASP_OPEN,
            State.GRASP_OPEN: State.APPROACH,
            State.APPROACH: State.GRASP_CLOSE,
            State.GRASP_CLOSE: State.LIFT,
            State.LIFT: State.HOME,
        }

        self.state = transitions[self.state]

        if data is not None:
            self.state_start_time = data.time

    def update(self, model, data):
        if self.state == State.HOME:
            self._home(model, data)
        elif self.state == State.GRASP_OPEN:
            self._grasp_open(model, data)
        elif self.state == State.APPROACH:
            self._approach(model, data)
        elif self.state == State.GRASP_CLOSE:
            self._grasp_close(model, data)
        elif self.state == State.LIFT:
            self._lift(model, data)

    def _home(self, model, data):
        self._homing_control(model, data)
        self._gripper_open(model, data)

        if self._near_home(model, data) and self._time_in_state(data) > 0.35:
            print("transition to GRASP_OPEN at", data.time)
            self.transition(data)

    def _grasp_open(self, model, data):
        self._hold_arm(model, data, self.q_home)
        self._gripper_open(model, data)

        if self._grasp_opened(model, data) and self._time_in_state(data) > 0.20:
            print("transition to APPROACH at", data.time)
            self.transition(data)

    def _approach(self, model, data):
        self._approach_control(model, data)
        self._gripper_open(model, data)

        if self._near_object(data):
            print("transition to GRASP_CLOSE at", data.time)
            self.grasp_hold = data.qpos[:self.n_arm].copy()
            self.transition(data)

    def _grasp_close(self, model, data):
        self._hold_arm(model, data, self.grasp_hold)
        self._gripper_close(model, data)

        if self._grasp_stable(model, data):
            print("transition to LIFT at", data.time)
            self.lift_start_z = data.body(self.cube_body_id).xpos[2]
            self.transition(data)

    def _lift(self, model, data):
        self._lift_control(model, data)
        self._gripper_close(model, data)

        if self._lifted(model, data):
            print("transition to HOME at", data.time)
            self.transition(data)

    def _near_home(self, model, data):
        q = data.qpos[:self.n_arm]
        qd = data.qvel[:self.n_arm]

        q_error = np.linalg.norm(self.q_home - q)
        qd_error = np.linalg.norm(qd)

        return q_error < 0.10 and qd_error < 0.50

    def _grasp_opened(self, model, data):
        fingers = data.qpos[self.n_arm:self.n_arm + 2]
        return np.linalg.norm(fingers - self.gripper_open) < 0.008

    def _near_object(self, data):
        cube_pos = data.body(self.cube_body_id).xpos.copy()
        ee_pos = data.site(self.site_id).xpos.copy()

        dx = cube_pos - ee_pos

        return np.linalg.norm(dx) < 0.035

    def _grasp_stable(self, model, data):
        return self._time_in_state(data) > 0.60

    def _lifted(self, model, data):
        cube_z = data.body(self.cube_body_id).xpos[2]
        waited = self._time_in_state(data) > 0.90

        cube_raised = cube_z > self.lift_start_z + 0.06
        arm_near_home = np.linalg.norm(self.q_home - data.qpos[:self.n_arm]) < 0.18

        return waited and (cube_raised or arm_near_home)

    def _homing_control(self, model, data):
        self._joint_pd_control(model, data, self.q_home)

    def _gripper_open(self, model, data):
        data.ctrl[self.n_arm:self.n_arm + 2] = self.gripper_open

    def _approach_control(self, model, data):
        cube_pos = data.body(self.cube_body_id).xpos.copy()

        target_pos = cube_pos + np.array([0.0, 0.0, 0.02])

        self._site_position_control(model, data, target_pos, q_bias=self.q_grasp_bias)

    def _gripper_close(self, model, data):
        data.ctrl[self.n_arm:self.n_arm + 2] = self.gripper_closed

    def _lift_control(self, model, data):
        self._joint_pd_control(model, data, self.q_home)

    def _time_in_state(self, data):
        return data.time - self.state_start_time

    def _hold_arm(self, model, data, q_target):
        self._joint_pd_control(model, data, q_target)

    def _joint_pd_control(self, model, data, q_target):
        q = data.qpos[:self.n_arm]
        qd = data.qvel[:self.n_arm]

        kp = np.array([90.0, 90.0, 80.0, 75.0, 50.0, 35.0, 25.0])
        kd = np.array([18.0, 18.0, 16.0, 14.0, 9.0, 7.0, 5.0])

        tau = kp * (q_target - q) - kd * qd + data.qfrc_bias[:self.n_arm]

        data.ctrl[:self.n_arm] = self._clip_arm_torque(model, tau)

    def _site_position_control(self, model, data, target_pos, q_bias=None):
        q = data.qpos[:self.n_arm]
        qd = data.qvel[:self.n_arm]

        ee_pos = data.site(self.site_id).xpos.copy()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, self.site_id)

        J = jacp[:, :self.n_arm]
        ee_vel = J @ qd

        pos_error = target_pos - ee_pos

        desired_force = 350.0 * pos_error - 30.0 * ee_vel
        desired_force = np.clip(desired_force, -40.0, 40.0)

        tau_task = J.T @ desired_force

        if q_bias is None:
            q_bias = self.q_grasp_bias

        tau_posture = 10.0 * (q_bias - q) - 2.5 * qd

        tau = tau_task + tau_posture + data.qfrc_bias[:self.n_arm]

        data.ctrl[:self.n_arm] = self._clip_arm_torque(model, tau)

    def _clip_arm_torque(self, model, tau):
        ctrl_min = model.actuator_ctrlrange[:self.n_arm, 0]
        ctrl_max = model.actuator_ctrlrange[:self.n_arm, 1]

        return np.clip(tau, ctrl_min, ctrl_max)

    def print_contacts(self, model, data):
        print("\nn contacts:", data.ncon)

        for i in range(data.ncon):
            con = data.contact[i]
            g1 = model.geom(con.geom1).name
            g2 = model.geom(con.geom2).name

            print("geoms:", g1, g2)

            if (
                g1 == "gripper0_right_finger1_pad_collision"
                or g1 == "gripper0_right_finger2_pad_collision"
                or g2 == "gripper0_right_finger1_pad_collision"
                or g2 == "gripper0_right_finger2_pad_collision"
            ):
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, forcetorque)
                print("force:", forcetorque[:3])

    def _gravity_compensation(self, model, data):
        return data.qfrc_bias[:model.nv].copy()
