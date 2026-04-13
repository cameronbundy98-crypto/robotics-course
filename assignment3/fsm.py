"""
Current version achieves a loop for the franka+2-fingered gripper where it reaches HOME with closed gripper, then opens gripper while going to grab, then home while closing (picking up block), then open gripper (dropping gripper) and go-to-grab etc. 
"""
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
    def __init__(self,model,data):
        self.state = State.HOME
        # End-effector site we wish to control.
        self.site_id = model.site("grasp_site").id
        self.q_home = np.array([0,0,0,-1.57079,0,1.57079,0.7853]) # home position joint angles for Franka
        self.grasp_hold = self.q_home.copy() # some modes may use PD to target joint positions

    def transition(self):
        transitions = {
            State.HOME:    State.GRASP_OPEN,
            State.GRASP_OPEN:    State.APPROACH,
            State.APPROACH:    State.GRASP_CLOSE,
            State.GRASP_CLOSE: State.LIFT, 
            State.LIFT: State.HOME, #loops
            ## to terminate after lifting instead of looping, replace HOME with LIFT in line above
        }
        self.state = transitions[self.state]

    def update(self, model, data):
        if self.state == State.HOME:
            self._home(model, data)
        elif self.state == State.GRASP_OPEN:
            self._grasp_open(model, data)
        if self.state == State.APPROACH:
            self._approach(model, data)
        elif self.state == State.GRASP_CLOSE:
            self._grasp_close(model, data)
        elif self.state == State.LIFT: 
            self._lift(model, data)


    # --- per-state logic ---

    def _home(self, model, data):
        # set arm joint targets, check if near enough
        self._homing_control(model,data)
        if  self._near_home(model,data):
            print("transition to GRASP OPEN at ",data.time)
            self.transition()

    def _grasp_open(self, model, data):
        # open gripper fingers
        self._gripper_open(model,data) ## 
        if self._grasp_opened(model,data):
            print("transition to APPROACH at ",data.time)
            self.transition()

    def _approach(self, model, data):
        # set arm joint targets, check if near enough
        self._approach_control(model,data)
        if self._near_object(data):
            print("transition to GRASP CLOSE at ",data.time)
            self.grasp_hold = data.qpos[:(model.nu-2)].copy() ## hold joint values corresponding to IK solution
            self.transition()

    def _grasp_close(self, model, data):
        # close gripper fingers
        self._gripper_close(model,data) ## 
        if self._grasp_stable(model,data):
            print("transition to HOME at ",data.time)
            self.grasp_hold = self.q_home.copy()
            self.transition()

    def _lift(self, model, data):
        # move arm upward
        self._lift_control(model,data) ## 
        if self._lifted(model,data):
            self.transition()
        pass

    # --- transition conditions ---

    def _near_home(self, model,data):
        err =  np.linalg.norm(self.q_home - data.qpos[:(model.nu-2)]) 
        return err < 0.15

    def _grasp_opened(self,model, data):
        # e.g. check contact forces or finger position error is small
        return data.qpos[model.nu-2] > 0.039

    def _near_object(self, data):
        # e.g. check distance between end effector and object body
        dx = data.body("cube_main").xpos - data.site(self.site_id).xpos
        return np.linalg.norm(dx) < 0.005

    def _grasp_stable(self,model, data):
        # e.g. check contact forces or finger position error is small
        print("Create identifier for grasp completion")
        return False

    def _lifted(self,model, data):
        # e.g. check contact forces or finger position error is small
        print("Create identifier for lift completion")
        return False

    # --- utils including controllers for each mode ---

    def _homing_control(self,model,data):
        pass

    def _gripper_open(self,model,data):
        pass

    def _approach_control(self,model,data):
        pass

    def _gripper_close(self,model,data):
        pass

    def _lift_control(self,model,data):
        pass

    def print_contacts(self,model,data):
        print("\nn contacts:", data.ncon)
        for i in range(data.ncon):
            con = data.contact[i]
            # ID of geoms in contact
            print("geoms:", model.geom(con.geom1).name, model.geom( con.geom2).name)
            if ( model.geom( con.geom2).name == "gripper0_right_finger2_pad_collision" or  model.geom( con.geom1).name == "gripper0_right_finger2_pad_collision" or  model.geom( con.geom2).name == "gripper0_right_finger1_pad_collision" or  model.geom( con.geom1).name == "gripper0_right_finger1_pad_collision"):
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, forcetorque)
                contact_force_local = forcetorque[:3] # [fx, fy, fz] in contact frame
                print("force: ",contact_force_local)


    def _gravity_compensation(self,model,data):
        # Gravity compensation: evaluate G(q) with q̇ = 0
        nv = model.nv
        qd_saved = data.qvel[:nv].copy()
        data.qvel[:nv] = 0.0
        mujoco.set_mjcb_control(None) # deactivate callback to prevent recursion
        mujoco.mj_forward(model, data)
        Gq = data.qfrc_bias[:nv].copy()
        data.qvel[:nv] = qd_saved
        mujoco.mj_forward(model, data)   # restore kinematics
        mujoco.set_mjcb_control(self.callback) #restore callback
        return Gq
