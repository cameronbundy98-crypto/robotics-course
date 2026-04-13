import mujoco
import numpy as np
import time
from fsm import State, ArmFSM

from matplotlib import pyplot as plt

import mediapy as media

## Simulation time step
dt: float = 0.002

## For saving animation
SAVE_ANIMATION = True # Set false to avoid delay while testing
animation_framerate=30
animation_save_path = "."

def main():
    model = mujoco.MjModel.from_xml_path("robosuite_model.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    fsm = ArmFSM(model,data)

    def mycallback(model, data):
        fsm.update(model, data)

    fsm.callback = mycallback
    mujoco.set_mjcb_control(mycallback)
    frames=[]
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:

            # Reset the free camera.
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("frontview").id
            # If you want the default view, uncomment:
            # mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Enable site frame visualization.
            # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()

                mujoco.mj_step(model, data)

                if SAVE_ANIMATION:
                    # Capture frame
                    renderer.update_scene(data, camera="frontview")
                    frames.append(renderer.render().copy())  # .copy() is important

                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    if SAVE_ANIMATION:
        with media.set_show_save_dir(animation_save_path):
            media.show_video(frames, fps=animation_framerate, border=True, loop=True, title="Simulation")


if __name__ == "__main__":
    main()
