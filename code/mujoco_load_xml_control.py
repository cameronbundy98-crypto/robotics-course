import mujoco
import mujoco.viewer
import time 

# Load a model from a file (won't do this here to avoid external file calls)
model = mujoco.MjModel.from_xml_path('../models/simple_pendulum_act.xml')


dt: float = 0.002
model.opt.timestep = dt
data = mujoco.MjData(model)
data.qpos[0]=0.01

# Create a viewer
viewer = mujoco.viewer.launch_passive(model, data,
        show_left_ui=False,
        show_right_ui=False,
      )
try:
    while viewer.is_running():
        step_start = time.time()
        # Step the simulation
        data.ctrl[0] = -100*data.qpos[0] - 20*data.qvel[0]
        print(data.qpos[0])
        mujoco.mj_step(model, data)
        # Update the viewer
        viewer.sync()
        # Add a small delay to control frame rate
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
except KeyboardInterrupt:
    print("Viewer closed by user")
finally:
    viewer.close()
