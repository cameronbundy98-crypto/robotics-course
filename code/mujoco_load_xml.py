import mujoco
import mujoco.viewer
import time 

# Load a model from a file (won't do this here to avoid external file calls)
model = mujoco.MjModel.from_xml_path('../models/simple_pendulum_act.xml')

data = mujoco.MjData(model)
data.qpos[0]=0.01

# Create a viewer
viewer = mujoco.viewer.launch(model, data, 
        show_left_ui=False,
        show_right_ui=False,
)
