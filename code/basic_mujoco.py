import mujoco
import mujoco.viewer
import time 

# Load a model from a file (won't do this here to avoid external file calls)
# model = mujoco.MjModel.from_xml_path('path/to/your/model.xml')

# Create a simple pendulum model
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  
  <worldbody>
    <light name="light1" pos="0.3 0.3 1"/>
    <geom name="ground" type="plane" pos=" 0 0 -1.2" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    
    <body name="pendulum" pos="0 0 0">
      <joint name="pivot" type="hinge" pos="0 0 0" axis="1 0 0" damping="0.1"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05" rgba="0.5 0.5 0.5 1"/>
      <geom name="mass" type="sphere" pos="0 0 0.6" size="0.07" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)
data.qpos[0]=0.01

# Create a viewer
viewer = mujoco.viewer.launch(model, data, 
        show_left_ui=False,
        show_right_ui=False,
)
