# Assignment 3 (ME/AER 676)

## Goal 

Demonstrate a loop of grasping the red block using the Franka arm, lifting it to the home position, and then dropping it onto the table. 


## Model and Sim

The main file to run is `franka_grasp_sim.py` which uses `fsm.py` for control involving a finite state machine (FSM). It loads the typical `model` and `data` objects from the XML, instantiates the FSM controller, and then runs the simulation. 

The `robosuite_model.xml` file defines the Franka robot with gripper, its mount, a table, and a block to be grasped. 

*Franka*: Has a `site` called `grasp_site` located between fingertips. The franka joint angles are in `data.qpos[:(model.nu-2)]` and its joint velocities are in `data.qvel[:(model.nu-2)]`.

*Gripper*: The gripper positions are in `data.qpos[(model.nu-2):model.nu]` and its joint velocities are in `data.qvel[(model.nu-2):model.nu]`.

*Block*: Name of the body is `cube_main`, useful for acquiring its pose.

*Controls*: The first seven actuators are the joint torques (rotational) for the Franka arm, the last two are the motor controllers (linear) for the fingers. 


## Finite State Machine (FSM)

Currently, there are five states (HOME, GRASP_OPEN,APPROACH, GRASP_CLOSE, LIFT). 

One way to use the FSM is to execute a sequence HOME $\to$ GRASP_OPEN $\to$ APPROACH $\to$ GRASP_CLOSE $\to$ LIFT $\to$ HOME, closing the loop.  This sequence has been defined in the `transition` function. 

The controller uses the `update` function, which

1. Points to which controller function to use (achieved by setting `data.ctrl` appropriate values in that function).
1. Points to which function to use to check if the finite state needs to transition, and perhaps some functions to execute when transitioning.

You can define your own FSM structure too. 

## Hints 

- Most modifications will occur in the `fsm.py` file.
- You should define continuous controllers that will reliably reach poses/conditions that trigger changing the FSM state, thereby moving on to the next 'subtask' in the sequence. 
- The arm torques can be accessed as `data.ctrl[:(model.nu-2)]` and the two finger motors as `data.ctrl[(model.nu-2):model.nu]`.
- What controller can get the Franka to a target fixed joint configuration reliably?
- What controller can get some frame on the Franka to a target fixed 3D pose  reliably?
- Gripper open corresponds to `np.array([0.04,-0.04])` and closed corresponds to `np.array([0.0,0.0])`.
- Some transitions may involve changing set-points for the joint angles and finger positions appropriately.
- See `Simulation.mp4` for a visualization


## Submission:

Submit your `fsm.py` solution file as `fsm_<lastname>.py`. I will include it using `import fsm_<lastname> import State, ArmFSM` and test your solution using the same `franka_grasp_sim.py` and `robosuite_model.xml` files. 

See `Simulation.mp4` for an example of the loop working. 
