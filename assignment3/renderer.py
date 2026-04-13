import mujoco
import mujoco.viewer
import time 
import argparse
import numpy as np

# Load a model from a file (won't do this here to avoid external file calls)
def main(args):
    model = mujoco.MjModel.from_xml_path(args.filename)

    data = mujoco.MjData(model)
    print("actuators: ", model.nu)
    # for i in range(0,model.nq):
    #     data.qpos[i]=0.0


    # Create a viewer. Use `mjpython` on `macOS` due to use of non-blocking `launch_passive` instead of blocking `launch`
    viewer = mujoco.viewer.launch_passive(model, data, 
            show_left_ui=False,
            show_right_ui=False,
    )
    if args.frames:
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY # Enables visualization of body frames
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # Enables visualization of body frames
        viewer.opt.sitegroup[:]=1
        viewer.opt.sitegroup[1]=1
        # You can also enable other visual aids like joint axes or contact forces
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        print("lookat:", viewer.cam.lookat)
        print("distance:", viewer.cam.distance)
        print("azimuth:", viewer.cam.azimuth)
        print("elevation:", viewer.cam.elevation)
        az = np.radians(viewer.cam.azimuth)
        el = np.radians(viewer.cam.elevation)
        d = viewer.cam.distance
        lookat = viewer.cam.lookat

        # Camera position in world frame
        pos = lookat + d * np.array([
            np.cos(el) * np.sin(az),
            -np.cos(el) * np.cos(az),
            np.sin(el)
        ])

        print(f'<camera name="my_cam" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
              f'lookat="{lookat[0]:.3f} {lookat[1]:.3f} {lookat[2]:.3f}"/>')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a path planning solution")
    parser.add_argument("filename", type=str, default = "franka_scene.xml", help="xml file name to render")
    parser.add_argument("--frames",type=bool, default = False)
    args = parser.parse_args()
    main(args)
