import glob
import os
import numpy as np
from planners import rrt_star_2d

def solve_one(map_file):
    data = np.load(map_file)
    grid = data["grid"]
    origin = data["origin"]
    res = float(data["resolution"])
    start = data["start"]
    goal = data["goal"]

    path = rrt_star_2d(
        grid, origin, res, start, goal,
        step_size=0.5,
        goal_thresh=0.5,
        max_iters=20000,
        goal_bias=0.05,
        gamma=2.0,
        r_max=2.0,
        stop_on_first_solution=True
    )

    if path is None:
        print(f"[WARN] No path found for {map_file}")
        path = np.zeros((0, 2), dtype=float)

    # map_2.npz -> "2"
    base = os.path.basename(map_file)
    N = base.split("_")[1].split(".")[0]
    out_file = f"map_{N}_rrt.npz"
    np.savez(out_file, path=path)
    print(f"Saved {out_file} | waypoints={len(path)}")

def main():
    maps = sorted(glob.glob("map_*.npz"))
    if not maps:
        print("No map_*.npz files found in this folder.")
        return

    for mf in maps:
        solve_one(mf)

if __name__ == "__main__":
    main()
