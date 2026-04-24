import glob
import os
import numpy as np
from planners import astar_8_connected

def solve_one(map_file):
    data = np.load(map_file)
    grid = data["grid"]
    origin = data["origin"]
    res = float(data["resolution"])
    start = data["start"]
    goal = data["goal"]

    path = astar_8_connected(grid, origin, res, start, goal)
    if path is None:
        print(f"[WARN] No A* path found for {map_file}")
        path = np.zeros((0, 2), dtype=float)

    base = os.path.basename(map_file)
    N = base.split("_")[1].split(".")[0]
    out_file = f"map_{N}_astar.npz"
    np.savez(out_file, path=path)
    print(f"Saved {out_file} | waypoints={len(path)}")

def main():
    maps = sorted(glob.glob("map_*.npz"))
    maps = [m for m in maps if "_rrt" not in m and "_astar" not in m]
    if not maps:
        print("No original map_*.npz files found.")
        return

    for mf in maps:
        solve_one(mf)

if __name__ == "__main__":
    main()
