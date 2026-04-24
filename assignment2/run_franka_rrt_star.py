import argparse
import numpy as np

from franka_utils import load_scene, load_problems
from planners import rrt_star_franka


def solve_one(model, data, problems, num,
              step_size=0.25, goal_thresh=0.25,
              max_iters=60000, goal_bias=0.10,
              gamma=2.0, r_max=1.0, n_edge_checks=20):
    q_start, q_goal = problems[num - 1]

    path = rrt_star_franka(
        model, data, q_start, q_goal,
        step_size=step_size,
        goal_thresh=goal_thresh,
        max_iters=max_iters,
        goal_bias=goal_bias,
        gamma=gamma,
        r_max=r_max,
        n_edge_checks=n_edge_checks,
        stop_on_first_solution=True
    )

    if path is None:
        print(f"[WARN] No path found for problem {num}")
        path = np.zeros((0, 7), dtype=float)

    out_file = f"franka_{num}_path.npz"
    np.savez(out_file, path=path)
    print(f"Saved {out_file} | waypoints={len(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Solve all Franka problems")
    ap.add_argument("num", type=int, nargs="?", default=1, help="Problem number (1-indexed)")
    args = ap.parse_args()

    model, data = load_scene()
    problems = load_problems()
    n = len(problems)

    if args.all:
        print(f"Solving all Franka problems: 1..{n}")
        for k in range(1, n + 1):
            solve_one(model, data, problems, k)
    else:
        if args.num < 1 or args.num > n:
            print(f"Problem {args.num} not found. Available: 1..{n}")
            return
        solve_one(model, data, problems, args.num)


if __name__ == "__main__":
    main()
