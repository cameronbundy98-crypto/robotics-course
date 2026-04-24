import heapq
import numpy as np

# ============================================================
# Grid / world helpers (2D maps)
# ============================================================

def world_to_grid(xy, origin, res, grid_shape):
    x, y = float(xy[0]), float(xy[1])
    x0, y0 = float(origin[0]), float(origin[1])
    j = int(np.round((x - x0) / res))
    i = int(np.round((y - y0) / res))
    i = int(np.clip(i, 0, grid_shape[0] - 1))
    j = int(np.clip(j, 0, grid_shape[1] - 1))
    return i, j

def grid_to_world(ij, origin, res):
    i, j = int(ij[0]), int(ij[1])
    x0, y0 = float(origin[0]), float(origin[1])
    x = x0 + j * res
    y = y0 + i * res
    return np.array([x, y], dtype=float)

def is_free_cell(grid, ij):
    i, j = ij
    return grid[i, j] == 0

# ============================================================
# Part 1: A* (8-connected)
# ============================================================

def astar_8_connected(grid, origin, res, start_xy, goal_xy):
    H, W = grid.shape
    start = world_to_grid(start_xy, origin, res, grid.shape)
    goal  = world_to_grid(goal_xy,  origin, res, grid.shape)

    if not is_free_cell(grid, start) or not is_free_cell(grid, goal):
        return None

    nbrs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            cost = np.sqrt(2.0) if (di != 0 and dj != 0) else 1.0
            nbrs.append((di, dj, cost))

    def h(ij):
        # Octile distance heuristic
        i, j = ij
        gi, gj = goal
        dx = abs(j - gj)
        dy = abs(i - gi)
        return (dx + dy) + (np.sqrt(2.0) - 2.0) * min(dx, dy)

    open_heap = []
    heapq.heappush(open_heap, (h(start), 0.0, start))
    came_from = {start: None}
    gscore = {start: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path_ij = []
            node = cur
            while node is not None:
                path_ij.append(node)
                node = came_from[node]
            path_ij.reverse()
            return np.vstack([grid_to_world(p, origin, res) for p in path_ij])

        for di, dj, step_cost in nbrs:
            ni, nj = cur[0] + di, cur[1] + dj
            if ni < 0 or ni >= H or nj < 0 or nj >= W:
                continue
            if grid[ni, nj] == 1:
                continue
            if di != 0 and dj != 0:
                if grid[cur[0] + di, cur[1]] == 1:
                    continue
                if grid[cur[0], cur[1] + dj] ==1:
                    continue

            nxt = (ni, nj)

            tentative = gscore[cur] + step_cost
            if (nxt not in gscore) or (tentative < gscore[nxt]):
                gscore[nxt] = tentative
                came_from[nxt] = cur
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))

    return None

# ============================================================
# Generic RRT* core
# ============================================================

class Node:
    __slots__ = ("q", "parent", "cost")
    def __init__(self, q, parent, cost):
        self.q = np.array(q, dtype=float)
        self.parent = int(parent)
        self.cost = float(cost)

def _reconstruct(nodes, idx):
    path = []
    while idx != -1:
        path.append(nodes[idx].q)
        idx = nodes[idx].parent
    path.reverse()
    return np.vstack(path)

def rrt_star(
    q_start,
    q_goal,
    sample_fn,            # (rng) -> q
    collision_free_fn,    # (q) -> bool
    edge_free_fn,         # (qa, qb) -> bool
    dist_fn,              # (qa, qb) -> float
    steer_fn,             # (q_from, q_to, step) -> q_new
    step_size,
    goal_thresh,
    max_iters=20000,
    goal_bias=0.05,
    dim=None,
    gamma=2.0,
    r_max=None,
    rng=None,
    stop_on_first_solution=True,   # ✅ NEW: fast mode
):
    """
    RRT*:
      - choose best parent among neighbors (min total cost)
      - rewire neighbors through new node if cheaper

    If stop_on_first_solution=True, returns immediately when it first
    connects to the goal (still does RRT* logic up to that point).
    """
    rng = np.random.default_rng() if rng is None else rng
    q_start = np.array(q_start, dtype=float)
    q_goal  = np.array(q_goal, dtype=float)
    if dim is None:
        dim = int(q_start.shape[0])

    if not collision_free_fn(q_start) or not collision_free_fn(q_goal):
        return None

    nodes = [Node(q_start, parent=-1, cost=0.0)]
    best_goal_idx = None
    best_goal_cost = np.inf

    for it in range(1, max_iters + 1):
        # sample
        q_rand = q_goal if (rng.random() < goal_bias) else np.array(sample_fn(rng), dtype=float)

        # nearest
        d_to_rand = np.array([dist_fn(n.q, q_rand) for n in nodes], dtype=float)
        near_idx = int(np.argmin(d_to_rand))
        q_near = nodes[near_idx].q

        # steer
        q_new = np.array(steer_fn(q_near, q_rand, step_size), dtype=float)
        if not collision_free_fn(q_new):
            continue
        if not edge_free_fn(q_near, q_new):
            continue

        # neighbor radius with practical floor
        n = len(nodes) + 1
        r = gamma * (np.log(n) / n) ** (1.0 / max(dim, 1))
        r = max(r, 2.0 * step_size)
        if r_max is not None:
            r = min(r, float(r_max))

        # near set
        d_to_new = np.array([dist_fn(n_.q, q_new) for n_ in nodes], dtype=float)
        near_set = np.where(d_to_new <= r)[0]
        if near_set.size == 0:
            near_set = np.array([near_idx], dtype=int)

        # choose best parent among near_set
        best_parent = near_idx
        best_cost = nodes[near_idx].cost + dist_fn(q_near, q_new)
        for j in near_set:
            j = int(j)
            qj = nodes[j].q
            cand = nodes[j].cost + dist_fn(qj, q_new)
            if cand < best_cost:
                if edge_free_fn(qj, q_new):
                    best_cost = cand
                    best_parent = j

        # add node
        nodes.append(Node(q_new, parent=best_parent, cost=best_cost))
        new_idx = len(nodes) - 1

        # rewire neighbors through q_new if cheaper
        for j in near_set:
            j = int(j)
            if j == best_parent:
                continue
            qj = nodes[j].q
            cand = nodes[new_idx].cost + dist_fn(q_new, qj)
            if cand + 1e-12 < nodes[j].cost:
                if edge_free_fn(q_new, qj):
                    nodes[j].parent = new_idx
                    nodes[j].cost = float(cand)

        # goal connection check
        if dist_fn(q_new, q_goal) <= goal_thresh and edge_free_fn(q_new, q_goal):
            cand_goal_cost = nodes[new_idx].cost + dist_fn(q_new, q_goal)
            if cand_goal_cost < best_goal_cost:
                nodes.append(Node(q_goal.copy(), parent=new_idx, cost=cand_goal_cost))
                best_goal_idx = len(nodes) - 1
                best_goal_cost = cand_goal_cost

                # ✅ NEW: stop early (fast)
                if stop_on_first_solution:
                    return _reconstruct(nodes, best_goal_idx)

    if best_goal_idx is None:
        return None
    return _reconstruct(nodes, best_goal_idx)

# ============================================================
# 2D RRT* wrapper that matches evaluator.py segment sampling
# ============================================================

def _world_to_grid_eval(point, origin, resolution):
    col = int(round((point[0] - origin[0]) / resolution))
    row = int(round((point[1] - origin[1]) / resolution))
    return row, col

def _edge_free_map_like_evaluator(qa, qb, grid, origin, res):
    # matches evaluator.check_segment_collision spacing: res * 0.5
    dist = float(np.linalg.norm(qb - qa))
    if dist < 1e-12:
        r, c = _world_to_grid_eval(qa, origin, res)
        H, W = grid.shape
        return (0 <= r < H) and (0 <= c < W) and (grid[r, c] == 0)

    n_samples = max(int(np.ceil(dist / (res * 0.5))), 2)
    H, W = grid.shape
    for t in np.linspace(0.0, 1.0, n_samples):
        pt = qa + t * (qb - qa)
        r, c = _world_to_grid_eval(pt, origin, res)
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        if grid[r, c] == 1:
            return False
    return True

def rrt_star_2d(
    grid, origin, res, start, goal,
    step_size=0.5,
    goal_thresh=0.5,
    max_iters=20000,
    goal_bias=0.05,
    gamma=2.0,
    r_max=2.0,
    stop_on_first_solution=True
):
    grid = np.array(grid)
    origin = np.array(origin, dtype=float)
    start = np.array(start, dtype=float)
    goal  = np.array(goal, dtype=float)
    H, W = grid.shape

    x0, y0 = float(origin[0]), float(origin[1])
    x_min, x_max = x0, x0 + (W - 1) * res
    y_min, y_max = y0, y0 + (H - 1) * res

    def sample_fn(rng):
        return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

    def collision_free_fn(q):
        r, c = _world_to_grid_eval(q, origin, res)
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        return grid[r, c] == 0

    def edge_free_fn(a, b):
        return _edge_free_map_like_evaluator(np.array(a, float), np.array(b, float), grid, origin, res)

    def dist_fn(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def steer_fn(q_from, q_to, step):
        v = np.array(q_to) - np.array(q_from)
        d = np.linalg.norm(v)
        if d <= step:
            return q_to
        return np.array(q_from) + (step / d) * v

    return rrt_star(
        q_start=start,
        q_goal=goal,
        sample_fn=sample_fn,
        collision_free_fn=collision_free_fn,
        edge_free_fn=edge_free_fn,
        dist_fn=dist_fn,
        steer_fn=steer_fn,
        step_size=step_size,
        goal_thresh=goal_thresh,
        max_iters=max_iters,
        goal_bias=goal_bias,
        dim=2,
        gamma=gamma,
        r_max=r_max,
        stop_on_first_solution=stop_on_first_solution
    )

# ============================================================
# Franka RRT* wrapper using your franka_utils functions
# ============================================================

def rrt_star_franka(
    model, data, q_start, q_goal,
    step_size=0.25,
    goal_thresh=0.25,
    max_iters=50000,
    goal_bias=0.10,
    gamma=2.0,
    r_max=1.0,
    n_edge_checks=20,
    stop_on_first_solution=True
):
    from franka_utils import check_collision, check_edge, get_joint_limits

    q_start = np.array(q_start, dtype=float)
    q_goal  = np.array(q_goal, dtype=float)
    lower, upper = get_joint_limits(model)

    def sample_fn(rng):
        return rng.uniform(lower, upper)

    def collision_free_fn(q):
        return not check_collision(model, data, q)

    def edge_free_fn(a, b):
        return check_edge(model, data, np.array(a, float), np.array(b, float), n_checks=n_edge_checks)

    def dist_fn(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def steer_fn(q_from, q_to, step):
        v = np.array(q_to) - np.array(q_from)
        d = np.linalg.norm(v)
        if d <= step:
            return q_to
        return np.array(q_from) + (step / d) * v

    return rrt_star(
        q_start=q_start,
        q_goal=q_goal,
        sample_fn=sample_fn,
        collision_free_fn=collision_free_fn,
        edge_free_fn=edge_free_fn,
        dist_fn=dist_fn,
        steer_fn=steer_fn,
        step_size=step_size,
        goal_thresh=goal_thresh,
        max_iters=max_iters,
        goal_bias=goal_bias,
        dim=7,
        gamma=gamma,
        r_max=r_max,
        stop_on_first_solution=stop_on_first_solution
    )
