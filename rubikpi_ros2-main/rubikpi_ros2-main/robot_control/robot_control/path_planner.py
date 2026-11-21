"""HW4 path planning (no ROS). Two planners on the same map:

  safe:  grid + Voronoi diagram + A*. Each grid cell gets its distance to every obstacle (4 walls
         and the box). Cells that are equally far from two different obstacles are the Voronoi
         diagram: the "ridge" through the middle of the free space. A* searches along it.
  short: visibility graph + A*. Nodes are the start, the goal and the 4 corners of the grown box.
         Two nodes are connected if the straight line between them is clear. A* finds the
         shortest route.

Obstacles are grown by the robot's radius + a margin (configuration space), so the robot can be
treated as a single point.

Run on the laptop to make the report pictures:
  python3 path_planner.py <out_dir> [run_dir ...]
(each run_dir is a ~/hw4_runs/<path>_<time> folder; its driven path is drawn on top)
"""

import heapq
import math
import os
import sys

import numpy as np
import yaml

GRID = 0.02            # grid cell size for the safe planner (2 cm)
SIMPLIFY = 0.03        # waypoints stay within 3 cm of the grid path


def load_map(path):
    with open(path) as f:
        return yaml.safe_load(f)


def grown_box(m):
    """Box corners after growing it by the robot radius + margin: (xmin, ymin, xmax, ymax)."""
    half = m['box']['size'] / 2 + m['robot_radius'] + m['margin']
    return (m['box']['x'] - half, m['box']['y'] - half, m['box']['x'] + half, m['box']['y'] + half)


def astar(start, goal, neighbors, heuristic):
    """A*: always expand the node with the lowest cost so far + estimated cost to the goal."""
    frontier = [(heuristic(start), 0.0, start)]
    came_from = {start: None}
    cost = {start: 0.0}
    while frontier:
        _, c, node = heapq.heappop(frontier)
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]
        if c > cost[node]:
            continue  # already found a cheaper way here
        for nxt, step in neighbors(node):
            new_cost = c + step
            if new_cost < cost.get(nxt, math.inf):
                cost[nxt] = new_cost
                came_from[nxt] = node
                heapq.heappush(frontier, (new_cost + heuristic(nxt), new_cost, nxt))
    return None


# ---------------- Max safety: Voronoi diagram on a grid ----------------

def clearance_maps(m):
    """Distance from every grid cell to each obstacle (left, bottom, right, top wall, box)."""
    room = m['room']
    n = int(round(room / GRID)) + 1
    xs = np.arange(n) * GRID
    X, Y = np.meshgrid(xs, xs, indexing='ij')  # X[i, j] = x of cell (i, j)
    b = m['box']
    half = b['size'] / 2
    dx = np.maximum(np.abs(X - b['x']) - half, 0)
    dy = np.maximum(np.abs(Y - b['y']) - half, 0)
    d = np.stack([X, Y, room - X, room - Y, np.hypot(dx, dy)])
    return xs, d


def plan_safe(m):
    xs, d = clearance_maps(m)
    d_sorted = np.sort(d, axis=0)
    clearance = d_sorted[0]                      # distance to the closest obstacle
    free = clearance >= m['robot_radius'] + m['margin']
    # Voronoi: the closest and second-closest obstacles are (almost) equally far away
    voronoi = free & (d_sorted[1] - d_sorted[0] <= GRID)

    def to_cell(p):
        return (int(round(p[0] / GRID)), int(round(p[1] / GRID)))

    def nearest_voronoi(p):
        cells = np.argwhere(voronoi)
        k = np.argmin(np.hypot(cells[:, 0] * GRID - p[0], cells[:, 1] * GRID - p[1]))
        return tuple(cells[k])

    start = (m['start']['x'], m['start']['y'])
    goal = (m['goal']['x'], m['goal']['y'])
    a, z = nearest_voronoi(start), nearest_voronoi(goal)

    def neighbors(c):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    nc = (c[0] + di, c[1] + dj)
                    if 0 <= nc[0] < len(xs) and 0 <= nc[1] < len(xs) and voronoi[nc]:
                        yield nc, GRID * math.hypot(di, dj)

    cells = astar(a, z, neighbors, lambda c: GRID * math.hypot(c[0] - z[0], c[1] - z[1]))
    grid_path = [start] + [(c[0] * GRID, c[1] * GRID) for c in cells] + [goal]
    info = {'xs': xs, 'clearance': clearance, 'voronoi': voronoi, 'grid_path': grid_path}
    return simplify(grid_path, SIMPLIFY), info


def simplify(points, tolerance):
    """Douglas-Peucker: keep only the points needed to stay within `tolerance` of the path."""
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    length = math.hypot(x1 - x0, y1 - y0) or 1e-9
    dists = [abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / length for x, y in points[1:-1]]
    k = int(np.argmax(dists)) + 1
    if dists[k - 1] <= tolerance:
        return [points[0], points[-1]]
    return simplify(points[:k + 1], tolerance)[:-1] + simplify(points[k:], tolerance)


# ---------------- Min distance: visibility graph ----------------

def segment_clear(p, q, box, room, inset):
    """True if the line p->q stays out of the grown box and inside the room (checked every 1 cm)."""
    steps = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.01))
    eps = 1e-6
    for k in range(steps + 1):
        t = k / steps
        x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        if box[0] + eps < x < box[2] - eps and box[1] + eps < y < box[3] - eps:
            return False
        if not (inset - eps <= x <= room - inset + eps and inset - eps <= y <= room - inset + eps):
            return False
    return True


def plan_short(m):
    box = grown_box(m)
    inset = m['robot_radius'] + m['margin']
    start = (m['start']['x'], m['start']['y'])
    goal = (m['goal']['x'], m['goal']['y'])
    corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    nodes = [start, goal] + corners
    edges = {p: [] for p in nodes}
    for i, p in enumerate(nodes):
        for q in nodes[i + 1:]:
            if segment_clear(p, q, box, m['room'], inset):
                length = math.hypot(q[0] - p[0], q[1] - p[1])
                edges[p].append((q, length))
                edges[q].append((p, length))
    path = astar(start, goal, lambda p: edges[p],
                 lambda p: math.hypot(goal[0] - p[0], goal[1] - p[1]))
    return path, {'nodes': nodes, 'edges': edges, 'box': box}


def plan(m, kind):
    """Waypoints [(x, y), ...] from start to goal for kind 'safe' or 'short'."""
    return (plan_safe if kind == 'safe' else plan_short)(m)[0]


def path_length(path):
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:]))


def min_clearance(m, path):
    """Smallest distance from the path to any wall or the box (checked every 1 cm)."""
    room, b = m['room'], m['box']
    half = b['size'] / 2
    best = math.inf
    for p, q in zip(path, path[1:]):
        steps = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.01))
        for k in range(steps + 1):
            x = p[0] + k / steps * (q[0] - p[0])
            y = p[1] + k / steps * (q[1] - p[1])
            box_d = math.hypot(max(abs(x - b['x']) - half, 0), max(abs(y - b['y']) - half, 0))
            best = min(best, x, y, room - x, room - y, box_d)
    return best


# ---------------- Pictures for the report ----------------

def draw(m, out_dir, runs=()):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    runs_by_kind = {}
    for run in runs:
        kind = 'safe' if os.path.basename(run.rstrip('/')).startswith('safe') else 'short'
        with open(os.path.join(run, 'trajectory.csv')) as f:
            rows = list(csv.DictReader(f))
        runs_by_kind.setdefault(kind, []).append(
            ([float(r['x']) for r in rows], [float(r['y']) for r in rows]))

    room, b = m['room'], m['box']
    safe_path, safe = plan_safe(m)
    short_path, short = plan_short(m)
    for kind, path in (('safe', safe_path), ('short', short_path)):
        fig, ax = plt.subplots(figsize=(7, 7))
        if kind == 'safe':
            im = ax.imshow(safe['clearance'].T, origin='lower', cmap='Greens',
                           extent=[0, room, 0, room], alpha=0.8)
            fig.colorbar(im, ax=ax, fraction=0.046, label='distance to nearest obstacle (m)')
            vi = np.argwhere(safe['voronoi'])
            ax.plot(vi[:, 0] * GRID, vi[:, 1] * GRID, '.', color='gray', ms=2,
                    label='Voronoi diagram')
            title = 'Max safety: Voronoi diagram + A*'
        else:
            gb = short['box']
            ax.add_patch(Rectangle((gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1], fill=False,
                                   ls='--', color='gray', label='box grown by robot size'))
            for p, nbrs in short['edges'].items():
                for q, _ in nbrs:
                    ax.plot([p[0], q[0]], [p[1], q[1]], color='lightgray', lw=1, zorder=1)
            ax.plot([], [], color='lightgray', label='visibility graph')
            title = 'Min distance: visibility graph + A*'
        inset = m['robot_radius'] + m['margin']
        ax.add_patch(Rectangle((inset, inset), room - 2 * inset, room - 2 * inset, fill=False,
                               ls=':', color='gray', label='walls grown by robot size'))
        half = b['size'] / 2
        ax.add_patch(Rectangle((b['x'] - half, b['y'] - half), b['size'], b['size'],
                               color='saddlebrown', label='box'))
        px, py = zip(*path)
        ax.plot(px, py, '-o', color='tab:blue', lw=2.5, ms=5, zorder=3,
                label=f'planned path ({path_length(path):.2f} m, '
                      f'closest {min_clearance(m, path) * 100:.0f} cm)')
        for k, (tx, ty) in enumerate(runs_by_kind.get(kind, [])):
            ax.plot(tx, ty, color='tab:red', lw=1.5, zorder=4,
                    label='driven (robot estimate)' if k == 0 else None)
        for tag_id, t in m['tags'].items():
            ax.plot(t['x'], t['y'], 's', color='black', ms=6)
            ax.annotate(str(tag_id), (t['x'], t['y']), textcoords='offset points',
                        xytext=(5, 5), fontsize=9)
        ax.plot(m['start']['x'], m['start']['y'], 'g*', ms=16, zorder=5, label='start')
        ax.plot(m['goal']['x'], m['goal']['y'], 'r*', ms=16, zorder=5, label='goal')
        ax.set_xlim(-0.1, room + 0.1)
        ax.set_ylim(-0.1, room + 0.1)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
        out = os.path.join(out_dir, f'{kind}_path.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'{kind}: {len(path)} waypoints, {path_length(path):.2f} m, '
              f'closest to an obstacle {min_clearance(m, path) * 100:.1f} cm -> {out}')
        for p in path:
            print(f'    ({p[0]:.3f}, {p[1]:.3f})')


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out_dir, exist_ok=True)
    draw(load_map(os.path.join(here, '..', 'config', 'hw4_map.yaml')), out_dir, sys.argv[2:])
