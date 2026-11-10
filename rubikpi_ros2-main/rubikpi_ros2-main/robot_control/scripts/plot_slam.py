"""Plot a HW3 SLAM run and compare its map with the measured ground truth.

Usage (on your laptop, after copying the run folder off the robot):
    python plot_slam.py ~/hw3_runs/square_20261110_153000

The robot's map is in its own frame (start pose = origin). Using the measured start pose
from hw3_ground_truth.yaml, it is moved into the world frame and compared tag by tag.
Saves slam_map.png and errors.csv in the run folder.

For a noise test run (shape:=none) it prints how much each tag's measurement jitters.
"""

import csv
import glob
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Ellipse

run_dir = sys.argv[1]
truth_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'hw3_ground_truth.yaml')
shape = os.path.basename(os.path.normpath(run_dir)).split('_')[0]


def read_csv(name):
    with open(os.path.join(run_dir, name)) as f:
        return list(csv.DictReader(f))


# ---------------- Noise test: robot standing still ----------------
if shape == 'none':
    rows = read_csv('measurements.csv')
    for tag_id in sorted({int(r['tag_id']) for r in rows}):
        f = np.array([float(r['forward']) for r in rows if int(r['tag_id']) == tag_id])
        l = np.array([float(r['left']) for r in rows if int(r['tag_id']) == tag_id])
        print(f'tag {tag_id}: {len(f)} readings, distance {math.hypot(f.mean(), l.mean()):.2f} m, '
              f'std forward {f.std() * 100:.2f} cm, std left {l.std() * 100:.2f} cm')
    sys.exit()

# ---------------- Move the robot's map into the world frame ----------------
with open(truth_file) as f:
    truth = yaml.safe_load(f)
tags = {int(k): (v['x'], v['y']) for k, v in truth['tags'].items()}
start = truth['start'][shape]
c, s = math.cos(start['theta']), math.sin(start['theta'])
R = np.array([[c, -s], [s, c]])


def to_world(x, y):
    return start['x'] + c * x - s * y, start['y'] + s * x + c * y


def read_map(name):
    out = {}
    for r in read_csv(name):
        cov = np.array([[float(r['var_x']), float(r['cov_xy'])],
                        [float(r['cov_xy']), float(r['var_y'])]])
        out[int(r['tag_id'])] = (*to_world(float(r['x']), float(r['y'])), R @ cov @ R.T)
    return out


# ---------------- Landmark error after each lap ----------------
lap_files = sorted(glob.glob(os.path.join(run_dir, 'landmarks_lap*.csv')),
                   key=lambda p: int(re.search(r'lap(\d+)', p).group(1)))
results = []
for path in lap_files + [os.path.join(run_dir, 'landmarks_final.csv')]:
    name = os.path.basename(path)
    est = read_map(name)
    errors = {i: math.hypot(est[i][0] - tags[i][0], est[i][1] - tags[i][1])
              for i in est if i in tags}
    avg = sum(errors.values()) / len(errors) if errors else float('nan')
    results.append((name, len(est), avg, errors))
    print(f'{name:22} {len(est)} landmarks, average error {avg * 100:.1f} cm')

with open(os.path.join(run_dir, 'errors.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['map', 'num_landmarks', 'avg_error_m'] + [f'tag_{i}_error_m' for i in sorted(tags)])
    for name, n, avg, errors in results:
        writer.writerow([name, n, f'{avg:.4f}'] + [f'{errors[i]:.4f}' if i in errors else ''
                                                  for i in sorted(tags)])

# ---------------- Plot: trajectory + map with covariance ----------------
traj = read_csv('trajectory.csv')
dots, next_time = [], 0.0
for r in traj:  # one dot every 0.5 s
    if float(r['time']) >= next_time:
        dots.append(to_world(float(r['x']), float(r['y'])))
        next_time += 0.5

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([d[0] for d in dots], [d[1] for d in dots], '.', color='tab:red', ms=3,
        label='estimated trajectory (every 0.5 s)')
ax.plot(start['x'], start['y'], 'k^', ms=10, label='start (measured)')
for i, (tx, ty) in tags.items():
    ax.plot(tx, ty, 'ks', ms=8, mfc='none')
    ax.annotate(str(i), (tx, ty), textcoords='offset points', xytext=(6, 6), fontsize=9)
for i, (lx, ly, cov) in read_map('landmarks_final.csv').items():
    ax.plot(lx, ly, 'b+', ms=10, mew=2)
    # 2-sigma ellipse: the landmark is inside it with ~86% probability
    vals, vecs = np.linalg.eigh(cov)
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse((lx, ly), 4 * math.sqrt(vals[1]), 4 * math.sqrt(vals[0]),
                         angle=angle, fill=False, color='tab:blue'))
ax.plot([], [], 'ks', mfc='none', label='landmark, measured (ground truth)')
ax.plot([], [], 'b+', mew=2, label='landmark, estimated by SLAM (2σ ellipse)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'EKF SLAM, {shape} path — avg landmark error {results[-1][2] * 100:.1f} cm')
ax.axis('equal')
ax.grid(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(run_dir, 'slam_map.png'), dpi=150)
print(f"Saved {os.path.join(run_dir, 'slam_map.png')}")
