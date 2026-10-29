"""Plot the robot's estimated trajectory and compute total distance moved.

Usage (on your laptop, after copying trajectory.csv off the robot):
    python plot_trajectory.py trajectory.csv

Draws one dot every 0.5 s (report part i) and prints the total distance
driven and total rotation (report part g).
"""

import math
import sys

import matplotlib.pyplot as plt

DOT_EVERY = 0.5  # seconds between dots
WAYPOINTS = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]

# Read the log written by waypoint_control: time,x,y,theta
rows = []
with open(sys.argv[1]) as f:
    next(f)  # skip header
    for line in f:
        t, x, y, theta = (float(v) for v in line.split(','))
        rows.append((t, x, y, theta))

# Total distance: add up every small step in position and in heading
distance = 0.0
rotation = 0.0
for (t0, x0, y0, th0), (t1, x1, y1, th1) in zip(rows, rows[1:]):
    distance += math.hypot(x1 - x0, y1 - y0)
    rotation += abs(math.atan2(math.sin(th1 - th0), math.cos(th1 - th0)))
print(f'Total distance driven: {distance:.2f} m')
print(f'Total rotation: {rotation:.2f} rad ({math.degrees(rotation):.0f} deg)')
print(f'Total time: {rows[-1][0]:.1f} s')

# Keep one point every DOT_EVERY seconds
dots = []
next_time = 0.0
for t, x, y, theta in rows:
    if t >= next_time:
        dots.append((x, y))
        next_time += DOT_EVERY

plt.plot([d[0] for d in dots], [d[1] for d in dots], 'r.', label='estimated pose (every 0.5 s)')
plt.plot([w[0] for w in WAYPOINTS], [w[1] for w in WAYPOINTS], 'kx', markersize=12, label='waypoints')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Robot trajectory')
plt.savefig('trajectory.png', dpi=150)
plt.show()
