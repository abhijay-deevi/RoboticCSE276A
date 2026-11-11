# HW3 SLAM runs

Each folder is one run of `ros2 launch robot_control hw3_launch.py`, named `<shape>_<date>_<time>` (UTC).

Files in each run folder:
- `slam_map.png`: estimated trajectory + landmark map with 2σ covariance ellipses, compared with the measured tag positions
- `errors.csv`: landmark error (meters) per tag, after each lap and at the end
- `trajectory.csv`: estimated robot pose every 0.05 s (and commanded v, w in later runs)
- `measurements.csv`: every tag reading (robot frame) and whether the EKF used, added or rejected it
- `landmarks_lap*.csv`, `landmarks_final.csv`: estimated tag positions and covariances

Plots and errors are made with `rubikpi_ros2-main/rubikpi_ros2-main/robot_control/scripts/plot_slam.py <run folder>`,
using the measured layout in `robot_control/config/hw3_ground_truth.yaml`.

## Best runs so far (9 ft x 9 ft area, tags 2 ft from the corners)

| Run | Shape | Tags mapped | Avg landmark error |
|---|---|---|---|
| `square_20261110_051131` | square, 4 ft, 1 lap | 8 / 8 | 13.7 cm |
| `octagon_20261110_053434` | octagon, 2 ft, 1 lap | 8 / 8 | 15.4 cm |
| `octagon_20261110_053713` | octagon, 2 ft, 1 lap | 7 / 8 | 15.7 cm |
| `octagon_20261110_053133` | octagon, 2 ft, 1 lap | 7 / 8 | 17.1 cm |

## Earlier runs (tuning; useful for the report discussion)

| Run | What it shows |
|---|---|
| `none_20261110_022602` | Noise test: robot standing still, ~680 readings per tag, jitter 0.1-0.2 cm |
| `square_20261110_023347` | Tag size set wrong (5 in instead of 4 in): every tag ~1.5x too far, 94 cm error |
| `square_20261110_023809` | Tag size fixed (40 cm); heading drift after turns |
| `square_20261110_024100` | Turn model + pause after turns (26 cm); robot drove into tag 5 |
| `square_20261110_045411` | Fast driving: only 84 camera readings per lap, 48 cm |
| `square_20261110_045837` | Slow driving in short bursts with stops: 30.5 cm |
| `square_20261110_050157` | Angle check after every move (within 5 deg): 18.4 cm |
| `square_20261110_050723` | Same, some tags moved by hand (ground truth not updated for them) |
| `octagon_20261110_052702` | First octagon, started at the square's start spot (not centered): 27.5 cm |

Note: these earlier runs used older settings and, before the 9 ft layout, an 8 ft area, so their
error numbers against the current ground-truth file are only rough.

Known issue: tags 1 and 7 come out ~20-30 cm off in every run. They are first seen near the edge of
the camera image, where the calibration is less accurate (~6-8 deg bearing error measured at the start).
