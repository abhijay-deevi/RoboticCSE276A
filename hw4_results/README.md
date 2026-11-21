# HW4 path planning runs

Each folder is one run of `ros2 launch robot_control hw4_launch.py path:=safe|short`, named
`<path>_<date>_<time>` (UTC). Map: 9 ft x 9 ft, 14 in box in the middle, 12 tags
(`robot_control/config/hw4_map.yaml`). Start (7.5 ft, 1.5 ft) facing the tag 4/5 wall, goal (1.5 ft, 7.5 ft).

Files in each run folder:
- `plan.csv`: the planned waypoints (start first)
- `trajectory.csv`: the robot's estimated pose every 0.05 s, the waypoint it was heading to, and the commanded v, w
- `measurements.csv`: every tag reading and whether the EKF used it (`updated`) or threw it out (`rejected`)
- `run.log` (final run only): the robot's console output, with a status line every 2 s
- `stopped.txt` (only if the robot gave up): why it stopped

Plots (`path_planner.py hw4_results <run folders>`): planned path in blue, driven path (robot estimate) in red.
- `safe_path.png`: Voronoi diagram + A* (3.40 m planned, at least 46 cm from every obstacle), with the final safe run
- `short_path.png`: visibility graph + A* (2.79 m planned, as close as 25 cm to the box), with the best short run

## Results used in the report

| Run | Path | Result |
|---|---|---|
| `safe_20261121_092451` | safe | **Final run.** Reached the goal (all 8 waypoints) in 49 s, ends 1 cm from the goal, 3 cm from the plan on average (14 cm max), never closer than 41 cm to the box or a wall. 346 tag readings used from tags 0, 2, 3, 4, 5, 6, 8, 9, 10. |
| `safe_20261121_055732` | safe | First run. Reached the goal in 74 s, but the room was dark (only tags 5, 6, 10 seen) and one bad tag 10 reading caused a detour near the top middle. |
| `short_20261121_070508` | short | Reached the box corner on plan, then the robot's computer restarted 22 s in, about 35 cm from the goal (see below). |

## Problems found along the way (for the report's discussion)

| Runs | What happened | Fix |
|---|---|---|
| `safe_..._055732` | Room too dark: most tags never detected | More light on the tags |
| `safe_..._061955` | Turns cut short: commands were sent once per turn and motor_control's 0.15 s safety stop halted the wheels, so the robot drove nearly straight | Re-send the command every 0.05 s |
| `safe_..._062242` | One long continuous turn spun much faster than the burst-measured turn rate and overshot ~45 deg, toward the box | Turn in short bursts |
| turn test | Bursts that the HW3 turn rate counted as 90 deg actually turned 180 deg | Turn rate doubled (90 deg in ~0.63 s) |
| `safe_..._064340` | Late camera frames after a turn "undid" the turn in the EKF, causing extra turning | Ignore images taken before the last turn ended |
| `safe_..._065226` to `safe_..._092147`, `short_*` | The robot's computer restarted 2-22 s into runs, only when the camera processing and motors ran together | Lighter tag detection (half-size image, 2 cores) and 75% drive speed; the final run then finished with no restart |

Folders with empty files (`safe_..._061419`, `safe_..._062832`) are runs cut off before anything was saved to disk.
