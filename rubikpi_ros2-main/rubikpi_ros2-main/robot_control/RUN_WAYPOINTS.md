# Run the hardcoded route

On the robot, build from your ROS workspace directory:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_control --symlink-install
source install/setup.bash
ros2 run robot_control motor_control --ros-args -p run_waypoints:=true
```

Use your installed ROS distribution if it is not Humble. The existing motor
driver expects `/dev/ttyUSB0`. Do not run keyboard control at the same time.

Place the robot at (0,0), facing positive x, before running. It starts immediately.
Ctrl+C stops the program and the motor driver
sends a final stop command.

The route is explicitly written in `run_route()` in the template's existing
`robot_control/motor_control.py`. It matches the current seven homework
waypoints, moving along x, then y, then turning to the requested angle.
It does not read `waypoints.txt`; edit the route if the assignment's points change.

Running without `-p run_waypoints:=true` keeps the original manual-control
behavior. There is no separate waypoint node or autonomous launch file.

Before the full run, calibrate the constants at the top of `motor_control.py`:

- `DRIVE_COMMAND`: straight-driving motor command (trial default 0.20).
- `DRIVE_SPEED`: actual measured m/s at that command (unmeasured default 0.20).
  Measure distance/time in a straight-motion trial and replace this value.
- `TURN_COMMAND`: fixed command for turning in place (trial default 0.08).
- `TURN_RATE`: actual measured rad/s at TURN_COMMAND (unmeasured default 0.80).
  Measure a turn angle divided by its duration and replace this value.

The kinematic model uses distance = speed * time for drives and
turn angle = turn rate * time for turns. `move()` repeats fixed left/right
commands until that time has elapsed, then stops. Commands repeat every 0.05
seconds because the template driver stops after 0.15 seconds without a command.

The route does not use IMU feedback, position checks, or heading corrections.
`self.heading` records the direction we assume the robot reached. The template
still publishes IMU data for other uses. Slip, motor imbalance, and acceleration
cause errors, so calibrate both speeds and verify the physical run.

`turn(angle)` subtracts the assumed current heading from the target heading.
Positive differences turn left; negative differences turn right. It does not
choose the shortest rotation, so some turns can exceed half a revolution.
