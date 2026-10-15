# Run the waypoint controller

Run these commands on the rover's Linux/ROS 2 system. From your ROS workspace
(the directory containing `src`), build and source the package:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_control --symlink-install
source install/setup.bash
```

Use the setup script for your installed ROS distribution if it is not Humble.
The existing motor driver expects the robot at `/dev/ttyUSB0` with baud 115200.

Place the robot at (0, 0) facing positive x. Its first valid IMU reading defines
zero heading. Start the controller with the absolute path to your waypoint file:

```bash
ros2 launch robot_control robot_waypoints_launch.py \
  waypoints_file:=/absolute/path/to/waypoints.txt
```

This launches the motor driver and waypoint controller together. Do not also run
the keyboard controller; it publishes to the same motor command topic. The robot
starts once valid IMU data arrives. Ctrl+C shuts down both processes and the
motor driver sends a stop command.

## Calibration values

The defaults are trial values, not measured properties of your robot:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `wheel_spacing` | 0.20 | Left-to-right wheel path separation, meters; effective value can differ during skid turns. |
| `left_speed_scale` | 1.0 | Estimated left ground speed divided by left motor command. |
| `right_speed_scale` | 1.0 | Estimated right ground speed divided by right motor command. |
| `imu_yaw_sign` | 1.0 | Set to -1.0 if the reported heading decreases during a physical counterclockwise turn. |

For straight travel, measure distance divided by time to obtain actual speed.
Divide speed by the motor command to estimate a speed scale. Check left/right
imbalance with additional trials. Verify turns, IMU direction, and the smallest
commands that reliably move the robot. Ground conditions affect calibration.

Pass your measured values with launch arguments, for example
`wheel_spacing:=YOUR_MEASURED_VALUE`, `left_speed_scale:=YOUR_MEASURED_VALUE`,
and `right_speed_scale:=YOUR_MEASURED_VALUE` on the launch command above.

The controller moves along x, then y, then aligns to each waypoint angle. It runs
at 20 Hz and defaults to 0.12 m/s maximum forward speed, 0.6 rad/s maximum turn
rate, 0.05 m tolerance per coordinate, and 0.087 rad heading tolerance. Additional
parameters and their defaults are near the top of `robot_control/waypoint_control.py`.

Position is estimated from commanded motor speeds and IMU heading. There is no
independent x/y measurement, so slip and motor calibration errors accumulate.
Completion messages report estimated poses, not verified physical positions.
Test a short route first and measure actual errors before the complete run.

No IMU at startup, stale IMU data, a missed motor-command deadline, or a waypoint
taking over 120 seconds causes a latched stop. Fix the issue, return the robot to
the starting pose, and restart the launch to try again.
