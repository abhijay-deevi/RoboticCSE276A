"""Closed-loop waypoint controller.

Every 0.05 s:
  1. dead reckoning: move our pose estimate using the speed we last commanded
  2. compare the pose to the current waypoint and pick a speed (v) and turn rate (w)
  3. convert (v, w) to left/right wheel commands and publish them on motor_commands
  4. log the pose to trajectory.csv

Whenever the localization node sees a tag, it publishes /robot_pose and we
replace our estimate with it. Between tags, dead reckoning keeps us going.

Run motor_control WITHOUT run_waypoints:=true so it listens on motor_commands.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D

# Same calibration as HW1 (see motor_control.py)
from robot_control.motor_control import (DRIVE_COMMAND, DRIVE_SPEED, TURN_COMMAND,
                                         LEFT_TURN_RATE, RIGHT_TURN_RATE)

# Waypoints after the start (0, 0, 0): (x, y, theta)
WAYPOINTS = [(1.0, 0.0, 0.0), (1.0, 2.0, math.pi), (0.0, 0.0, 0.0)]

POSITION_TOLERANCE = 0.05  # meters: close enough to a waypoint
ANGLE_TOLERANCE = 0.1      # radians (~6 deg): close enough to a heading
DRIVE_ANGLE_LIMIT = 0.3    # radians: if pointing further off than this, turn in place first
HEADING_GAIN = 1.0         # P gain: how hard to steer back toward the waypoint while driving

TURN_RATE = (LEFT_TURN_RATE + RIGHT_TURN_RATE) / 2  # rad/s when turning in place
DT = 0.05                                            # control loop period (s)


def wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class WaypointControl(Node):
    def __init__(self):
        super().__init__('waypoint_control')

        # Pose estimate. The robot starts at (0, 0) facing +x.
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.v, self.w = 0.0, 0.0  # last commanded speed (m/s) and turn rate (rad/s)
        self.waypoint_index = 0

        self.create_subscription(Pose2D, '/robot_pose', self.on_pose, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        self.start_time = time.time()
        self.log = open('trajectory.csv', 'w')
        self.log.write('time,x,y,theta\n')

        self.timer = self.create_timer(DT, self.loop)
        self.get_logger().info('Waypoint control started')

    def on_pose(self, msg):
        # A tag was seen: trust the camera and replace our estimate
        self.x, self.y, self.theta = msg.x, msg.y, msg.theta

    def loop(self):
        # 1. Dead reckoning: move the estimate by what we commanded during the last DT
        self.x += self.v * math.cos(self.theta) * DT
        self.y += self.v * math.sin(self.theta) * DT
        self.theta = wrap(self.theta + self.w * DT)

        # 4. Log the pose for the trajectory plot
        self.log.write(f'{time.time() - self.start_time:.2f},{self.x:.3f},{self.y:.3f},{self.theta:.3f}\n')

        if self.waypoint_index >= len(WAYPOINTS):
            self.drive(0.0, 0.0)  # done: stay stopped
            return

        # 2. Controller: compare pose to the waypoint
        goal_x, goal_y, goal_theta = WAYPOINTS[self.waypoint_index]
        distance = math.hypot(goal_x - self.x, goal_y - self.y)

        if distance > POSITION_TOLERANCE:
            # Not there yet: point at the waypoint, then drive to it
            angle_error = wrap(math.atan2(goal_y - self.y, goal_x - self.x) - self.theta)
            if abs(angle_error) > DRIVE_ANGLE_LIMIT:
                self.drive(0.0, math.copysign(TURN_RATE, angle_error))  # turn in place
            else:
                self.drive(DRIVE_SPEED, HEADING_GAIN * angle_error)    # drive and steer
        else:
            # At the position: turn to the waypoint's heading
            angle_error = wrap(goal_theta - self.theta)
            if abs(angle_error) > ANGLE_TOLERANCE:
                self.drive(0.0, math.copysign(TURN_RATE, angle_error))
            else:
                self.get_logger().info(
                    f'Reached waypoint {WAYPOINTS[self.waypoint_index]}: estimated pose '
                    f'x={self.x:.3f} y={self.y:.3f} theta={self.theta:.3f}')
                self.waypoint_index += 1
                self.drive(0.0, 0.0)

    def drive(self, v, w):
        # 3. Convert speed (m/s) and turn rate (rad/s) to wheel commands,
        # using the HW1 calibration: DRIVE_COMMAND gives DRIVE_SPEED,
        # and TURN_COMMAND (opposite on each wheel) gives TURN_RATE.
        self.v, self.w = v, w
        forward = v * DRIVE_COMMAND / DRIVE_SPEED
        turn = w * TURN_COMMAND / TURN_RATE
        msg = Float32MultiArray()
        msg.data = [float(forward - turn), float(forward + turn)]  # [left, right]
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.log.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
