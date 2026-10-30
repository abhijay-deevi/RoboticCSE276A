"""Closed-loop waypoint controller with AprilTag localization.

Localization (whenever the camera sees tags):
  - read which tag IDs are visible (/detections)
  - look up where each tag is relative to the camera (TF camera_frame -> tag36h11:<id>)
  - look up where that tag is in the world (tag_map.yaml)
  - work backwards to where the robot must be, average over all visible tags,
    and replace our pose estimate with it

  - tags are ignored while spinning in place (blurry images, delayed detections)

Control loop (every 0.05 s):
  0. wait until the camera is publishing detections, then a few more seconds
  1. dead reckoning: move our pose estimate using the speed we last commanded
     (this keeps us going when no tag is visible)
  2. log the pose to ~/trajectory.csv
  3. compare the pose to the current waypoint and pick a speed (v) and turn rate (w)
     At a waypoint: stop, pause so the camera gets clean looks, re-check, then
     record the estimated pose to ~/waypoints.csv and move on.
  4. convert (v, w) to left/right wheel commands and publish them on motor_commands

Started by hw2_launch.py, which runs motor_control WITHOUT run_waypoints:=true
so it listens on motor_commands.
"""

import math
import os
import time

import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory

# Same calibration as HW1 (see motor_control.py)
from robot_control.motor_control import (DRIVE_COMMAND, DRIVE_SPEED, TURN_COMMAND,
                                         LEFT_TURN_RATE, RIGHT_TURN_RATE)

# Waypoints after the start (0, 0, 0): (x, y, theta)
WAYPOINTS = [(1.0, 0.0, 0.0), (1.0, 2.0, math.pi), (0.0, 0.0, 0.0)]

POSITION_TOLERANCE = 0.05  # meters: close enough to a waypoint
ANGLE_TOLERANCE = 0.1      # radians (~6 deg): close enough to a heading
DRIVE_ANGLE_LIMIT = 0.3    # radians: if pointing further off than this, turn in place first
HEADING_GAIN = 1.0         # P gain: how hard to steer back toward the waypoint while driving
START_DELAY = 3.0          # seconds to wait after the camera is up before driving
WAYPOINT_PAUSE = 2.0       # seconds to stop at each waypoint (shows clearly in the video)

TURN_RATE = (LEFT_TURN_RATE + RIGHT_TURN_RATE) / 2  # rad/s when turning in place
DT = 0.05                                            # control loop period (s)
CAMERA_OFFSET = 0.09                                 # meters the camera sits in front of the robot center


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
        self.camera_up_time = None  # when the first /detections message arrived
        self.started = False
        self.pause_until = 0.0      # stopped at a waypoint until this time
        self.paused_here = False    # already paused at the current waypoint?

        # Load the tag map: {tag_id: {'x': .., 'y': .., 'yaw': ..}}
        map_file = os.path.join(get_package_share_directory('robot_control'),
                                'config', 'tag_map.yaml')
        with open(map_file) as f:
            self.tag_map = yaml.safe_load(f)['tags']

        # TF listener: receives the camera -> tag transforms from the apriltag node
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        self.start_time = time.time()
        self.log = open(os.path.expanduser('~/trajectory.csv'), 'w')
        self.log.write('time,x,y,theta\n')
        self.waypoint_log = open(os.path.expanduser('~/waypoints.csv'), 'w')
        self.waypoint_log.write('goal_x,goal_y,goal_theta,est_x,est_y,est_theta\n')

        self.timer = self.create_timer(DT, self.loop)
        self.get_logger().info(f'Waypoint control ready, {len(self.tag_map)} tags in map. '
                               'Waiting for camera...')

    # ---------------- Localization ----------------

    def on_detections(self, msg):
        if self.camera_up_time is None:
            self.camera_up_time = time.time()
        if self.v == 0.0 and self.w != 0.0:
            return  # spinning in place: images are blurry, trust dead reckoning

        estimates = []  # one (x, y, theta) guess per visible tag
        for det in msg.detections:
            if det.id not in self.tag_map:
                continue  # tag not on our map, ignore it
            estimate = self.pose_from_tag(det.id)
            if estimate is not None:
                estimates.append(estimate)

        if not estimates:
            return  # no usable tag: keep the dead-reckoning estimate

        # Average the guesses and trust the camera. Angles are averaged with
        # sin/cos so that +179 deg and -179 deg average to 180 deg, not 0 deg.
        self.x = sum(e[0] for e in estimates) / len(estimates)
        self.y = sum(e[1] for e in estimates) / len(estimates)
        self.theta = math.atan2(sum(math.sin(e[2]) for e in estimates),
                                sum(math.cos(e[2]) for e in estimates))

    def pose_from_tag(self, tag_id):
        # Where is the tag relative to the camera?
        try:
            t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{tag_id}', Time())
        except Exception:
            return None
        p = t.transform.translation
        q = t.transform.rotation

        # Camera frame: z = forward, x = right, y = down.
        # Convert to the robot's 2D frame: forward and left, from the robot center.
        forward = p.z + CAMERA_OFFSET
        left = -p.x

        # Which way does the tag face, seen from the robot?
        # The tag's z axis points out of its printed side. Take that axis
        # (third column of the rotation matrix) and keep its forward/left parts.
        face_forward = 1 - 2 * (q.x * q.x + q.y * q.y)   # camera z part
        face_left = -2 * (q.x * q.z + q.w * q.y)         # minus camera x part
        tag_angle_seen = math.atan2(face_left, face_forward)

        # Where is the tag in the world? (from tag_map.yaml)
        tag = self.tag_map[tag_id]

        # Work backwards:
        # tag world yaw = robot yaw + tag angle seen  ->  solve for robot yaw
        theta = wrap(tag['yaw'] - tag_angle_seen)

        # tag world position = robot position + (forward, left) rotated by robot yaw
        # -> solve for robot position
        x = tag['x'] - (forward * math.cos(theta) - left * math.sin(theta))
        y = tag['y'] - (forward * math.sin(theta) + left * math.cos(theta))
        return x, y, theta

    # ---------------- Control ----------------

    def loop(self):
        # 0. Wait for the camera, then give it a few seconds to see a tag
        if not self.started:
            self.drive(0.0, 0.0)
            if self.camera_up_time is None or time.time() - self.camera_up_time < START_DELAY:
                return
            self.started = True
            self.start_time = time.time()
            self.get_logger().info(f'GO! Starting pose x={self.x:.3f} y={self.y:.3f} '
                                   f'theta={self.theta:.3f}')

        # 1. Dead reckoning: move the estimate by what we commanded during the last DT
        self.x += self.v * math.cos(self.theta) * DT
        self.y += self.v * math.sin(self.theta) * DT
        self.theta = wrap(self.theta + self.w * DT)

        # 2. Log the pose for the trajectory plot
        self.log.write(f'{time.time() - self.start_time:.2f},{self.x:.3f},{self.y:.3f},{self.theta:.3f}\n')

        if self.waypoint_index >= len(WAYPOINTS):
            self.drive(0.0, 0.0)  # done: stay stopped
            return

        if time.time() < self.pause_until:
            self.drive(0.0, 0.0)  # stopped at a waypoint, camera is correcting the pose
            return

        # 3. Controller: compare pose to the waypoint
        goal_x, goal_y, goal_theta = WAYPOINTS[self.waypoint_index]
        distance = math.hypot(goal_x - self.x, goal_y - self.y)

        if distance > POSITION_TOLERANCE:
            # Not there yet: point at the waypoint, then drive to it
            self.paused_here = False
            angle_error = wrap(math.atan2(goal_y - self.y, goal_x - self.x) - self.theta)
            if abs(angle_error) > DRIVE_ANGLE_LIMIT:
                self.drive(0.0, math.copysign(TURN_RATE, angle_error))  # turn in place
            else:
                self.drive(DRIVE_SPEED, HEADING_GAIN * angle_error)    # drive and steer
        else:
            # At the position: turn to the waypoint's heading
            angle_error = wrap(goal_theta - self.theta)
            if abs(angle_error) > ANGLE_TOLERANCE:
                self.paused_here = False
                self.drive(0.0, math.copysign(TURN_RATE, angle_error))
            elif not self.paused_here:
                # Looks done: stop and let the camera take clean looks, then check again
                self.paused_here = True
                self.pause_until = time.time() + WAYPOINT_PAUSE
                self.drive(0.0, 0.0)
            else:
                # Still within tolerance after the pause: this waypoint is done
                self.get_logger().info(
                    f'Reached waypoint {WAYPOINTS[self.waypoint_index]}: estimated pose '
                    f'x={self.x:.3f} y={self.y:.3f} theta={self.theta:.3f}')
                self.waypoint_log.write(f'{goal_x},{goal_y},{goal_theta:.3f},'
                                        f'{self.x:.3f},{self.y:.3f},{self.theta:.3f}\n')
                self.waypoint_log.flush()
                self.waypoint_index += 1
                self.paused_here = False
                self.drive(0.0, 0.0)
                if self.waypoint_index == len(WAYPOINTS):
                    self.get_logger().info('Route finished. Results in ~/waypoints.csv '
                                           'and ~/trajectory.csv')

    def drive(self, v, w):
        # 4. Convert speed (m/s) and turn rate (rad/s) to wheel commands,
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
    node.waypoint_log.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
