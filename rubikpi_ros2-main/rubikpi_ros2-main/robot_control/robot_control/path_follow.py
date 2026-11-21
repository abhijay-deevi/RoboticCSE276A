"""HW4: plan a path from start to goal and drive it.

  - plans with path_planner.py: path:=safe (Voronoi + A*) or path:=short (visibility graph + A*)
  - knows where every tag is (hw4_map.yaml) and starts at the measured start pose
  - localization: the HW3 EKF, but with all tags already in the map and held fixed,
    so camera readings only correct the robot's pose
  - drives in the same short bursts as HW3: turn until facing the next waypoint, drive, stop, repeat
  - far tags are often too small/dark to read, so when there has been no good reading for a few
    seconds, it turns to face the closest tag it should be able to see (it knows where they all
    are), waits for a reading, then carries on
  - stops if the camera goes quiet
  - logs go to ~/hw4_runs/<path>_<date_time>/
"""

import math
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory

from robot_control.ekf_slam import EkfSlam, wrap
from robot_control.path_planner import load_map, plan, path_length, segment_clear
# same floor-tested driving settings and noise as HW3
from robot_control.slam_drive import (DRIVE_COMMAND, DRIVE_SPEED, TURN_STRENGTH, TURN_RATE,
                                      POSITION_TOLERANCE, TURN_DONE, DRIVE_BURST, TURN_BURST,
                                      REST, HEADING_GAIN, START_DELAY, CORNER_PAUSE, DT,
                                      CAMERA_OFFSET, SPEED_NOISE, TURN_NOISE, MEAS_NOISE,
                                      MEAS_NOISE_PER_M, OUTLIER_GATE)

START_POSITION_STD = 0.03          # robot placed by hand: within about 3 cm
START_ANGLE_STD = math.radians(3)  # and about 3 degrees
FIX_AGE = 3.0         # no good tag reading for this long -> go look at the closest tag
LOOK_TIMEOUT = 6.0    # give up looking after this long and keep going
CAMERA_TIMEOUT = 2.0  # no camera frames for this long -> stop and wait


class PathFollow(Node):
    def __init__(self):
        super().__init__('path_follow')
        self.kind = self.declare_parameter('path', 'safe').value

        m = load_map(os.path.join(get_package_share_directory('robot_control'),
                                  'config', 'hw4_map.yaml'))
        self.map = m
        self.route = plan(m, self.kind)[1:]  # first point is the start itself

        # EKF with the known map: robot at the measured start, every tag fixed in place
        self.ekf = EkfSlam(SPEED_NOISE, TURN_NOISE, MEAS_NOISE, MEAS_NOISE_PER_M, OUTLIER_GATE)
        s = m['start']
        self.ekf.ids = list(m['tags'])
        self.ekf.x = np.array([s['x'], s['y'], s['theta']] +
                              [v for t in m['tags'].values() for v in (t['x'], t['y'])])
        self.ekf.P = np.diag([START_POSITION_STD ** 2] * 2 + [START_ANGLE_STD ** 2] +
                             [1e-8] * (2 * len(m['tags'])))

        self.waypoint_index = 0
        self.v, self.w = 0.0, 0.0
        self.burst_until = 0.0
        self.rest_until = 0.0
        self.pause_until = 0.0
        self.camera_up_time = None
        self.last_frame_time = None
        self.last_fix_time = 0.0    # last accepted tag reading
        self.look_tag = None        # tag we turned to look at, if any
        self.look_started = 0.0
        self.started = False
        self.finished = False

        self.run_dir = os.path.expanduser(
            f"~/hw4_runs/{self.kind}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.run_dir)
        with open(os.path.join(self.run_dir, 'plan.csv'), 'w') as f:
            f.write('x,y\n')
            for x, y in [(s['x'], s['y'])] + self.route:
                f.write(f'{x:.4f},{y:.4f}\n')
        # line-buffered so nothing is lost if the run is cut short
        self.traj_file = open(os.path.join(self.run_dir, 'trajectory.csv'), 'w', buffering=1)
        self.traj_file.write('time,waypoint,x,y,theta,v,w\n')
        self.meas_file = open(os.path.join(self.run_dir, 'measurements.csv'), 'w', buffering=1)
        self.meas_file.write('time,tag_id,forward,left,result\n')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        self.start_time = time.time()
        self.create_timer(DT, self.loop)
        self.get_logger().info(
            f'{self.kind} path: {len(self.route)} waypoints, '
            f'{path_length([(s["x"], s["y"])] + self.route):.2f} m. '
            f'Logging to {self.run_dir}. Waiting for camera...')

    # ---------------- Measurements ----------------

    def on_detections(self, msg):
        if self.camera_up_time is None:
            self.camera_up_time = time.time()
        self.last_frame_time = time.time()
        if self.v == 0.0 and self.w != 0.0:
            return  # turning in place: blurry, late images
        for det in msg.detections:
            if det.id not in self.ekf.ids:
                continue  # not a tag from our map
            try:
                t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{det.id}', Time())
            except Exception:
                continue
            forward = t.transform.translation.z + CAMERA_OFFSET
            left = -t.transform.translation.x
            result = self.ekf.update(det.id, forward, left)
            if result == 'updated':
                self.last_fix_time = time.time()
            self.meas_file.write(f'{time.time() - self.start_time:.2f},{det.id},'
                                 f'{forward:.4f},{left:.4f},{result}\n')

    def closest_visible_tag(self, x, y):
        """Closest tag the camera should be able to see from (x, y): the box isn't in the way,
        and for tags on the box, the robot is on the side that tag faces."""
        b = self.map['box']
        half = b['size'] / 2 - 0.01  # line of sight may touch the face the tag is on
        box = (b['x'] - half, b['y'] - half, b['x'] + half, b['y'] + half)
        best, best_d = None, math.inf
        for tag_id, t in self.map['tags'].items():
            on_box = abs(t['x'] - b['x']) <= b['size'] / 2 + 0.01 and \
                abs(t['y'] - b['y']) <= b['size'] / 2 + 0.01
            if on_box and (x - t['x']) * (t['x'] - b['x']) + (y - t['y']) * (t['y'] - b['y']) <= 0:
                continue  # robot is behind this face of the box
            if not segment_clear((x, y), (t['x'], t['y']), box, self.map['room'], 0.0):
                continue
            d = math.hypot(t['x'] - x, t['y'] - y)
            if d < best_d:
                best, best_d = tag_id, d
        return best

    # ---------------- Control loop (same burst driving as HW3) ----------------

    def loop(self):
        if not self.started:
            self.drive(0.0, 0.0)
            if self.camera_up_time is None or time.time() - self.camera_up_time < START_DELAY:
                return
            self.started = True
            self.start_time = time.time()
            self.get_logger().info('GO!')

        self.ekf.predict(self.v, self.w, DT)
        x, y, theta = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        self.traj_file.write(f'{time.time() - self.start_time:.2f},{self.waypoint_index},'
                             f'{x:.4f},{y:.4f},{theta:.4f},{self.v:.4f},{self.w:.4f}\n')

        if self.finished:
            self.drive(0.0, 0.0)
            return
        now = time.time()
        if now < self.burst_until:
            self.drive(self.v, self.w)
            return
        if self.v != 0.0 or self.w != 0.0:
            self.rest_until = now + REST
            self.drive(0.0, 0.0)
            return
        if now < self.rest_until or now < self.pause_until:
            self.drive(0.0, 0.0)
            return
        if now - self.last_frame_time > CAMERA_TIMEOUT:
            self.drive(0.0, 0.0)  # camera went quiet: don't drive blind
            self.get_logger().warn('No camera frames, waiting...', throttle_duration_sec=2.0)
            return

        # Been a while since a good reading: turn to face the closest tag and wait for one
        if self.look_tag is None and now - self.last_fix_time > FIX_AGE:
            self.look_tag = self.closest_visible_tag(x, y)
            self.look_started = now
            if self.look_tag is None:
                self.last_fix_time = now  # nothing to look at from here: drive on for a bit
            else:
                self.get_logger().info(f'No recent tag reading, looking at tag {self.look_tag}')
        if self.look_tag is not None:
            if self.last_fix_time > self.look_started or now - self.look_started > LOOK_TIMEOUT:
                self.look_tag = None  # got a fresh reading (or gave up): back to the path
                self.last_fix_time = max(self.last_fix_time, now)
            else:
                t = self.map['tags'][self.look_tag]
                angle_error = wrap(math.atan2(t['y'] - y, t['x'] - x) - theta)
                if abs(angle_error) > TURN_DONE:
                    seconds = max(DT, min(TURN_BURST, abs(angle_error) / TURN_RATE))
                    self.burst(0.0, math.copysign(TURN_RATE, angle_error), seconds)
                return  # facing it: stand still until a reading comes in

        goal_x, goal_y = self.route[self.waypoint_index]
        distance = math.hypot(goal_x - x, goal_y - y)
        if distance > POSITION_TOLERANCE:
            angle_error = wrap(math.atan2(goal_y - y, goal_x - x) - theta)
            if abs(angle_error) > TURN_DONE:
                seconds = max(DT, min(TURN_BURST, abs(angle_error) / TURN_RATE))
                self.burst(0.0, math.copysign(TURN_RATE, angle_error), seconds)
            else:
                seconds = max(0.1, min(DRIVE_BURST, distance / DRIVE_SPEED))
                self.burst(DRIVE_SPEED, HEADING_GAIN * angle_error, seconds)
            return

        # Reached a waypoint
        self.get_logger().info(f'Waypoint {self.waypoint_index + 1}/{len(self.route)} reached '
                               f'at ({x:.2f}, {y:.2f})')
        self.waypoint_index += 1
        self.pause_until = time.time() + CORNER_PAUSE
        self.drive(0.0, 0.0)
        if self.waypoint_index == len(self.route):
            self.finished = True
            self.traj_file.flush()
            self.meas_file.flush()
            self.get_logger().info(f'Goal reached. Results in {self.run_dir}')

    def burst(self, v, w, seconds):
        self.burst_until = time.time() + seconds
        self.drive(v, w)

    def drive(self, v, w):
        self.v, self.w = v, w
        forward = v * DRIVE_COMMAND / DRIVE_SPEED
        turn = w * TURN_STRENGTH / TURN_RATE
        msg = Float32MultiArray()
        msg.data = [float(forward - turn), float(forward + turn)]  # [left, right]
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollow()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.traj_file.close()
    node.meas_file.close()
    try:
        node.drive(0.0, 0.0)
    except Exception:
        pass  # already shut down; motor_control stops the wheels 0.15 s after commands stop
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
