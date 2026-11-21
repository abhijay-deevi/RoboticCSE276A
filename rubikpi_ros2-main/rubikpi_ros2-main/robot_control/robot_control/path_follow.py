"""HW4: plan a path from start to goal and drive it.

  - plans with path_planner.py: path:=safe (Voronoi + A*) or path:=short (visibility graph + A*)
  - knows where every tag is (hw4_map.yaml) and starts at the measured start pose
  - localization: the HW3 EKF, but with all tags already in the map and held fixed,
    so camera readings only correct the robot's pose. With no tags in view it keeps going
    on the wheel model (floor-tested in HW3), so short camera range doesn't stop it.
  - drives smoothly toward each waypoint, steering as it goes; one turn in place only when
    the next waypoint is off at a sharp angle
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
from robot_control.path_planner import load_map, plan, path_length
# same floor-tested driving settings and noise as HW3
from robot_control.slam_drive import (DRIVE_COMMAND, DRIVE_SPEED, TURN_STRENGTH, TURN_RATE,
                                      POSITION_TOLERANCE, HEADING_GAIN, START_DELAY, DT,
                                      CAMERA_OFFSET, SPEED_NOISE, TURN_NOISE, MEAS_NOISE,
                                      MEAS_NOISE_PER_M, OUTLIER_GATE)

START_POSITION_STD = 0.03          # robot placed by hand: within about 3 cm
START_ANGLE_STD = math.radians(3)  # and about 3 degrees
TURN_IN_PLACE = math.radians(30)   # next waypoint further off than this: stop and turn to it first
LINED_UP = math.radians(5)         # after a turn in place, good enough to start driving
MAX_STEER = 0.5                    # rad/s of steering allowed while driving
SETTLE = 0.3                       # seconds still after a turn in place


class PathFollow(Node):
    def __init__(self):
        super().__init__('path_follow')
        self.kind = self.declare_parameter('path', 'safe').value

        m = load_map(os.path.join(get_package_share_directory('robot_control'),
                                  'config', 'hw4_map.yaml'))
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
        self.turn_until = 0.0      # end of the current turn in place
        self.settle_until = 0.0
        self.camera_up_time = None
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
            self.meas_file.write(f'{time.time() - self.start_time:.2f},{det.id},'
                                 f'{forward:.4f},{left:.4f},{result}\n')

    # ---------------- Control loop ----------------

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
        # keep sending commands every loop: motor_control stops the wheels after 0.15 s of silence
        if now < self.turn_until:
            self.drive(self.v, self.w)  # turning in place: keep going
            return
        if self.v == 0.0 and self.w != 0.0:
            self.settle_until = now + SETTLE  # turn just ended
            self.drive(0.0, 0.0)
            return
        if now < self.settle_until:
            self.drive(0.0, 0.0)
            return

        goal_x, goal_y = self.route[self.waypoint_index]
        distance = math.hypot(goal_x - x, goal_y - y)
        angle_error = wrap(math.atan2(goal_y - y, goal_x - x) - theta)

        # Reached the waypoint (or just passed it): on to the next one without stopping
        if distance < POSITION_TOLERANCE or (distance < 0.15 and abs(angle_error) > math.pi / 2):
            self.get_logger().info(f'Waypoint {self.waypoint_index + 1}/{len(self.route)} '
                                   f'reached at ({x:.2f}, {y:.2f})')
            self.waypoint_index += 1
            if self.waypoint_index == len(self.route):
                self.finished = True
                self.drive(0.0, 0.0)
                self.get_logger().info(f'Goal reached. Results in {self.run_dir}')
            else:
                self.drive(self.v, self.w)
            return

        stopped = self.v == 0.0
        if abs(angle_error) > TURN_IN_PLACE or (stopped and abs(angle_error) > LINED_UP):
            # one turn in place, timed with the floor-tested turn rate
            self.turn_until = now + abs(angle_error) / TURN_RATE
            self.drive(0.0, math.copysign(TURN_RATE, angle_error))
            return
        steer = max(-MAX_STEER, min(MAX_STEER, HEADING_GAIN * angle_error))
        self.drive(DRIVE_SPEED, steer)

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
