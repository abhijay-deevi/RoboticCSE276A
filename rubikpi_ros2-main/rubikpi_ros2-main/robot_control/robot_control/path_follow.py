"""HW4: plan a path from start to goal and drive it.

  - plans with path_planner.py: path:=safe (Voronoi + A*) or path:=short (visibility graph + A*)
  - knows where every tag is (hw4_map.yaml) and starts at the measured start pose
  - localization: the HW3 EKF, but with all tags already in the map and held fixed,
    so camera readings only correct the robot's pose
  - turns in short bursts (rate from the turn test) with a pause after each so the camera gets
    fresh frames, then drives straight continuously, steering as it goes
  - stops and says why if it sees no tags or makes no progress for 20 s
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
from robot_control.slam_drive import (DRIVE_COMMAND, DRIVE_SPEED, TURN_STRENGTH,
                                      POSITION_TOLERANCE, DRIVE_BURST, TURN_BURST,
                                      REST, HEADING_GAIN, START_DELAY, DT,
                                      CAMERA_OFFSET, SPEED_NOISE, TURN_NOISE, MEAS_NOISE,
                                      MEAS_NOISE_PER_M, OUTLIER_GATE)

TURN_RATE = math.pi / 1.25         # turn test on this floor: bursts that HW3's number counted as
                                   # 90 deg turned 180 deg, so 90 deg takes ~0.63 s (rad/s)
SLOW_SPEED = 0.75 * DRIVE_SPEED    # wheel command 0.15 instead of 0.20: less motor load
START_POSITION_STD = 0.03          # robot placed by hand: within about 3 cm
START_ANGLE_STD = math.radians(3)  # and about 3 degrees
TURN_IN_PLACE = math.radians(10)   # only turn in place when further off than this
MIN_TURN = 0.1                     # shortest turn burst that actually moves the wheels (s)
GIVE_UP = 20.0                     # stop if no tags seen / no progress for this long (s)


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
        self.burst_until = 0.0
        self.rest_until = 0.0
        self.pause_until = 0.0
        self.camera_up_time = None
        self.started = False
        self.finished = False
        # for the 2 s status line and the 20 s give-up checks
        self.frames = 0
        self.frames_recent = 0
        self.seen = {}
        self.last_seen_time = time.time()
        self.last_status = 0.0
        self.turn_end_ns = 0  # when the last turn in place finished (camera images before it are stale)
        self.best_distance = math.inf
        self.progress_time = time.time()

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
        self.frames += 1
        for det in msg.detections:
            if det.id in self.ekf.ids:
                self.last_seen_time = time.time()
                self.seen[det.id] = self.seen.get(det.id, 0) + 1
        if self.v == 0.0 and self.w != 0.0:
            return  # turning in place: blurry, late images
        if Time.from_msg(msg.header.stamp).nanoseconds < self.turn_end_ns:
            return  # image was taken before the last turn finished: it would undo the turn
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

    # ---------------- Control loop (same burst driving as HW3) ----------------

    def loop(self):
        if not self.started:
            self.drive(0.0, 0.0)
            if self.camera_up_time is None or time.time() - self.camera_up_time < START_DELAY:
                return
            self.started = True
            self.start_time = time.time()
            self.last_seen_time = self.progress_time = self.start_time
            self.get_logger().info('GO!')

        self.ekf.predict(self.v, self.w, DT)
        x, y, theta = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        self.traj_file.write(f'{time.time() - self.start_time:.2f},{self.waypoint_index},'
                             f'{x:.4f},{y:.4f},{theta:.4f},{self.v:.4f},{self.w:.4f}\n')

        if self.finished:
            self.drive(0.0, 0.0)
            return
        now = time.time()
        if now - self.last_status >= 2.0:
            self.status(now, x, y, theta)
        if now - self.last_seen_time > GIVE_UP:
            return self.give_up(f'no tags seen for {GIVE_UP:.0f} s (camera frames in the last 2 s: '
                                f'{self.frames_recent}). Tags are too far, too dark, or out of view.')
        if now - self.progress_time > GIVE_UP:
            return self.give_up(f'no progress toward waypoint {self.waypoint_index + 1} for '
                                f'{GIVE_UP:.0f} s (stuck turning or driving in place)')
        if now < self.burst_until:
            self.drive(self.v, self.w)
            return
        if self.v == 0.0 and self.w != 0.0:
            # a turn burst just ended: stop so the camera gets fresh, sharp frames
            self.turn_end_ns = self.get_clock().now().nanoseconds
            self.rest_until = now + REST
            self.drive(0.0, 0.0)
            return
        if now < self.rest_until or now < self.pause_until:
            self.drive(0.0, 0.0)
            return
        # (a drive burst that just ended rolls straight into the next one: every restart from a
        # stop draws a current surge, and the robot's computer was browning out on those)

        goal_x, goal_y = self.route[self.waypoint_index]
        distance = math.hypot(goal_x - x, goal_y - y)
        if distance < self.best_distance - 0.02:  # got at least 2 cm closer
            self.best_distance, self.progress_time = distance, now
        if distance > POSITION_TOLERANCE:
            angle_error = wrap(math.atan2(goal_y - y, goal_x - x) - theta)
            # small errors are fixed by steering while driving; a tiny turn in place doesn't
            # move the wheels at all, so it would just repeat forever
            if abs(angle_error) > TURN_IN_PLACE:
                if self.v != 0.0:
                    # still rolling: stop first (the turn rate was measured from a standstill)
                    self.rest_until = now + REST
                    self.drive(0.0, 0.0)
                    return
                seconds = max(MIN_TURN, min(TURN_BURST, abs(angle_error) / TURN_RATE))
                self.burst(0.0, math.copysign(TURN_RATE, angle_error), seconds)
            else:
                seconds = max(0.1, min(DRIVE_BURST, distance / SLOW_SPEED))
                self.burst(SLOW_SPEED, HEADING_GAIN * angle_error, seconds)
            return

        # Reached a waypoint
        self.get_logger().info(f'Waypoint {self.waypoint_index + 1}/{len(self.route)} reached '
                               f'at ({x:.2f}, {y:.2f})')
        self.waypoint_index += 1
        self.best_distance, self.progress_time = math.inf, now
        if self.waypoint_index < len(self.route):
            self.drive(self.v, self.w)  # keep rolling; it stops on its own if the next leg needs a turn
        else:
            self.drive(0.0, 0.0)
            self.finished = True
            self.traj_file.flush()
            self.meas_file.flush()
            self.get_logger().info(f'Goal reached. Results in {self.run_dir}')

    def status(self, now, x, y, theta):
        """One line every 2 s so a log always shows what the robot was doing."""
        doing = 'turning' if self.v == 0.0 and self.w != 0.0 else \
            'driving' if self.v != 0.0 else 'stopped'
        self.frames_recent = self.frames
        self.get_logger().info(
            f'[{now - self.start_time:5.1f} s] waypoint {self.waypoint_index + 1}/{len(self.route)}, '
            f'at ({x:.2f}, {y:.2f}) facing {math.degrees(theta):.0f} deg, {doing}; '
            f'camera frames {self.frames}, tags seen {self.seen or "none"}')
        self.frames, self.seen, self.last_status = 0, {}, now
        for f in (self.traj_file, self.meas_file):
            f.flush()
            os.fsync(f.fileno())  # on disk even if the power is cut

    def give_up(self, reason):
        self.finished = True
        self.drive(0.0, 0.0)
        self.get_logger().error(f'STOPPED: {reason}')
        with open(os.path.join(self.run_dir, 'stopped.txt'), 'w') as f:
            f.write(reason + '\n')

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
