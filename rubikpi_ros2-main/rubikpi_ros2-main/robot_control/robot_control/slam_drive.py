"""HW3: drive a square or octagon while running EKF SLAM on the AprilTags.

The robot knows nothing about the tags. Its start pose is the map origin (0, 0, 0).
  - every 0.05 s: EKF predict with the commanded speed, then steer to the next corner
  - corners: timed turn in place (90 deg = 0.8 s at full strength, measured on our floor)
  - every tag detection: EKF update (new tag -> added to the map)
  - logs go to ~/hw3_runs/<shape>_<date_time>/

Parameters: shape (square | octagon | none), side (meters), laps
  shape:=none keeps the robot still and only logs measurements (for measuring noise).
"""

import csv
import math
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener

from robot_control.ekf_slam import EkfSlam, wrap
from robot_control.motor_control import DRIVE_COMMAND

# Measured on THIS floor with our forward/turn tests (HW1's numbers were for another floor)
DRIVE_SPEED = 1.22 / 5.8          # m/s at DRIVE_COMMAND (0.20): 4 ft took 5.8 s
TURN_STRENGTH = 0.50              # wheel command for turning in place (motor driver maximum)
TURN_RATE = (math.pi / 2) / 0.8   # rad/s at TURN_STRENGTH: 90 deg took 0.8 s

POSITION_TOLERANCE = 0.05  # meters: close enough to a corner
DRIVE_ANGLE_LIMIT = 0.3    # radians: if pointing further off than this, turn in place first
HEADING_GAIN = 1.0         # P gain for steering while driving (same as HW2)
START_DELAY = 3.0          # seconds to wait after the camera is up before driving
CORNER_PAUSE = 1.0         # seconds to stop at each corner so the camera gets clear looks
TURN_PAUSE = 0.5           # seconds to stop after a turn, before driving (HW1 stopped 0.3 s)

DT = 0.05                  # loop period (s)
CAMERA_OFFSET = 0.09       # meters the camera sits in front of the robot center

# EKF noise settings (see report: how noise was chosen)
SPEED_NOISE = 0.15         # wheels: speed is off by about 15%
TURN_NOISE = 0.20          # wheels: timed turns are off by up to ~20%
MEAS_NOISE = 0.02          # camera: 2 cm error up close
MEAS_NOISE_PER_M = 0.03    # camera: plus 3 cm per meter of distance
OUTLIER_GATE = 50.0        # reject only readings ~7 sigma off; 3.7 sigma (13.8) also threw out real
                           # drift corrections in our test laps


def path_corners(shape, side):
    """Corners of the path, starting at (0, 0) facing +x and turning left at each corner."""
    sides, turn = {'square': (4, math.pi / 2), 'octagon': (8, math.pi / 4)}[shape]
    corners = []
    x = y = heading = 0.0
    for _ in range(sides):
        x += side * math.cos(heading)
        y += side * math.sin(heading)
        corners.append((x, y))
        heading += turn
    return corners  # the last corner is back at the start


class SlamDrive(Node):
    def __init__(self):
        super().__init__('slam_drive')
        self.shape = self.declare_parameter('shape', 'square').value
        side = self.declare_parameter('side', 1.22).value
        self.laps = self.declare_parameter('laps', 3).value

        self.ekf = EkfSlam(SPEED_NOISE, TURN_NOISE, MEAS_NOISE, MEAS_NOISE_PER_M, OUTLIER_GATE)
        self.corners = [] if self.shape == 'none' else path_corners(self.shape, side)
        self.route = self.corners * self.laps
        self.corner_index = 0
        self.v, self.w = 0.0, 0.0
        self.turn_steps_left = 0   # loops left in the current timed turn
        self.turn_wheels = 0.0     # +TURN_STRENGTH for a left turn, -TURN_STRENGTH for right
        self.camera_up_time = None
        self.started = False
        self.pause_until = 0.0
        self.finished = False

        # Log folder for this run
        self.run_dir = os.path.expanduser(
            f"~/hw3_runs/{self.shape}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.run_dir)
        self.traj_file = open(os.path.join(self.run_dir, 'trajectory.csv'), 'w')
        self.traj_file.write('time,lap,x,y,theta,num_landmarks,v,w\n')
        self.meas_file = open(os.path.join(self.run_dir, 'measurements.csv'), 'w')
        self.meas_file.write('time,tag_id,forward,left,result\n')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        self.start_time = time.time()
        self.create_timer(DT, self.loop)
        self.get_logger().info(f'SLAM {self.shape}, side {side} m, {self.laps} laps. '
                               f'Logging to {self.run_dir}. Waiting for camera...')

    # ---------------- Measurements ----------------

    def on_detections(self, msg):
        if self.camera_up_time is None:
            self.camera_up_time = time.time()
        if self.turn_steps_left > 0:
            return  # turning fast: blurry, late images; rely on the prediction
        for det in msg.detections:
            try:
                t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{det.id}', Time())
            except Exception:
                continue
            # Camera frame: z = forward, x = right. Robot frame: forward, left (from robot center).
            forward = t.transform.translation.z + CAMERA_OFFSET
            left = -t.transform.translation.x
            result = self.ekf.update(det.id, forward, left)
            self.meas_file.write(f'{time.time() - self.start_time:.2f},{det.id},'
                                 f'{forward:.4f},{left:.4f},{result}\n')
            if result == 'new':
                self.get_logger().info(f'New landmark: tag {det.id} '
                                       f'({len(self.ekf.ids)} in map)')

    # ---------------- Control loop ----------------

    def loop(self):
        if not self.started:
            self.drive(0.0, 0.0)
            if self.camera_up_time is None or time.time() - self.camera_up_time < START_DELAY:
                return
            self.started = True
            self.start_time = time.time()
            self.get_logger().info('GO!')

        # EKF predict: move the estimate by what we commanded during the last DT
        self.ekf.predict(self.v, self.w, DT)

        x, y, theta = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        lap = self.corner_index // len(self.corners) if self.corners else 0
        self.traj_file.write(f'{time.time() - self.start_time:.2f},{lap},'
                             f'{x:.4f},{y:.4f},{theta:.4f},{len(self.ekf.ids)},{self.v:.4f},{self.w:.4f}\n')

        if self.finished or not self.route:
            self.drive(0.0, 0.0)
            return
        if self.turn_steps_left > 0:
            # In the middle of a timed turn: keep the wheels turning
            self.publish_wheels(-self.turn_wheels, self.turn_wheels)
            self.turn_steps_left -= 1
            if self.turn_steps_left == 0:
                self.pause_until = time.time() + TURN_PAUSE  # turn done: stop briefly
                self.drive(0.0, 0.0)
            return
        if time.time() < self.pause_until:
            self.drive(0.0, 0.0)  # stopped at a corner, camera is correcting the estimate
            return

        goal_x, goal_y = self.route[self.corner_index]
        if math.hypot(goal_x - x, goal_y - y) > POSITION_TOLERANCE:
            # Pointing far off: timed turn in place. Otherwise drive and steer (same as HW2).
            angle_error = wrap(math.atan2(goal_y - y, goal_x - x) - theta)
            if abs(angle_error) > DRIVE_ANGLE_LIMIT:
                self.start_turn(angle_error)
            else:
                self.drive(DRIVE_SPEED, HEADING_GAIN * angle_error)
            return

        # Reached a corner: stop for a moment, then head to the next one
        self.corner_index += 1
        self.pause_until = time.time() + CORNER_PAUSE
        self.drive(0.0, 0.0)
        if self.corner_index % len(self.corners) == 0:
            lap = self.corner_index // len(self.corners)
            self.save_map(f'landmarks_lap{lap}.csv')
            self.get_logger().info(f'Lap {lap} done, {len(self.ekf.ids)} landmarks in map')
        if self.corner_index == len(self.route):
            self.finished = True
            self.save_map('landmarks_final.csv')
            self.get_logger().info(f'Route finished. Results in {self.run_dir}')

    def start_turn(self, angle):
        """Turn in place by `angle` (+ = left) at full strength: time = angle / TURN_RATE."""
        self.turn_steps_left = max(1, round(abs(angle) / TURN_RATE / DT))
        self.turn_wheels = TURN_STRENGTH if angle > 0 else -TURN_STRENGTH
        # What the EKF predicts each loop: the whole turn spread evenly over the turn time
        self.v, self.w = 0.0, angle / (self.turn_steps_left * DT)
        self.publish_wheels(-self.turn_wheels, self.turn_wheels)
        self.turn_steps_left -= 1

    def publish_wheels(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.motor_pub.publish(msg)

    def drive(self, v, w):
        # Same idea as HW2: speed (m/s) and turn rate (rad/s) -> wheel commands
        self.v, self.w = v, w
        forward = v * DRIVE_COMMAND / DRIVE_SPEED
        turn = w * TURN_STRENGTH / TURN_RATE
        self.publish_wheels(forward - turn, forward + turn)

    def save_map(self, name):
        with open(os.path.join(self.run_dir, name), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tag_id', 'x', 'y', 'var_x', 'cov_xy', 'var_y'])
            for tag_id, lx, ly, cov in self.ekf.landmarks():
                writer.writerow([tag_id, f'{lx:.4f}', f'{ly:.4f}',
                                 f'{cov[0, 0]:.6f}', f'{cov[0, 1]:.6f}', f'{cov[1, 1]:.6f}'])
        self.traj_file.flush()
        self.meas_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = SlamDrive()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C here, or the launch file shutting everything down
    if not node.finished:
        node.save_map('landmarks_final.csv')  # stopped early: still keep the map so far
    node.traj_file.close()
    node.meas_file.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
