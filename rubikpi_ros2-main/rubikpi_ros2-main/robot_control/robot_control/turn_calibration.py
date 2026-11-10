"""HW3: measure how fast the robot really turns on THIS floor, using the tags.

Drives the square (or octagon) once from the measured start pose. At the start and at
every corner it:
  1. measures its real heading with the camera (known tag positions from hw3_ground_truth.yaml)
  2. turns toward the next side for (angle / turn rate) seconds
  3. measures again -> how far it REALLY turned -> updates the turn rate
  4. if still more than 3 deg off, does small correction turns until it faces the right way
  5. drives the side (forward moves are accurate)
The measured rates are saved to ~/hw3_turn_calibration.yaml; slam_drive uses them.

This uses the known tag map ONLY to calibrate the turns; the SLAM runs never see it.
"""

import math
import os
import statistics
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener

from robot_control.ekf_slam import wrap
from robot_control.motor_control import (DRIVE_COMMAND, DRIVE_SPEED, TURN_COMMAND,
                                         LEFT_TURN_RATE, LEFT_45_TURN_RATE)
from robot_control.slam_drive import CAMERA_OFFSET, path_corners

CALIBRATION_FILE = os.path.expanduser('~/hw3_turn_calibration.yaml')
HEADING_TOLERANCE = math.radians(3)  # close enough to the right direction
MAX_CORRECTIONS = 4                  # small correction turns per corner, at most
LOOK_TIME = 1.0                      # seconds of camera readings per heading measurement
SETTLE_TIME = 0.5                    # seconds to stand still before measuring
SEND_PERIOD = 0.05                   # resend motor commands (driver stops after 0.15 s)


class TurnCalibration(Node):
    def __init__(self):
        super().__init__('turn_calibration')
        self.shape = self.declare_parameter('shape', 'square').value
        self.side = self.declare_parameter('side', 1.22).value

        truth_file = os.path.join(get_package_share_directory('robot_control'),
                                  'config', 'hw3_ground_truth.yaml')
        with open(truth_file) as f:
            truth = yaml.safe_load(f)
        self.tags = {int(k): (v['x'], v['y']) for k, v in truth['tags'].items()}
        self.start = truth['start'][self.shape]

        self.readings = None  # list of (tag_id, forward, left) while looking, else None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

    def on_detections(self, msg):
        if self.readings is None:
            return
        for det in msg.detections:
            if det.id not in self.tags:
                continue
            try:
                t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{det.id}', Time())
            except Exception:
                continue
            self.readings.append((det.id, t.transform.translation.z + CAMERA_OFFSET,
                                  -t.transform.translation.x))

    # ---------------- Helpers ----------------

    def hold(self, left, right, seconds):
        """Send the same wheel command for `seconds` (same idea as HW1's move())."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            msg = Float32MultiArray()
            msg.data = [float(left), float(right)]
            self.motor_pub.publish(msg)
            time.sleep(SEND_PERIOD)
        self.stop()

    def stop(self):
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.motor_pub.publish(msg)

    def to_world(self, px, py):
        """Path point (path frame: start = origin, facing +x) -> world frame."""
        c, s = math.cos(self.start['theta']), math.sin(self.start['theta'])
        return (self.start['x'] + c * px - s * py, self.start['y'] + s * px + c * py)

    def measure_heading(self, position):
        """Real heading at `position` (world): direction to each visible tag minus its bearing."""
        time.sleep(SETTLE_TIME)
        self.readings = []
        time.sleep(LOOK_TIME)
        readings, self.readings = self.readings, None
        if not readings:
            return None
        rx, ry = position
        headings = [math.atan2(self.tags[i][1] - ry, self.tags[i][0] - rx) - math.atan2(left, forward)
                    for i, forward, left in readings]
        return math.atan2(sum(math.sin(h) for h in headings), sum(math.cos(h) for h in headings))

    def turn(self, angle, rate):
        """Turn in place by `angle` (radians, + = left) assuming `rate`. Returns seconds used."""
        seconds = abs(angle) / rate
        direction = 1 if angle > 0 else -1
        self.hold(-direction * TURN_COMMAND, direction * TURN_COMMAND, seconds)
        return seconds

    # ---------------- Calibration drive ----------------

    def run(self):
        while self.measure_heading(self.to_world(0.0, 0.0)) is None:
            self.get_logger().info('Waiting for the camera to see a tag...')
        big = self.shape == 'square'  # square corners are 90 deg turns, octagon 45 deg
        main_rate = LEFT_TURN_RATE if big else LEFT_45_TURN_RATE  # starting guess, then measured
        small_rate = LEFT_45_TURN_RATE                            # for correction turns
        main_samples, small_samples = [], []

        corners = [(0.0, 0.0)] + path_corners(self.shape, self.side)
        turn = math.pi / 2 if big else math.pi / 4
        for k in range(len(corners) - 1):
            position = self.to_world(*corners[k])
            target = wrap(self.start['theta'] + k * turn)  # direction of the next side
            heading = self.measure_heading(position)
            if heading is None:
                self.get_logger().warning(f'Corner {k}: no tags visible, skipping heading check')
            else:
                for attempt in range(MAX_CORRECTIONS + 1):
                    needed = wrap(target - heading)
                    if abs(needed) < HEADING_TOLERANCE:
                        break
                    is_main = abs(needed) > math.radians(20)
                    seconds = self.turn(needed, main_rate if is_main else small_rate)
                    new_heading = self.measure_heading(position)
                    if new_heading is None:
                        break
                    turned = wrap(new_heading - heading)
                    if abs(turned) > math.radians(1):
                        (main_samples if is_main else small_samples).append(abs(turned) / seconds)
                        if is_main:
                            main_rate = statistics.mean(main_samples)
                        else:
                            small_rate = statistics.mean(small_samples)
                    self.get_logger().info(
                        f'Corner {k}: wanted {math.degrees(needed):+.1f} deg, turned '
                        f'{math.degrees(turned):+.1f} deg in {seconds:.2f} s -> rates now '
                        f'main {main_rate:.3f}, small {small_rate:.3f} rad/s')
                    heading = new_heading
            self.hold(DRIVE_COMMAND, DRIVE_COMMAND, self.side / DRIVE_SPEED)

        self.stop()
        result = {}
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE) as f:
                result = yaml.safe_load(f) or {}
        result['rate_90' if big else 'rate_45'] = round(main_rate, 4)
        if small_samples:
            result['rate_small'] = round(small_rate, 4)
        result[f'samples_{self.shape}'] = [round(r, 4) for r in main_samples]
        with open(CALIBRATION_FILE, 'w') as f:
            yaml.safe_dump(result, f)
        self.get_logger().info(f'Calibration done: {result}. Saved to {CALIBRATION_FILE}')


def main(args=None):
    rclpy.init(args=args)
    node = TurnCalibration()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
