"""HW3: find a slow turn that works on THIS floor, and measure how fast it turns.

Drives the square (or octagon) once from the measured start pose. At the start and at
every corner it:
  1. turns slowly in place while the camera keeps measuring its real heading
     (known tag positions from hw3_ground_truth.yaml)
  2. stops as soon as it faces the next side (within 3 deg)
  3. records how fast it really turned (degrees turned / seconds)
  4. drives the side (forward moves are accurate)
If the robot does not rotate at all, the slow wheel command is raised a little.
The command and the measured turn rate are saved to ~/hw3_turn_calibration.yaml;
slam_drive uses them.

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
from robot_control.motor_control import DRIVE_COMMAND, DRIVE_SPEED
from robot_control.slam_drive import (CAMERA_DELAY, CAMERA_OFFSET, SLOW_TURN_COMMAND,
                                      SLOW_TURN_RATE, TURN_TOLERANCE, path_corners)

CALIBRATION_FILE = os.path.expanduser('~/hw3_turn_calibration.yaml')
SEND_PERIOD = 0.05     # resend motor commands (driver stops after 0.15 s)
SETTLE_TIME = 0.5      # seconds to stand still before a stationary heading measurement
LOOK_TIME = 1.0        # seconds of camera readings for a stationary heading measurement
FRESH = 0.3            # seconds: a camera heading newer than this is used directly
STUCK_TIME = 1.5       # seconds without turning -> raise the slow command
COMMAND_STEP = 0.05    # how much to raise the turn command when stuck
MAX_COMMAND = 0.50     # motor driver maximum
TURN_TIMEOUT = 20.0    # seconds: give up on one turn after this long


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

        self.position = None       # where the robot is standing (world), set per corner
        self.last_heading = None   # (time, heading) of the latest camera heading
        self.readings = None       # headings collected during a stationary look, else None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

    def on_detections(self, msg):
        """Real heading = direction to a known tag minus where it appears in the camera."""
        if self.position is None:
            return
        headings = []
        for det in msg.detections:
            if det.id not in self.tags:
                continue
            try:
                t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{det.id}', Time())
            except Exception:
                continue
            forward, left = t.transform.translation.z + CAMERA_OFFSET, -t.transform.translation.x
            tx, ty = self.tags[det.id]
            headings.append(math.atan2(ty - self.position[1], tx - self.position[0])
                            - math.atan2(left, forward))
        if headings:
            heading = math.atan2(sum(math.sin(h) for h in headings),
                                 sum(math.cos(h) for h in headings))
            self.last_heading = (time.monotonic(), heading)
            if self.readings is not None:
                self.readings.append(heading)

    # ---------------- Helpers ----------------

    def send(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.motor_pub.publish(msg)

    def hold(self, left, right, seconds):
        """Send the same wheel command for `seconds` (same idea as HW1's move())."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(left, right)
            time.sleep(SEND_PERIOD)
        self.send(0.0, 0.0)

    def to_world(self, px, py):
        """Path point (path frame: start = origin, facing +x) -> world frame."""
        c, s = math.cos(self.start['theta']), math.sin(self.start['theta'])
        return (self.start['x'] + c * px - s * py, self.start['y'] + s * px + c * py)

    def look(self):
        """Stand still and average the camera heading for a moment. None if no tags."""
        time.sleep(SETTLE_TIME)
        self.readings = []
        time.sleep(LOOK_TIME)
        readings, self.readings = self.readings, None
        if not readings:
            return None
        return math.atan2(sum(math.sin(h) for h in readings), sum(math.cos(h) for h in readings))

    def slow_turn_to(self, target, start_heading, command, rate):
        """Turn slowly until the heading is within TURN_TOLERANCE of `target`.

        Between camera readings (e.g. no tag in view) the heading is predicted from
        the last reading and the turn rate.
        Returns (seconds turned, command used, whether the command had to be raised).
        """
        heading, began = start_heading, time.monotonic()
        last_progress = (began, start_heading)
        moving, raised = False, False
        while time.monotonic() - began < TURN_TIMEOUT:
            now = time.monotonic()
            if self.last_heading and now - self.last_heading[0] < FRESH:
                heading = self.last_heading[1]
            else:
                heading = wrap(heading + math.copysign(rate * SEND_PERIOD, wrap(target - heading)))
            error = wrap(target - heading)
            # The camera lags, so once we are rotating, stop a little early
            if abs(error) <= TURN_TOLERANCE + (rate * CAMERA_DELAY if moving else 0.0):
                break
            if abs(wrap(heading - last_progress[1])) > math.radians(2):
                last_progress = (now, heading)
                moving = True
            elif now - last_progress[0] > STUCK_TIME and command < MAX_COMMAND:
                command = min(MAX_COMMAND, command + COMMAND_STEP)  # not rotating: push harder
                last_progress = (now, heading)
                raised = True
                self.get_logger().info(f'Not turning, raising slow command to {command:.2f}')
            direction = 1.0 if error > 0 else -1.0
            self.send(-direction * command, direction * command)
            time.sleep(SEND_PERIOD)
        self.send(0.0, 0.0)
        return time.monotonic() - began, command, raised

    # ---------------- Calibration drive ----------------

    def run(self):
        command, rate = SLOW_TURN_COMMAND, SLOW_TURN_RATE
        samples = []
        corners = [(0.0, 0.0)] + path_corners(self.shape, self.side)
        turn = math.pi / 2 if self.shape == 'square' else math.pi / 4

        self.position = self.to_world(0.0, 0.0)
        while self.look() is None:
            self.get_logger().info('Waiting for the camera to see a tag...')

        for k in range(len(corners) - 1):
            self.position = self.to_world(*corners[k])
            target = wrap(self.start['theta'] + k * turn)  # direction of the next side
            heading = self.look()
            if heading is None:
                self.get_logger().warning(f'Corner {k}: no tags visible, not turning')
            elif abs(wrap(target - heading)) > TURN_TOLERANCE:
                seconds, command, raised = self.slow_turn_to(target, heading, command, rate)
                if raised:
                    samples.clear()  # rates measured with the old, too-weak command don't count
                after = self.look()
                if after is not None:
                    turned = wrap(after - heading)
                    # Only full corner turns at an unchanged command give a good rate
                    if abs(turned) > math.radians(20) and not raised:
                        samples.append(abs(turned) / seconds)
                        rate = statistics.mean(samples)
                    self.get_logger().info(
                        f'Corner {k}: turned {math.degrees(turned):+.1f} deg in {seconds:.2f} s, '
                        f'now {math.degrees(wrap(target - after)):+.1f} deg from target. '
                        f'Slow command {command:.2f}, rate {rate:.3f} rad/s')
            self.hold(DRIVE_COMMAND, DRIVE_COMMAND, self.side / DRIVE_SPEED)

        self.send(0.0, 0.0)
        result = {}
        if os.path.exists(CALIBRATION_FILE):  # keep results from the other shape
            with open(CALIBRATION_FILE) as f:
                result = yaml.safe_load(f) or {}
        result.update({'slow_command': round(command, 3), 'slow_rate': round(rate, 4),
                       f'samples_{self.shape}': [round(r, 4) for r in samples]})
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
        node.send(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
