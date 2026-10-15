"""Visit waypoints by moving along x, then y, then turning to the final angle.

Position is dead reckoned from calibrated motor commands and IMU heading.
The robot must start at (0, 0), facing the positive x direction.
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


def wrap_angle(angle):
    """Return an angle between -pi and pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def read_waypoints(filename):
    """Read x, y, angle rows, allowing empty lines and # comments."""
    waypoints = []
    with open(filename, encoding='utf-8') as waypoint_file:
        for line_number, line in enumerate(waypoint_file, start=1):
            line = line.split('#')[0].strip()
            if not line:
                continue
            values = line.split(',')
            if len(values) != 3:
                raise ValueError(f'Line {line_number}: expected x,y,angle')
            x, y, angle = [float(value) for value in values]
            if not all(math.isfinite(value) for value in (x, y, angle)):
                raise ValueError(f'Line {line_number}: values must be finite')
            waypoints.append((x, y, wrap_angle(angle)))
    if not waypoints:
        raise ValueError('The waypoint file is empty')
    return waypoints


class WaypointController(Node):
    """A small controller with one stage variable and a 20 Hz update loop."""

    def __init__(self):
        super().__init__('waypoint_controller')

        # Trial values: measure motor speeds and wheel spacing on your robot.
        defaults = {
            'waypoints_file': '',
            'wheel_spacing': 0.20,
            'left_speed_scale': 1.0,
            'right_speed_scale': 1.0,
            'imu_yaw_sign': 1.0,
            'position_tolerance': 0.05,
            'angle_tolerance': 0.087,
            'drive_speed': 0.12,
            'minimum_speed': 0.04,
            'maximum_turn_rate': 0.6,
            'minimum_turn_rate': 0.2,
            'distance_gain': 0.6,
            'turn_gain': 1.5,
            'imu_timeout': 0.3,
            'startup_timeout': 10.0,
            'waypoint_timeout': 120.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        # Keep parameters as ordinary attributes so the equations stay readable.
        for name in defaults:
            setattr(self, name, self.get_parameter(name).value)
        for name in defaults:
            if name not in ('waypoints_file', 'imu_yaw_sign'):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
        if self.imu_yaw_sign not in (-1.0, 1.0):
            raise ValueError('imu_yaw_sign must be 1.0 or -1.0')
        if self.minimum_speed > self.drive_speed:
            raise ValueError('minimum_speed must not exceed drive_speed')
        if self.minimum_turn_rate > self.maximum_turn_rate:
            raise ValueError('minimum_turn_rate exceeds maximum_turn_rate')
        if not 0 < self.angle_tolerance < math.pi / 4:
            raise ValueError('angle_tolerance must be between 0 and pi/4')
        self.waypoints = read_waypoints(self.waypoints_file)

        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)
        self.subscription = self.create_subscription(
            Imu, 'imu_data', self.read_imu, 10)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.heading_offset = None
        self.last_imu_time = None
        self.left_command = 0.0
        self.right_command = 0.0
        self.waypoint_index = 0
        self.stage = 'TURN_X'
        self.drive_heading = 0.0
        self.finished = False
        self.start_time = time.monotonic()
        self.last_update = self.start_time
        self.waypoint_start = None
        self.timer = self.create_timer(0.05, self.update)
        self.get_logger().warning(
            'Dead reckoning uses estimated motor speeds. Calibrate speed scales '
            'and wheel_spacing. Start at (0,0), facing +x. '
            'Do not run the keyboard controller at the same time.')

    def read_imu(self, message):
        """Use orientation yaw, relative to the first valid IMU measurement."""
        if message.orientation_covariance[0] == -1:
            return
        q = message.orientation
        values = (q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values):
            return
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 0.000001:
            return
        x, y, z, w = [value / norm for value in values]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw *= self.imu_yaw_sign
        if self.heading_offset is None:
            self.heading_offset = yaw
        self.theta = wrap_angle(yaw - self.heading_offset)
        self.last_imu_time = time.monotonic()

    def send_motion(self, speed, turn_rate):
        """Convert body motion to calibrated, bounded left/right commands."""
        # Differential-drive model: vL = v - b*omega/2; vR = v + b*omega/2.
        left_speed = speed - self.wheel_spacing * turn_rate / 2
        right_speed = speed + self.wheel_spacing * turn_rate / 2
        left = left_speed / self.left_speed_scale
        right = right_speed / self.right_speed_scale

        # Match the driver's +/-0.5 limit while preserving the wheel ratio.
        scale = max(1.0, abs(left) / 0.5, abs(right) / 0.5)
        self.left_command = left / scale
        self.right_command = right / scale
        message = Float32MultiArray()
        message.data = [self.left_command, self.right_command]
        self.publisher.publish(message)

    def stop(self, reason):
        """Latch a stop; restart the program to resume after a fault."""
        self.send_motion(0.0, 0.0)
        self.finished = True
        self.get_logger().info(reason)

    def turn_command(self, error):
        """Proportional turn speed, with a minimum to overcome friction."""
        rate = min(self.maximum_turn_rate, self.turn_gain * abs(error))
        rate = max(self.minimum_turn_rate, rate)
        return math.copysign(rate, error)

    def update(self):
        """Estimate motion, then perform one step of the current stage."""
        now = time.monotonic()
        dt = now - self.last_update
        self.last_update = now
        if self.finished:
            self.send_motion(0.0, 0.0)
            return
        if self.last_imu_time is None:
            self.send_motion(0.0, 0.0)
            if now - self.start_time > self.startup_timeout:
                self.stop('No valid IMU orientation received. Stopping.')
            return
        if now - self.last_imu_time > self.imu_timeout:
            self.stop('IMU data is stale. Stopping.')
            return
        if dt > 0.15:
            self.stop('Control loop missed the motor timeout. Stopping.')
            return

        # Estimate travel from the commands sent during the previous interval.
        # v = (vR + vL)/2; heading comes from the IMU instead of integrating omega.
        left_speed = self.left_command * self.left_speed_scale
        right_speed = self.right_command * self.right_speed_scale
        speed = (left_speed + right_speed) / 2
        self.x += speed * math.cos(self.theta) * dt
        self.y += speed * math.sin(self.theta) * dt

        if self.waypoint_start is None:
            self.waypoint_start = now
            self.get_logger().info(
                f'Waypoint {self.waypoint_index + 1}: '
                f'{self.waypoints[self.waypoint_index]}')
        if now - self.waypoint_start > self.waypoint_timeout:
            self.stop('Waypoint timed out. Check motion and calibration.')
            return

        target_x, target_y, target_angle = self.waypoints[self.waypoint_index]
        dx = target_x - self.x
        dy = target_y - self.y

        if self.stage in ('TURN_X', 'TURN_Y'):
            if self.stage == 'TURN_X':
                error = dx
                desired_heading = 0.0 if dx >= 0 else math.pi
                next_stage = 'TURN_Y'
                drive_stage = 'DRIVE_X'
            else:
                error = dy
                desired_heading = math.pi / 2 if dy >= 0 else -math.pi / 2
                next_stage = 'ALIGN'
                drive_stage = 'DRIVE_Y'
            if abs(error) <= self.position_tolerance:
                self.stage = next_stage
                self.send_motion(0.0, 0.0)
                return
            heading_error = wrap_angle(desired_heading - self.theta)
            if abs(heading_error) <= self.angle_tolerance:
                self.drive_heading = desired_heading
                self.stage = drive_stage
                self.send_motion(0.0, 0.0)
            else:
                self.send_motion(0.0, self.turn_command(heading_error))

        elif self.stage in ('DRIVE_X', 'DRIVE_Y'):
            if self.stage == 'DRIVE_X':
                remaining = dx * math.cos(self.drive_heading)
                next_stage = 'TURN_Y'
                turn_stage = 'TURN_X'
            else:
                remaining = dy * math.sin(self.drive_heading)
                next_stage = 'ALIGN'
                turn_stage = 'TURN_Y'
            heading_error = wrap_angle(self.drive_heading - self.theta)
            if abs(remaining) <= self.position_tolerance:
                self.stage = next_stage
                self.send_motion(0.0, 0.0)
            elif remaining < 0 or abs(heading_error) > 2 * self.angle_tolerance:
                self.stage = turn_stage
                self.send_motion(0.0, 0.0)
            else:
                speed = min(self.drive_speed, self.distance_gain * remaining)
                speed = max(self.minimum_speed, speed)
                turn_rate = self.turn_gain * heading_error
                turn_rate = max(-self.maximum_turn_rate,
                                min(self.maximum_turn_rate, turn_rate))
                self.send_motion(speed, turn_rate)

        elif self.stage == 'ALIGN':
            error = wrap_angle(target_angle - self.theta)
            if abs(error) > self.angle_tolerance:
                self.send_motion(0.0, self.turn_command(error))
                return
            self.send_motion(0.0, 0.0)
            if max(abs(dx), abs(dy)) > self.position_tolerance:
                self.stage = 'TURN_X'
                return
            self.get_logger().info(
                f'Reached waypoint {self.waypoint_index + 1}; estimated pose: '
                f'({self.x:.3f}, {self.y:.3f}, {self.theta:.3f})')
            self.waypoint_index += 1
            if self.waypoint_index == len(self.waypoints):
                self.stop('All waypoints completed. Motors stopped.')
            else:
                self.stage = 'TURN_X'
                self.waypoint_start = None


def main(args=None):
    """Run the controller and stop the robot before normal shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.send_motion(0.0, 0.0)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
