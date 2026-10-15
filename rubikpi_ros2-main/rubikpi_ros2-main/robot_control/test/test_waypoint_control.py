"""Exercise the real controller with a simulated robot, without ROS installed."""

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import mock_open, patch


class FakeNode:
    """Only the ROS interfaces used by the controller."""

    parameters = {}

    def __init__(self, name):
        self.values = dict(self.parameters)
        self.messages = []

    def declare_parameter(self, name, default):
        self.values.setdefault(name, default)

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def create_publisher(self, *args):
        return SimpleNamespace(publish=lambda message: None)

    def create_subscription(self, *args):
        return None

    def create_timer(self, *args):
        return None

    def get_logger(self):
        return SimpleNamespace(info=self.messages.append, warning=self.messages.append)

    def destroy_node(self):
        return None


def load_controller():
    """Load in isolation so these stand-ins do not replace installed ROS."""
    modules = {}
    for name in ('rclpy', 'rclpy.node', 'sensor_msgs', 'sensor_msgs.msg',
                 'std_msgs', 'std_msgs.msg'):
        modules[name] = ModuleType(name)
    modules['rclpy.node'].Node = FakeNode
    modules['sensor_msgs.msg'].Imu = SimpleNamespace
    modules['std_msgs.msg'].Float32MultiArray = SimpleNamespace
    path = Path(__file__).parents[1] / 'robot_control' / 'waypoint_control.py'
    spec = importlib.util.spec_from_file_location('tested_waypoint_control', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    return module


controller = load_controller()
WAYPOINTS = Path(__file__).parents[2] / 'waypoints.txt'


class WaypointTests(unittest.TestCase):
    """Check actual stage transitions, motor bounds, and fault stops."""

    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(
            controller, 'time', SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.addCleanup(self.clock.stop)
        FakeNode.parameters = {'waypoints_file': str(WAYPOINTS)}

    def imu(self, node, heading):
        # A nonzero initial yaw verifies the relative-heading reference.
        heading += 0.7
        node.read_imu(SimpleNamespace(
            orientation_covariance=[0.0] * 9,
            orientation=SimpleNamespace(
                x=0.0, y=0.0, z=math.sin(heading / 2), w=math.cos(heading / 2))))

    def test_complete_route(self):
        # Different left/right calibration and saturation exercise command units.
        FakeNode.parameters.update(left_speed_scale=0.25, right_speed_scale=0.35)
        node = controller.WaypointController()
        x, y, heading = 0.0, 0.0, 0.0
        self.imu(node, heading)
        arrivals = []
        for step in range(20000):
            dt = 0.05
            left = node.left_command * node.left_speed_scale
            right = node.right_command * node.right_speed_scale
            speed = (left + right) / 2
            omega = (right - left) / node.wheel_spacing
            x += speed * math.cos(heading + omega * dt / 2) * dt
            y += speed * math.sin(heading + omega * dt / 2) * dt
            heading = controller.wrap_angle(heading + omega * dt)
            self.now += dt
            self.imu(node, heading)
            previous_index = node.waypoint_index
            node.update()
            self.assertLessEqual(abs(node.left_command), 0.5)
            self.assertLessEqual(abs(node.right_command), 0.5)
            if node.waypoint_index != previous_index:
                target_x, target_y, target_heading = node.waypoints[previous_index]
                self.assertAlmostEqual(x, target_x, delta=0.06)
                self.assertAlmostEqual(y, target_y, delta=0.06)
                self.assertLessEqual(
                    abs(controller.wrap_angle(heading - target_heading)),
                    node.angle_tolerance)
                arrivals.append(previous_index)
            if node.finished:
                break
        self.assertEqual(arrivals, list(range(7)), node.messages)
        self.assertTrue(node.finished)
        self.assertEqual((node.left_command, node.right_command), (0.0, 0.0))

    def test_imu_loss_stops_and_does_not_resume(self):
        node = controller.WaypointController()
        self.imu(node, 0.0)
        node.send_motion(0.1, 0.0)
        node.last_update = 0.31
        self.now = 0.35
        node.update()
        self.assertTrue(node.finished)
        self.imu(node, 0.0)
        self.now += 0.05
        node.update()
        self.assertEqual((node.left_command, node.right_command), (0.0, 0.0))

    def test_missed_control_deadline_stops(self):
        node = controller.WaypointController()
        self.now = 0.2
        self.imu(node, 0.0)
        node.update()
        self.assertTrue(node.finished)
        self.assertIn('motor timeout', node.messages[-1])

    def test_no_imu_never_moves(self):
        node = controller.WaypointController()
        self.now = node.startup_timeout + 1
        node.update()
        self.assertTrue(node.finished)
        self.assertEqual((node.left_command, node.right_command), (0.0, 0.0))

    def test_invalid_imu_does_not_start(self):
        node = controller.WaypointController()
        node.read_imu(SimpleNamespace(
            orientation_covariance=[0.0] * 9,
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)))
        self.assertIsNone(node.last_imu_time)

    def test_angle_wrap(self):
        error = controller.wrap_angle(math.radians(-179) - math.radians(179))
        self.assertAlmostEqual(error, math.radians(2))

    def test_bad_waypoint_files(self):
        for text in ('', '1,2\n', '1,2,nan\n', '1,2,inf\n', '1,2,no\n'):
            with self.subTest(text=text):
                with patch('builtins.open', mock_open(read_data=text)):
                    with self.assertRaises(ValueError):
                        controller.read_waypoints('unused.txt')

    def test_waypoint_timeout(self):
        node = controller.WaypointController()
        node.waypoint_start = 0.0
        self.now = node.waypoint_timeout + 1
        node.last_update = self.now - 0.05
        self.imu(node, 0.0)
        node.update()
        self.assertTrue(node.finished)
        self.assertIn('Waypoint timed out', node.messages[-1])

    def test_driver_stops_hardware_after_ros_shutdown(self):
        modules = {}
        for name in ('rclpy', 'rclpy.node', 'sensor_msgs', 'sensor_msgs.msg',
                     'std_msgs', 'std_msgs.msg', 'geometry_msgs',
                     'geometry_msgs.msg', 'serial', 'numpy'):
            modules[name] = ModuleType(name)
        modules['rclpy'].ok = lambda: False
        modules['rclpy.node'].Node = FakeNode
        modules['sensor_msgs.msg'].Imu = SimpleNamespace
        modules['std_msgs.msg'].Float32MultiArray = SimpleNamespace
        modules['geometry_msgs.msg'].Vector3 = SimpleNamespace
        path = Path(__file__).parents[1] / 'robot_control' / 'motor_control.py'
        spec = importlib.util.spec_from_file_location('tested_motor_control', path)
        driver = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', modules):
            spec.loader.exec_module(driver)
        node = driver.MotorControllerNode.__new__(driver.MotorControllerNode)
        FakeNode.__init__(node, 'test')
        events = []
        thread = SimpleNamespace(join=lambda timeout: events.append('join'))
        node.motor_tx_thread = thread
        node.imu_tx_thread = thread
        node.rx_thread = thread
        node.robot = SimpleNamespace(
            write=lambda data: events.append(data),
            close=lambda: events.append('close'))
        node.destroy_node()
        self.assertFalse(node.running)
        self.assertEqual(events[:3], ['join'] * 3)
        self.assertIn(b'"L": 0.0', events[3])
        self.assertIn(b'"R": 0.0', events[3])
        self.assertEqual(events[4], 'close')


if __name__ == '__main__':
    unittest.main()
