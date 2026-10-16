"""Check the route added to the template motor driver without hardware."""

import importlib.util
import math
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

class FakeNode:
    def __init__(self, name):
        self.commands = []
        self.parameters = {}

    def declare_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def create_publisher(self, *args):
        return SimpleNamespace(publish=lambda msg: self.commands.append(msg.data))

    def create_subscription(self, *args):
        return None

    def get_logger(self):
        return SimpleNamespace(info=lambda text: None, debug=lambda text: None,
                               warn=lambda text: None, error=lambda text: None)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None))

    def destroy_node(self):
        return None

modules = {}
for name in ('rclpy', 'rclpy.node', 'sensor_msgs', 'sensor_msgs.msg',
             'std_msgs', 'std_msgs.msg', 'geometry_msgs', 'geometry_msgs.msg',
             'serial', 'numpy'):
    modules[name] = ModuleType(name)
modules['rclpy.node'].Node = FakeNode
modules['rclpy'].ok = lambda: True
modules['rclpy'].spin_once = lambda *args, **kwargs: None
modules['sensor_msgs.msg'].Imu = lambda: SimpleNamespace(
    header=SimpleNamespace(), orientation=SimpleNamespace(),
    angular_velocity=SimpleNamespace(), linear_acceleration=SimpleNamespace())
modules['std_msgs.msg'].Float32MultiArray = SimpleNamespace
modules['geometry_msgs.msg'].Vector3 = SimpleNamespace
modules['numpy'].clip = lambda value, low, high: max(low, min(high, value))
modules['serial'].Serial = lambda *args, **kwargs: SimpleNamespace()
path = Path(__file__).parents[1] / 'robot_control' / 'motor_control.py'
spec = importlib.util.spec_from_file_location('tested_motor_control', path)
controller = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', modules):
    spec.loader.exec_module(controller)

class WaypointTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        clock = patch.object(
            controller, 'time', SimpleNamespace(
                monotonic=lambda: self.now, time=lambda: self.now,
                sleep=lambda seconds: setattr(self, 'now', self.now + seconds)))
        clock.start()
        self.addCleanup(clock.stop)
        self.node = controller.MotorControllerNode.__new__(controller.MotorControllerNode)
        FakeNode.__init__(self.node, 'test')
        self.node.heading = 0.0
        self.node.robot = True
        self.node.command_lock = threading.Lock()
        self.node.timeout_active = False
        self.node.imu_publisher = SimpleNamespace(publish=lambda message: None)
        original_send = self.node.send

        def record_send(left, right):
            original_send(left, right)
            self.node.commands.append([
                self.node.motor_command['L'], self.node.motor_command['R']])

        self.node.send = record_send

    def test_explicit_route_matches_seven_poses(self):
        # Record the actual hardcoded calls and check their ideal geometry.
        node = self.node
        actions = []
        poses = []
        x, y, heading = 0.0, 0.0, 0.0

        def turn(angle):
            nonlocal heading
            heading = angle
            actions.append(('turn', angle))
            poses.append((x, y, heading))

        def drive(distance):
            nonlocal x, y
            x += distance * math.cos(heading)
            y += distance * math.sin(heading)
            actions.append(('drive', distance))
            poses.append((x, y, heading))

        node.turn = turn
        node.drive = drive
        node.run_route()
        targets = [(0, 0, 0), (1, 0, 0), (1, 1, 1.57), (2, 1, 0),
                   (2, 2, -1.57), (1, 1, -0.78), (0, 0, 0)]
        for index, target in zip((0, 3, 6, 9, 12, 17, 22), targets):
            for actual, expected in zip(poses[index], target):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(sum(action == 'drive' for action, _ in actions), 8)
        self.assertEqual(node.commands[-1], [0.0, 0.0])

    def test_default_mode_remains_manual(self):
        fake_thread = SimpleNamespace(start=lambda: None)
        with patch.object(controller.threading, 'Thread', return_value=fake_thread):
            node = controller.MotorControllerNode()
        self.assertFalse(node.run_waypoints)

    def test_drive_duration_and_stop(self):
        self.node.drive(1.0)
        self.assertAlmostEqual(self.now, 1.0 / controller.DRIVE_SPEED, delta=0.051)
        self.assertEqual(self.node.commands[0], [0.2, 0.2])
        self.assertEqual(self.node.commands[-1], [0.0, 0.0])

    def test_turn_uses_direct_angle_difference(self):
        self.node.heading = math.radians(179)
        self.node.turn(math.radians(-175))
        left, right = self.node.commands[0]
        self.assertGreater(left, 0)
        self.assertLess(right, 0)
        self.assertAlmostEqual(self.now, math.radians(354) / controller.TURN_RATE,
                               delta=0.051)
        self.assertEqual(self.node.commands[-1], [0.0, 0.0])

    def test_turn_right_and_no_turn_when_already_facing_target(self):
        self.node.turn(-math.pi / 2)
        left, right = self.node.commands[0]
        self.assertGreater(left, 0)
        self.assertLess(right, 0)
        previous_time = self.now
        self.node.commands.clear()
        self.node.turn(-math.pi / 2)
        self.assertEqual(self.now, previous_time)
        self.assertEqual(self.node.commands, [[0.0, 0.0]])

if __name__ == '__main__':
    unittest.main()
