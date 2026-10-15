"""Start autonomous control without the keyboard command publisher."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('waypoints_file', description='Absolute waypoint file path'),
        DeclareLaunchArgument('wheel_spacing', default_value='0.20'),
        DeclareLaunchArgument('left_speed_scale', default_value='1.0'),
        DeclareLaunchArgument('right_speed_scale', default_value='1.0'),
        DeclareLaunchArgument('imu_yaw_sign', default_value='1.0'),
    ]
    parameters = {
        'waypoints_file': ParameterValue(
            LaunchConfiguration('waypoints_file'), value_type=str),
    }
    for name in ('wheel_spacing', 'left_speed_scale', 'right_speed_scale',
                 'imu_yaw_sign'):
        parameters[name] = ParameterValue(LaunchConfiguration(name), value_type=float)

    motor = Node(package='robot_control', executable='motor_control', output='screen')
    controller = Node(
        package='robot_control', executable='waypoint_control', output='screen',
        parameters=[parameters])
    # Register before starting nodes so an early startup failure is handled too.
    handlers = [
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[Shutdown(reason='A waypoint-control process exited')]))
        for node in (motor, controller)
    ]
    return LaunchDescription(arguments + handlers + [motor, controller])
