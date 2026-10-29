"""HW2: starts camera, AprilTag detector, motor driver and localization.

Start waypoint_control separately once the robot is placed at (0, 0) facing +x.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])))

    # Same as: ros2 run apriltag_ros apriltag_node --ros-args -p family:=36h11 -p size:=0.10 ...
    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        parameters=[{'family': '36h11', 'size': 0.10}],
        remappings=[('image_rect', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info')])

    motor = Node(package='robot_control', executable='motor_control', output='screen')
    localization = Node(package='robot_control', executable='localization', output='screen')

    return LaunchDescription([camera, apriltag, motor, localization])
