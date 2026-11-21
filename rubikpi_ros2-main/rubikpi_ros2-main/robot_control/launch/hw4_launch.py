"""HW4: camera, AprilTag detector, motor driver and the path follower in one command.

  ros2 launch robot_control hw4_launch.py path:=safe     (Voronoi + A*)
  ros2 launch robot_control hw4_launch.py path:=short    (visibility graph + A*)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    path = DeclareLaunchArgument('path', default_value='safe')

    # Same camera and tag detector as HW3 (4 in tags)
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        launch_arguments={'image_rectify': 'true'}.items())

    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        parameters=[{'family': '36h11', 'size': 0.1016, 'image_transport': 'compressed',
                     'detector.decimate': 1.0, 'detector.threads': 4}],
        remappings=[('image_rect', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info')])

    motor = Node(package='robot_control', executable='motor_control', output='screen')

    follow = Node(package='robot_control', executable='path_follow', output='screen',
                  parameters=[{'path': LaunchConfiguration('path')}])

    return LaunchDescription([path, camera, apriltag, motor, follow])
