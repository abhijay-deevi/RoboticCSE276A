"""HW3: camera, AprilTag detector, motor driver and the SLAM driver in one command.

  ros2 launch robot_control hw3_launch.py shape:=square side:=1.22 laps:=3
  ros2 launch robot_control hw3_launch.py shape:=octagon side:=0.61 laps:=3
  ros2 launch robot_control hw3_launch.py shape:=none     (robot stays still: noise test)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('shape', default_value='square'),
        DeclareLaunchArgument('side', default_value='1.22'),  # meters (4 ft)
        DeclareLaunchArgument('laps', default_value='3'),
    ]

    # Same camera setup as HW2: undistorted images using camera_parameter.yaml
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        launch_arguments={'image_rectify': 'true'}.items())

    # Same as HW2, but our HW3 tags have a 4 in (0.1016 m) black square (measured).
    # decimate 1.0: search the full-resolution image (default 2.0 halves it, and a
    # 5 in tag 1.7 m away is then only ~16 pixels wide -> mostly missed).
    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        parameters=[{'family': '36h11', 'size': 0.1016, 'image_transport': 'compressed',
                     'detector.decimate': 1.0, 'detector.threads': 4}],
        remappings=[('image_rect', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info')])

    motor = Node(package='robot_control', executable='motor_control', output='screen')

    slam = Node(
        package='robot_control',
        executable='slam_drive',
        output='screen',
        parameters=[{
            'shape': LaunchConfiguration('shape'),
            'side': ParameterValue(LaunchConfiguration('side'), value_type=float),
            'laps': ParameterValue(LaunchConfiguration('laps'), value_type=int),
        }])

    return LaunchDescription(args + [camera, apriltag, motor, slam])
