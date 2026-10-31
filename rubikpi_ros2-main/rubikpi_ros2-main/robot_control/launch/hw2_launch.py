"""HW2: starts camera, AprilTag detector, motor driver, and waypoint controller.

Place the robot at (0, 0) facing +x BEFORE launching. It starts driving on its
own a few seconds after the camera is up.
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
            FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        # Undistort images using camera_parameter.yaml; the AprilTag node expects this.
        launch_arguments={'image_rectify': 'true'}.items())

    # Same as: ros2 run apriltag_ros apriltag_node --ros-args -p family:=36h11 -p size:=0.10 ...
    # Reads the JPEG-compressed images: the full 1280x720 raw images (2.7 MB each)
    # only arrive about 1-2 times per second between processes; compressed gets all 10.
    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        parameters=[{'family': '36h11', 'size': 0.10, 'image_transport': 'compressed'}],
        remappings=[('image_rect', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info')])

    motor = Node(package='robot_control', executable='motor_control', output='screen')

    waypoints = Node(package='robot_control', executable='waypoint_control', output='screen')

    return LaunchDescription([camera, apriltag, motor, waypoints])
