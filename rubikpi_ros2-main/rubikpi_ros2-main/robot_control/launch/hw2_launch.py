from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

# the point of this is to start the 4 programs that are needed for the robot to work (theyre the 4 variables))
def generate_launch_description():
    camera = IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        # Undistort images using camera_parameter.yaml; the AprilTag node expects this.
        launch_arguments={'image_rectify': 'true'}.items())
    apriltag = Node(package='apriltag_ros', executable='apriltag_node', parameters=[{'family': '36h11', 'size': 0.10, 'image_transport': 'compressed'}], remappings=[('image_rect', '/camera/image_raw'), ('camera_info', '/camera/camera_info')])
    motor = Node(package='robot_control', executable='motor_control', output='screen')
    waypoints = Node(package='robot_control', executable='waypoint_control', output='screen')

    return LaunchDescription([camera, apriltag, motor, waypoints])
