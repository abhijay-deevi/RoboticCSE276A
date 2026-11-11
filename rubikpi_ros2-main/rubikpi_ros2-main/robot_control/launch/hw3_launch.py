from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [ # basically just sets info for the run
        DeclareLaunchArgument('shape', default_value='square'),
        DeclareLaunchArgument('side', default_value='1.22'),  # meters (4 ft)
        DeclareLaunchArgument('laps', default_value='3'),
    ]

    # Same camera setup as HW2
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        launch_arguments={'image_rectify': 'true'}.items())

    # Same as HW2 we fixed the inches of the tag since it was wrong initially
    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        parameters=[{'family': '36h11', 'size': 0.1016, 'image_transport': 'compressed',
                     'detector.decimate': 1.0, 'detector.threads': 4}],
        remappings=[('image_rect', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info')])

    motor = Node(package='robot_control', executable='motor_control', output='screen')

    slam = Node( # this handles the 3 arguments from bfore
        package='robot_control',
        executable='slam_drive',
        output='screen',
        parameters=[{
            'shape': LaunchConfiguration('shape'),
            'side': ParameterValue(LaunchConfiguration('side'), value_type=float),
            'laps': ParameterValue(LaunchConfiguration('laps'), value_type=int),
        }])

    return LaunchDescription(args + [camera, apriltag, motor, slam])
