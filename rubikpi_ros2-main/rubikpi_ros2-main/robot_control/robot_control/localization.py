"""Localization node: estimates the robot pose (x, y, theta) from AprilTags.

Each time the camera sees tags:
  1. read which tag IDs are visible (/detections)
  2. look up where each tag is relative to the camera (TF camera_frame -> tag36h11:<id>)
  3. look up where that tag is in the world (tag_map.yaml)
  4. work backwards to where the robot must be
  5. average over all visible tags
  6. publish the result on /robot_pose

If no known tag is visible, nothing is published. The controller then keeps
using its own dead-reckoning estimate until a tag is seen again.
"""

import math
import os

import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose2D
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory

CAMERA_OFFSET = 0.09  # meters the camera sits in front of the robot center


class Localization(Node):
    def __init__(self):
        super().__init__('localization')

        # Load the tag map: {tag_id: {'x': .., 'y': .., 'yaw': ..}}
        map_file = os.path.join(get_package_share_directory('robot_control'),
                                'config', 'tag_map.yaml')
        with open(map_file) as f:
            self.tag_map = yaml.safe_load(f)['tags']
        self.get_logger().info(f'Loaded {len(self.tag_map)} tags from {map_file}')

        # TF listener: receives the camera -> tag transforms from the apriltag node
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(AprilTagDetectionArray, '/detections',
                                 self.on_detections, 10)
        self.pose_pub = self.create_publisher(Pose2D, '/robot_pose', 10)

    def on_detections(self, msg):
        estimates = []  # one (x, y, theta) guess per visible tag
        for det in msg.detections:
            if det.id not in self.tag_map:
                continue  # tag not on our map, ignore it
            estimate = self.pose_from_tag(det.id)
            if estimate is not None:
                estimates.append(estimate)

        if not estimates:
            return  # no usable tag: publish nothing

        # Average the guesses. Angles are averaged with sin/cos so that
        # +179 deg and -179 deg average to 180 deg, not 0 deg.
        pose = Pose2D()
        pose.x = sum(e[0] for e in estimates) / len(estimates)
        pose.y = sum(e[1] for e in estimates) / len(estimates)
        pose.theta = math.atan2(sum(math.sin(e[2]) for e in estimates),
                                sum(math.cos(e[2]) for e in estimates))
        self.pose_pub.publish(pose)

    def pose_from_tag(self, tag_id):
        # Where is the tag relative to the camera?
        try:
            t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{tag_id}', Time())
        except Exception:
            return None
        p = t.transform.translation
        q = t.transform.rotation

        # Camera frame: z = forward, x = right, y = down.
        # Convert to the robot's 2D frame: forward and left, from the robot center.
        forward = p.z + CAMERA_OFFSET
        left = -p.x

        # Which way does the tag face, seen from the robot?
        # The tag's z axis points out of its printed side. Take that axis
        # (third column of the rotation matrix) and keep its forward/left parts.
        face_forward = 1 - 2 * (q.x * q.x + q.y * q.y)   # camera z part
        face_left = -2 * (q.x * q.z + q.w * q.y)         # minus camera x part
        tag_angle_seen = math.atan2(face_left, face_forward)

        # Where is the tag in the world? (from tag_map.yaml)
        tag = self.tag_map[tag_id]

        # Work backwards:
        # tag world yaw = robot yaw + tag angle seen  ->  solve for robot yaw
        theta = tag['yaw'] - tag_angle_seen
        theta = math.atan2(math.sin(theta), math.cos(theta))  # wrap to [-pi, pi]

        # tag world position = robot position + (forward, left) rotated by robot yaw
        # -> solve for robot position
        x = tag['x'] - (forward * math.cos(theta) - left * math.sin(theta))
        y = tag['y'] - (forward * math.sin(theta) + left * math.cos(theta))
        return x, y, theta


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
