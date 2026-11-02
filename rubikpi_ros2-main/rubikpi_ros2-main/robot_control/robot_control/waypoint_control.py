import math
import os
import time

import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory

# these are the same things from motor control 
from robot_control.motor_control import (DRIVE_COMMAND, DRIVE_SPEED, TURN_COMMAND,
                                         LEFT_TURN_RATE, RIGHT_TURN_RATE)
WAYPOINTS = [(1.0, 0.0, 0.0), (1.0, 2.0, math.pi), (0.0, 0.0, 0.0)] # route from hw

POSITION_TOLERANCE = 0.05 # if we get within 5 cm, then we good
ANGLE_TOLERANCE = 0.1 # if we within 6 degrees, then we good
DRIVE_ANGLE_LIMIT = 0.3 # if off by this much, then we have to turn bfore movign
HEADING_GAIN = 1.0 # how much should we steer it when the car is driving
START_DELAY = 3.0 # the amount of time to wait for cam to start
WAYPOINT_PAUSE = 2.0 # pausing at each checkpoint
TURN_RATE = (LEFT_TURN_RATE + RIGHT_TURN_RATE) / 2  # takes from hw 1
DT = 0.05 # loop running time                                      
CAMERA_OFFSET = 0.09 # where the cam is on the robot

# this func basically just keeps the angle between -180 and 180 instead of turning a huge distance
def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WaypointControl(Node):
    def __init__(self):
        super().__init__('waypoint_control')

        # setting up the params
        self.x, self.y, self.theta = 0.0, 0.0, 0.0 # these explain current position and speed
        self.v, self.w = 0.0, 0.0
        self.waypoint_index = 0 # this is the point its goin to
        self.camera_up_time = None
        self.started = False
        self.pause_until = 0.0 
        self.paused_here = False

        # Load the tag map 
        map_file = os.path.join(get_package_share_directory('robot_control'),
                                'config', 'tag_map.yaml')
        with open(map_file) as f:
            self.tag_map = yaml.safe_load(f)['tags']

        # gets cam and tag uses april tag
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        self.start_time = time.time()
        self.log = open(os.path.expanduser('~/trajectory.csv'), 'w')
        self.log.write('time,x,y,theta\n')
        self.waypoint_log = open(os.path.expanduser('~/waypoints.csv'), 'w')
        self.waypoint_log.write('goal_x,goal_y,goal_theta,est_x,est_y,est_theta\n')

        self.timer = self.create_timer(DT, self.loop)
        self.get_logger().info(f'Waypoint control ready, {len(self.tag_map)} tags in map. '
                               'Waiting for camera...')

    # this function basically just tries to find curr location based on the tags
    def on_detections(self, msg):
        if self.camera_up_time is None: # checks the cam is working
            self.camera_up_time = time.time()
        if self.v == 0.0 and self.w != 0.0: # this is a specific case where its spinning 
            return

        estimates = [] # setups of a list of guesses
        # this loop goes through each tag the cam sees and if itsn't in the yaml map, then skip it 
        # also gets curr position based on the tag it sees
        for det in msg.detections:
            if det.id not in self.tag_map:
                continue  # tag not on our map, ignore it
            estimate = self.pose_from_tag(det.id)
            if estimate is not None:
                estimates.append(estimate)

        if not estimates: # this is if it doesn't see any tags so just pass
            return

        # Average the guesses and trust the camera
        # it also keeps within that -180 180 like above
        self.x = sum(e[0] for e in estimates) / len(estimates)
        self.y = sum(e[1] for e in estimates) / len(estimates)
        self.theta = math.atan2(sum(math.sin(e[2]) for e in estimates),
                                sum(math.cos(e[2]) for e in estimates))

    # this method gets the position based on the tag and where it exists in our map
    def pose_from_tag(self, tag_id):
        # this try just gets the tag_id (whats given)
        try:
            t = self.tf_buffer.lookup_transform('camera_frame', f'tag36h11:{tag_id}', Time())
        except Exception:
            return None
        p = t.transform.translation
        q = t.transform.rotation

        # switches to robot pov by using its own axis w those
        forward = p.z + CAMERA_OFFSET
        left = -p.x
        face_forward = 1 - 2 * (q.x * q.x + q.y * q.y)
        # this stuff figures out which direction the tag is facing based on the z axis  and the two things below make up the equation to pull it out + turns it into an angle
        face_left = -2 * (q.x * q.z + q.w * q.y)
        tag_angle_seen = math.atan2(face_left, face_forward)
        tag = self.tag_map[tag_id] # refers back to the map

        # figures out angle between tag and robot, the distance between, and come up with the current location
        theta = wrap(tag['yaw'] - tag_angle_seen)
        x = tag['x'] - (forward * math.cos(theta) - left * math.sin(theta))
        y = tag['y'] - (forward * math.sin(theta) + left * math.cos(theta))
        return x, y, theta

    # this is the acutal main thing that runs it
    def loop(self):
        #  this stuff just keeps the robot stopped until the camera is ready and the wait tiem has been hit
        if not self.started:
            self.drive(0.0, 0.0)
            if self.camera_up_time is None or time.time() - self.camera_up_time < START_DELAY:
                return
            self.started = True
            self.start_time = time.time()
            self.get_logger().info(f'GO! Starting pose x={self.x:.3f} y={self.y:.3f} '
                                   f'theta={self.theta:.3f}')

        # this section is what the dead reckooning is where we just move based on what we were doing previously. 
        self.x += self.v * math.cos(self.theta) * DT
        self.y += self.v * math.sin(self.theta) * DT
        self.theta = wrap(self.theta + self.w * DT)

        # this is more for our report where we can log all the important stuff
        self.log.write(f'{time.time() - self.start_time:.2f},{self.x:.3f},{self.y:.3f},{self.theta:.3f}\n')
        if self.waypoint_index >= len(WAYPOINTS): # these 2 are edge cases
            self.drive(0.0, 0.0)
            return
        if time.time() < self.pause_until:
            self.drive(0.0, 0.0)
            return

        #compare pose to the waypoint
        goal_x, goal_y, goal_theta = WAYPOINTS[self.waypoint_index]
        distance = math.hypot(goal_x - self.x, goal_y - self.y)

        # this checks if its longer than 5 cm, then follow this path
        if distance > POSITION_TOLERANCE:
            self.paused_here = False
            angle_error = wrap(math.atan2(goal_y - self.y, goal_x - self.x) - self.theta)
            if abs(angle_error) > DRIVE_ANGLE_LIMIT:
                self.drive(0.0, math.copysign(TURN_RATE, angle_error))  # turn in place
            else:
                self.drive(DRIVE_SPEED, HEADING_GAIN * angle_error)    # drive and steer
        else: # this checks if its within 5 cm then we just go finish and start going 
            angle_error = wrap(goal_theta - self.theta)
            if abs(angle_error) > ANGLE_TOLERANCE:
                self.paused_here = False
                self.drive(0.0, math.copysign(TURN_RATE, angle_error))
            elif not self.paused_here:
                # Looks finished, check again
                self.paused_here = True
                self.pause_until = time.time() + WAYPOINT_PAUSE
                self.drive(0.0, 0.0)
            else:
                # if its still within tolerance after the pause, then it means this waypoint is done
                self.get_logger().info(
                    f'Reached waypoint {WAYPOINTS[self.waypoint_index]}: estimated pose '
                    f'x={self.x:.3f} y={self.y:.3f} theta={self.theta:.3f}')
                self.waypoint_log.write(f'{goal_x},{goal_y},{goal_theta:.3f},'
                                        f'{self.x:.3f},{self.y:.3f},{self.theta:.3f}\n')
                self.waypoint_log.flush()
                self.waypoint_index += 1
                self.paused_here = False
                self.drive(0.0, 0.0)
                if self.waypoint_index == len(WAYPOINTS):
                    self.get_logger().info('Route finished. Results in ~/waypoints.csv '
                                           'and ~/trajectory.csv')

    def drive(self, v, w): # just drive it
        self.v, self.w = v, w
        forward = v * DRIVE_COMMAND / DRIVE_SPEED
        turn = w * TURN_COMMAND / TURN_RATE
        msg = Float32MultiArray()
        msg.data = [float(forward - turn), float(forward + turn)]  # [left, right]
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.log.close()
    node.waypoint_log.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
