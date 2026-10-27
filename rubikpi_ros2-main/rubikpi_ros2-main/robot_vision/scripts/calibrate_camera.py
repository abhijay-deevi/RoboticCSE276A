#!/usr/bin/env python3
"""
Camera calibration for the Rubik Pi camera (OpenCV checkerboard method).

Step 1 - capture images (camera node must be running with image_rectify:=false):
    python3 calibrate_camera.py capture
  Hold the checkerboard in front of the camera and move it around
  (near/far, left/right, tilted). A picture is saved automatically every
  time the board is found, until 25 pictures are saved.

Step 2 - compute calibration and write the YAML the camera node reads:
    python3 calibrate_camera.py calibrate

Checkerboard size is counted in INNER corners (where black squares touch),
e.g. a board with 10 x 7 squares has 9 x 6 inner corners.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'calib_images')
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, '..', 'config', 'camera_parameter.yaml')


def find_corners(gray, board_size):
    """Return refined checkerboard corners, or None if the board is not found."""
    found, corners = cv2.findChessboardCorners(
        gray, board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def capture(args):
    """Subscribe to the camera topic and save frames where the checkerboard is visible."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    os.makedirs(args.image_dir, exist_ok=True)
    board_size = (args.cols, args.rows)

    class CaptureNode(Node):
        def __init__(self):
            super().__init__('calibration_capture')
            self.count = 0
            self.last_save = 0.0
            self.create_subscription(Image, args.topic, self.image_callback, 10)
            self.get_logger().info(f'Listening on {args.topic}. Show the checkerboard to the camera.')

        def image_callback(self, msg):
            if self.count >= args.num_images or time.time() - self.last_save < args.interval:
                return
            # Convert ROS Image (rgb8) to an OpenCV image without cv_bridge
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            frame = frame[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if find_corners(gray, board_size) is None:
                return
            self.count += 1
            self.last_save = time.time()
            path = os.path.join(args.image_dir, f'calib_{self.count:02d}.png')
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f'Saved {self.count}/{args.num_images}: {path}  -> move the board')

    rclpy.init()
    node = CaptureNode()
    try:
        while rclpy.ok() and node.count < args.num_images:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.get_logger().info('Done capturing. Now run: python3 calibrate_camera.py calibrate')
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


def calibrate(args):
    """Run OpenCV calibration on saved images and write the camera YAML."""
    board_size = (args.cols, args.rows)

    # 3D positions of the checkerboard corners on the flat board (z = 0), in meters
    board_points = np.zeros((args.rows * args.cols, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    object_points, image_points = [], []
    image_size = None
    for path in sorted(glob.glob(os.path.join(args.image_dir, '*.png'))):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_size = (gray.shape[1], gray.shape[0])
        corners = find_corners(gray, board_size)
        if corners is None:
            print(f'  board not found, skipping: {path}')
            continue
        object_points.append(board_points)
        image_points.append(corners)

    if len(image_points) < 10:
        print(f'Only {len(image_points)} usable images; capture at least 10 (ideally 20+).')
        return

    rms, K, D, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    print(f'Used {len(image_points)} images. Reprojection error: {rms:.3f} px (under ~0.5 is good)')

    # The camera node undistorts with exactly this matrix when image_rectify:=true,
    # and the AprilTag node reads it from P, so P must be this "new" camera matrix.
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, image_size, 0.0, image_size)
    P = np.hstack([new_K, np.zeros((3, 1))])

    fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', image_size[0])
    fs.write('image_height', image_size[1])
    fs.write('camera_name', 'camera')
    fs.write('camera_matrix', K)
    fs.write('distortion_model', 'plumb_bob')
    fs.write('distortion_coefficients', D.reshape(1, -1))
    fs.write('rectification_matrix', np.eye(3))
    fs.write('projection_matrix', P)
    fs.release()

    print('Camera matrix K:\n', np.round(K, 2))
    print('Distortion D:', np.round(D.ravel(), 4))
    print(f'Wrote {os.path.abspath(args.output)}')


def main():
    parser = argparse.ArgumentParser(description='Checkerboard camera calibration')
    parser.add_argument('mode', choices=['capture', 'calibrate'])
    parser.add_argument('--cols', type=int, default=9, help='inner corners across')
    parser.add_argument('--rows', type=int, default=6, help='inner corners down')
    parser.add_argument('--square', type=float, default=0.025, help='square size in meters')
    parser.add_argument('--image-dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--topic', default='/camera/image_raw')
    parser.add_argument('--num-images', type=int, default=25)
    parser.add_argument('--interval', type=float, default=1.5, help='seconds between saves')
    args = parser.parse_args()

    capture(args) if args.mode == 'capture' else calibrate(args)


if __name__ == '__main__':
    main()
