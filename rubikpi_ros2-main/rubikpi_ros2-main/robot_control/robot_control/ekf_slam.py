"""EKF SLAM math (no ROS): one Kalman filter for the robot pose AND the landmark map.

State x = [robot_x, robot_y, robot_theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
  - starts with only the robot (3 numbers); grows by 2 each time a new tag is seen
Measurement z = [forward, left]: where a tag is, as seen from the robot
"""

import math

import numpy as np


def wrap(angle):
    """Keep an angle between -pi and pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(theta):
    """2D rotation matrix: turns robot-frame (forward, left) into world-frame (x, y)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


class EkfSlam:
    def __init__(self, speed_noise, turn_noise, meas_noise, meas_noise_per_m, outlier_gate):
        # The start pose IS the origin of the map, so it has zero uncertainty
        self.x = np.zeros(3)
        self.P = np.zeros((3, 3))
        self.ids = []  # tag ID stored in each landmark slot, in the order they were found

        self.speed_noise = speed_noise            # std of speed error, as a fraction of speed
        self.turn_noise = turn_noise              # std of turn-rate error, as a fraction of turn rate
        self.meas_noise = meas_noise              # std of a tag measurement at 0 m (meters)
        self.meas_noise_per_m = meas_noise_per_m  # extra std per meter of distance to the tag
        self.outlier_gate = outlier_gate          # reject measurements this unlikely (Mahalanobis^2)

    # ---------------- Predict: move the robot using what we commanded ----------------

    def predict(self, v, w, dt):
        theta = self.x[2]
        c, s = math.cos(theta), math.sin(theta)

        # Motion model (same as HW2 dead reckoning). Landmarks don't move.
        self.x[0] += v * c * dt
        self.x[1] += v * s * dt
        self.x[2] = wrap(theta + w * dt)

        # F = how the new state depends on the old state (Jacobian of the motion model)
        n = len(self.x)
        F = np.eye(n)
        F[0, 2] = -v * s * dt
        F[1, 2] = v * c * dt

        # G = how the new state depends on the controls (v, w)
        G = np.zeros((n, 2))
        G[0, 0] = c * dt
        G[1, 0] = s * dt
        G[2, 1] = dt

        # Control noise: wheels don't do exactly what we command. Bigger when moving faster.
        Q = np.diag([(self.speed_noise * v) ** 2 + 1e-6, (self.turn_noise * w) ** 2 + 1e-6])

        self.P = F @ self.P @ F.T + G @ Q @ G.T

    # ---------------- Update: correct robot + map with a tag measurement ----------------

    def measurement_noise(self, forward, left):
        std = self.meas_noise + self.meas_noise_per_m * math.hypot(forward, left)
        return np.eye(2) * std ** 2

    def update(self, tag_id, forward, left):
        """Use one tag measurement. Returns 'new', 'updated' or 'rejected'."""
        if tag_id not in self.ids:
            self.add_landmark(tag_id, forward, left)
            return 'new'

        i = 3 + 2 * self.ids.index(tag_id)
        rx, ry, theta = self.x[0], self.x[1], self.x[2]
        dx, dy = self.x[i] - rx, self.x[i + 1] - ry
        c, s = math.cos(theta), math.sin(theta)

        # Predicted measurement h(x): the landmark rotated into the robot's frame
        z_pred = np.array([c * dx + s * dy, -s * dx + c * dy])

        # H = how the measurement depends on the state (Jacobian of h).
        # Only the robot pose and THIS landmark matter; everything else is 0.
        H = np.zeros((2, len(self.x)))
        H[:, 0:2] = [[-c, -s], [s, -c]]
        H[:, 2] = [-s * dx + c * dy, -c * dx - s * dy]
        H[:, i:i + 2] = [[c, s], [-s, c]]

        R = self.measurement_noise(forward, left)
        innovation = np.array([forward, left]) - z_pred
        S = H @ self.P @ H.T + R

        # Throw out readings that disagree wildly with what we expect (bad detections)
        if innovation @ np.linalg.solve(S, innovation) > self.outlier_gate:
            return 'rejected'

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2  # keep it symmetric (numerical safety)
        return 'updated'

    def add_landmark(self, tag_id, forward, left):
        """First time seeing this tag: put it where the camera says, relative to the robot."""
        theta = self.x[2]
        c, s = math.cos(theta), math.sin(theta)
        z = np.array([forward, left])
        landmark = self.x[0:2] + rotation(theta) @ z

        # Its uncertainty comes from the robot's uncertainty plus the measurement noise
        Gr = np.array([[1, 0, -s * forward - c * left],
                       [0, 1, c * forward - s * left]])  # d(landmark) / d(robot pose)
        Gz = rotation(theta)                               # d(landmark) / d(measurement)
        P_rr = self.P[0:3, 0:3]
        P_ll = Gr @ P_rr @ Gr.T + Gz @ self.measurement_noise(forward, left) @ Gz.T
        P_lx = Gr @ self.P[0:3, :]  # correlation with everything already in the state

        n = len(self.x)
        P_new = np.zeros((n + 2, n + 2))
        P_new[:n, :n] = self.P
        P_new[n:, :n] = P_lx
        P_new[:n, n:] = P_lx.T
        P_new[n:, n:] = P_ll
        self.P = P_new
        self.x = np.concatenate([self.x, landmark])
        self.ids.append(tag_id)

    # ---------------- Read out the results ----------------

    def landmarks(self):
        """List of (tag_id, x, y, covariance 2x2)."""
        out = []
        for k, tag_id in enumerate(self.ids):
            i = 3 + 2 * k
            out.append((tag_id, self.x[i], self.x[i + 1], self.P[i:i + 2, i:i + 2].copy()))
        return out
