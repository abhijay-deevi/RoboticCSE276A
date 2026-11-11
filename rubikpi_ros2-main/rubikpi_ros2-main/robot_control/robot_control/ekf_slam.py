# keeps track of where robot is, where each tag is, and uncertainty
import math
import numpy as np
def wrap(angle): # same as bfore
    return math.atan2(math.sin(angle), math.cos(angle))
def rotation(theta): # same as before
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

class EkfSlam:
    def __init__(self, speed_noise, turn_noise, meas_noise, meas_noise_per_m, outlier_gate):
        self.x = np.zeros(3) # robot loc
        self.P = np.zeros((3, 3)) # uncertainty
        self.ids = []  # tag ID stored in 

        self.speed_noise = speed_noise            # std of speed error
        self.turn_noise = turn_noise              # std of turn-rate error
        self.meas_noise = meas_noise              # std of a tag measurement at 0 m
        self.meas_noise_per_m = meas_noise_per_m  # extra std per meter of distance to the tag
        self.outlier_gate = outlier_gate          # reject measurements this unlikely (Mahalanobis^2)

    def predict(self, v, w, dt): # basically same as hw2
        theta = self.x[2]
        c, s = math.cos(theta), math.sin(theta)
        self.x[0] += v * c * dt
        self.x[1] += v * s * dt
        self.x[2] = wrap(theta + w * dt)

        # new state depends on old state
        n = len(self.x)
        F = np.eye(n)
        F[0, 2] = -v * s * dt
        F[1, 2] = v * c * dt

        # this basically maps errors in the commands in x,y,theta. You have the speed error for x and y, turn error for theta
        G = np.zeros((n, 2))
        G[0, 0] = c * dt
        G[1, 0] = s * dt
        G[2, 1] = dt

        # this is specifically for the wheels since we were having problems with it 
        Q = np.diag([(self.speed_noise * v) ** 2 + 1e-6, (self.turn_noise * w) ** 2 + 1e-6])

        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def measurement_noise(self, forward, left): # adds a little bit of distance to completely trust where the tag is
        std = self.meas_noise + self.meas_noise_per_m * math.hypot(forward, left)
        return np.eye(2) * std ** 2

    def update(self, tag_id, forward, left): # this helps us update tags we've seen 
        if tag_id not in self.ids: # if the tag isn't added yet, then add it
            self.add_landmark(tag_id, forward, left)
            return 'new'

        #finds the tags x and y in the list
        i = 3 + 2 * self.ids.index(tag_id)
        rx, ry, theta = self.x[0], self.x[1], self.x[2]
        dx, dy = self.x[i] - rx, self.x[i + 1] - ry
        c, s = math.cos(theta), math.sin(theta)

        # finds the landmark position based on the robots position (where it should appear from the robot)
        z_pred = np.array([c * dx + s * dy, -s * dx + c * dy])

        # Basically tells us how z prediction would change if each number changed slightly 
        H = np.zeros((2, len(self.x)))
        H[:, 0:2] = [[-c, -s], [s, -c]] # if robot moves forward, then the tag is closer
        H[:, 2] = [-s * dx + c * dy, -c * dx - s * dy] # if the robot turns, the tag shifts sideways
        H[:, i:i + 2] = [[c, s], [-s, c]] # if the tag moves then it shows up as moved and then rotates to what the robot wants

        R = self.measurement_noise(forward, left)
        innovation = np.array([forward, left]) - z_pred # what the camera says vs what we predicted
        S = H @ self.P @ H.T + R # basically tells us max uncertainty

        # outlier checker
        if innovation @ np.linalg.solve(S, innovation) > self.outlier_gate:
            return 'rejected'

        # finalizes the correction and chooses between camera or robots estimate
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2  # keep it symmetric (numerical safety)
        return 'updated'

    def add_landmark(self, tag_id, forward, left): # this is for adding a full new tag
        theta = self.x[2]
        c, s = math.cos(theta), math.sin(theta)
        z = np.array([forward, left])
        landmark = self.x[0:2] + rotation(theta) @ z

        # getting the unceertrainty  + camera noise\
        Gr = np.array([[1, 0, -s * forward - c * left],
                       [0, 1, c * forward - s * left]])  # d(landmark) / d(robot pose)
        Gz = rotation(theta)                               # d(landmark) / d(measurement)
        P_rr = self.P[0:3, 0:3]
        P_ll = Gr @ P_rr @ Gr.T + Gz @ self.measurement_noise(forward, left) @ Gz.T
        P_lx = Gr @ self.P[0:3, :]  # correlation with everything already in the state

        # adding the landmarks 
        n = len(self.x)
        P_new = np.zeros((n + 2, n + 2))
        P_new[:n, :n] = self.P
        P_new[n:, :n] = P_lx
        P_new[:n, n:] = P_lx.T
        P_new[n:, n:] = P_ll
        self.P = P_new
        self.x = np.concatenate([self.x, landmark])
        self.ids.append(tag_id)

    def landmarks(self): # this just lists them
        out = []
        for k, tag_id in enumerate(self.ids):
            i = 3 + 2 * k
            out.append((tag_id, self.x[i], self.x[i + 1], self.P[i:i + 2, i:i + 2].copy()))
        return out
