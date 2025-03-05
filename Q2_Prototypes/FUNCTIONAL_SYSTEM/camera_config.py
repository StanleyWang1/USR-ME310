import numpy as np

# (Average) camera intrinsic for both Oak-D S-2 cameras (at 360p res)
scale_factor = 6
K = np.array([
    [3077.994/scale_factor, 0.0, 1932.107/scale_factor],
    [0.0, 3076.787/scale_factor, 1088.077/scale_factor],
    [0.0, 0.0, 1.0]
])
K_inv = np.linalg.inv(K)

# Camera extrinsic poses
h = 0.750  # [m]
w = 0.750  # [m]
theta = np.deg2rad(60)  # [deg]

R_1 = np.array([
    [0, -np.sin(theta), np.cos(theta)],
    [-1, 0, 0],
    [0, -np.cos(theta), -np.sin(theta)]
])
t_1 = np.array([[-w / 2], [0], [h]])
W_T_C1 = np.block([[R_1, t_1], [np.zeros((1, 3)), 1]])

R_2 = np.array([
    [0, np.sin(theta), -np.cos(theta)],
    [1, 0, 0],
    [0, -np.cos(theta), -np.sin(theta)]
])
t_2 = np.array([[w / 2], [0], [h]])
W_T_C2 = np.block([[R_2, t_2], [np.zeros((1, 3)), 1]])

# Camera projection matrices - terms for least squares
M = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])
P1 = K @ M @ np.linalg.inv(W_T_C1)
P2 = K @ M @ np.linalg.inv(W_T_C2)

# b = t_2 - t_1