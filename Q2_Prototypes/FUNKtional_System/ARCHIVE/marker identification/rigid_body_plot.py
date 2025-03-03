import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Define the original points
A = np.array([0, 4, 0])
B = np.array([0, 0, 0])
C = np.array([2, 4, 0])

# Define transformation parameters
translation = np.array([0, 0, 2])  # XYZ translation
rotation_angles = np.array([0, 0, 0])  # Rotation in degrees around X, Y, Z

# Convert rotation angles to radians
rotation_radians = np.radians(rotation_angles)

# Create rotation matrix using scipy Rotation
rotation_matrix = R.from_euler('xyz', rotation_radians, degrees=False).as_matrix()

# Function to apply rotation and translation
def transform_point(point, rotation_matrix, translation):
    return rotation_matrix @ point + translation

# Apply transformation to each point
A_transformed = transform_point(A, rotation_matrix, translation)
B_transformed = transform_point(B, rotation_matrix, translation)
C_transformed = transform_point(C, rotation_matrix, translation)

# Print transformed coordinates
print("Transformed Coordinates:")
print(f"A': {A_transformed}")
print(f"B': {B_transformed}")
print(f"C': {C_transformed}")

# Function to set equal scaling for a 3D plot
def set_axes_equal(ax):
    """Set equal scaling for a 3D plot."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2

    x_middle = sum(x_limits) / 2
    y_middle = sum(y_limits) / 2
    z_middle = sum(z_limits) / 2

    ax.set_xlim([x_middle - max_range, x_middle + max_range])
    ax.set_ylim([y_middle - max_range, y_middle + max_range])
    ax.set_zlim([z_middle - max_range, z_middle + max_range])

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original points
ax.scatter(*A, color='r', s=100, label='A (original)')
ax.scatter(*B, color='g', s=100, label='B (original)')
ax.scatter(*C, color='b', s=100, label='C (original)')

# Plot original lines
ax.plot(*zip(*[A, B]), color='black', linestyle='dashed')
ax.plot(*zip(*[B, C]), color='black', linestyle='dashed')

# Plot transformed points
ax.scatter(*A_transformed, color='r', s=100, marker='x', label="A' (transformed)")
ax.scatter(*B_transformed, color='g', s=100, marker='x', label="B' (transformed)")
ax.scatter(*C_transformed, color='b', s=100, marker='x', label="C' (transformed)")

# Plot transformed lines
ax.plot(*zip(*[A_transformed, B_transformed]), color='black')
ax.plot(*zip(*[B_transformed, C_transformed]), color='black')

# Labels and formatting
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Original and Transformed Rigid Body')
ax.legend()

# Enforce equal scale
set_axes_equal(ax)

# Show plot
plt.show()