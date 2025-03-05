import mujoco
import time
import numpy as np
from mujoco.viewer import launch_passive
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# Global viewer instance, model, data, and previous control command
viewer = None
model = None
data = None
previous_control = None

# Define joint limits for the first 6 joints (from XML constraints)
joint_limits = np.array([
    [-3.14158, 3.14158],  # Waist
    [-1.85005, 1.25664],  # Shoulder
    [-1.76278, 1.6057],   # Elbow
    [-3.14158, 3.14158],  # Forearm Roll
    [-1.8675, 2.23402],   # Wrist Angle
    [-3.14158, 3.14158]   # Wrist Rotate
])

# Offsets if needed (e.g., to shift the gripper)
y_offset = -0.05
z_offset = -0.1  # in meters

def initialize_simulation(head_pitch, head_yaw):
    """
    Initializes the MuJoCo model, data, and viewer. Called only once.
    """
    global viewer, model, data, previous_control

    xml_path = "./Q2_Prototypes/FUNCTIONAL_SYSTEM/ViperX_300S/scene.xml"  # Update path as needed
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    viewer = launch_passive(model, data)
    viewer.cam.azimuth = 270 + head_yaw   # Rotation around Z-axis (degrees)
    viewer.cam.elevation = head_pitch      # Up/Down tilt angle
    viewer.cam.distance = 1.0              # Distance from the model

    previous_control = data.ctrl[:6].copy()

def forward_kinematics(qpos):
    """
    Compute the end-effector position and orientation (as a quaternion)
    given the first 6 joint angles.
    """
    data.qpos[:6] = qpos
    mujoco.mj_forward(model, data)

    # Extract position from the "pinch" site
    ee_site_id = model.site("pinch").id
    ee_pos = np.copy(data.site_xpos[ee_site_id])
    
    # Extract the 3x3 rotation matrix and convert it to a quaternion
    ee_rot_matrix = np.copy(data.site_xmat[ee_site_id].reshape(3, 3))
    ee_quat = R.from_matrix(ee_rot_matrix).as_quat()

    return ee_pos, ee_quat

def quaternion_distance(q1, q2):
    """
    Computes the difference between two quaternions as the norm of the relative rotation vector.
    This method is generally more stable near the identity.
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_rel = r2 * r1.inv()
    return np.linalg.norm(r_rel.as_rotvec())

def inverse_kinematics(target_pos, target_quat, q_init):
    """
    Solve for joint angles that bring the end-effector to the target position & orientation.
    Uses solver options to limit iterations and loosen tolerances for faster convergence.
    """
    def cost_function(q):
        ee_pos, ee_quat = forward_kinematics(q)
        pos_error = np.linalg.norm(ee_pos - target_pos)
        quat_error = quaternion_distance(ee_quat, target_quat)
        return pos_error + 0.5 * quat_error  # Adjust weight as needed

    # Solver options: lower iteration count and tolerance for faster (but less precise) solutions.
    options = {'maxiter': 50, 'ftol': 1e-3}
    result = minimize(cost_function, q_init, bounds=joint_limits, method="L-BFGS-B", options=options)
    
    if result.success:
        print("IK Solved:", result.x)
        return result.x
    else:
        print("IK Optimization Failed!")
        return None

def command_robot_pose(target_position, target_euler, head_pitch, head_yaw):
    """
    Takes in a target position (x, y, z) and Euler angles (Rx, Ry, Rz),
    performs IK, and commands the robot accordingly.
    If no valid IK solution is found, the robot keeps its previous pose.
    """
    global previous_control

    if viewer is None:
        raise RuntimeError("Simulation not initialized. Call initialize_simulation() first.")

    # Convert target Euler angles (degrees) to a quaternion.
    target_quat = R.from_euler('xyz', target_euler, degrees=True).as_quat()

    # Use the current joint positions as an initial guess.
    q_init = data.qpos[:6].copy()
    q_solution = inverse_kinematics(target_position, target_quat, q_init)

    if q_solution is not None:
        data.ctrl[:6] = q_solution
        previous_control = q_solution.copy()
    else:
        data.ctrl[:6] = previous_control

    # Optionally add offsets if needed (example shown for y and z)
    data.ctrl[1] += y_offset
    data.ctrl[2] += z_offset

    # Step the simulation and update the viewer camera
    mujoco.mj_step(model, data)
    viewer.cam.azimuth = 270 + head_yaw
    viewer.cam.elevation = head_pitch
    viewer.cam.distance = 1.0
    viewer.sync()

def close_simulation():
    """
    Closes the viewer and cleans up the simulation.
    """
    global viewer
    if viewer is not None:
        viewer.close()
        viewer = None

def main():
    # Initialize simulation with desired camera parameters
    initialize_simulation(head_pitch=20, head_yaw=0)

    # Example control loop: continuously update target pose
    try:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            # Simulate oscillatory target position and orientation:
            target_position = np.array([0.3 + 0.1*np.sin(t), 0.0, 0.2])
            target_euler = [0, 45*np.sin(t), 0]  # Euler angles in degrees

            command_robot_pose(target_position, target_euler, head_pitch=20, head_yaw=0)
            time.sleep(0.01)
    except KeyboardInterrupt:
        close_simulation()

if __name__ == "__main__":
    main()
