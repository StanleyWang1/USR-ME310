import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# Load the ViperX 300S model and create the data object
model = mujoco.MjModel.from_xml_path("Q2_Prototypes/FUNCTIONAL_SYSTEM/ViperX_300S/scene.xml")
data = mujoco.MjData(model)

# End-effector site name (make sure this matches your XML model)
EE_SITE_NAME = "pinch"
ee_site_id = model.site(EE_SITE_NAME).id

# Define joint limits for the first 6 joints based on the XML constraints
joint_limits = np.array([
    [-3.14158, 3.14158],  # Waist
    [-1.85005, 1.25664],  # Shoulder
    [-1.76278, 1.6057],   # Elbow
    [-3.14158, 3.14158],  # Forearm Roll
    [-1.8675, 2.23402],   # Wrist Angle
    [-3.14158, 3.14158]   # Wrist Rotate
])

def forward_kinematics(qpos):
    """
    Compute the end-effector position and orientation (as a quaternion)
    given the first 6 joint angles.
    """
    data.qpos[:6] = qpos
    mujoco.mj_forward(model, data)
    
    # Get position from the site
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
    # Compute the relative rotation: r_rel = r2 * inverse(r1)
    r_rel = r2 * r1.inv()
    # The magnitude of the rotation vector is the angle (in radians)
    return np.linalg.norm(r_rel.as_rotvec())

def inverse_kinematics(target_pos, target_quat, q_init):
    """Solve for joint angles that bring the end-effector to the target position & orientation."""
    def cost_function(q):
        ee_pos, ee_quat = forward_kinematics(q)
        pos_error = np.linalg.norm(ee_pos - target_pos)
        quat_error = quaternion_distance(ee_quat, target_quat)
        return pos_error + 0.5 * quat_error  # Adjust weight as needed

    result = minimize(cost_function, q_init, bounds=joint_limits, method="L-BFGS-B")
    if result.success:
        print(f"IK Solved: {result.x}")
        return result.x
    else:
        print("IK Optimization Failed!")
        return None

def main():
    # Define the target end-effector pose (position and orientation)
    target_position = np.array([0.3, 0.0, 0.2])
    # Example: 45° rotation about Y-axis
    target_orientation = R.from_euler('xyz', [0, 0.001, 0], degrees=True).as_quat()
    
    # Use current joint positions as an initial guess
    q_init = data.qpos[:6].copy()
    
    # Solve IK for the desired pose
    q_solution = inverse_kinematics(target_position, target_orientation, q_init)
    
    # Apply the solution using control inputs (for smooth actuation)
    if q_solution is not None:
        print("Applying IK solution...")
        data.ctrl[:6] = q_solution
    else:
        print("Using initial joint positions...")
        data.ctrl[:6] = q_init

    # Launch the MuJoCo viewer to visualize the result
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(5):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
