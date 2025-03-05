import mujoco
import mujoco.viewer
import time

# Load MuJoCo XML model (replace with your actual path)
model = mujoco.MjModel.from_xml_path("Q2_Prototypes/FUNCTIONAL_SYSTEM/ViperX_300S/scene.xml")
data = mujoco.MjData(model)

# Open the MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)  # Step the simulation
        time.sleep(0.01)
        viewer.sync()  # Refresh the viewer
