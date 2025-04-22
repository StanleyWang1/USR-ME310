from collections import deque
import serial
import threading
import time

import cv2
import depthai as dai
import keyboard
import numpy as np
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from audio_driver import play_audio
from camera_config import P1, P2
from esp_link import espStatus
from head_tracking import head_tracker
from forcep_sim import initialize_simulation, command_pose

# Shared frame storage for both cameras
frames = [None, None]
fps_values = [0.0, 0.0]

# Global variables to store the most recent blob coordinates
cam1_point1 = np.zeros((2,1))
cam2_point1 = np.zeros((2,1))
cam1_point2 = np.zeros((2,1))
cam2_point2 = np.zeros((2,1))
cam1_point3 = np.zeros((2,1))
cam2_point3 = np.zeros((2,1))

# Global variable for the latest 3D points
current_3d_points = np.zeros((3, 3))

# Mapped 3D gripper control coordinates
gripper_position = np.zeros((3,))
gripper_orientation = np.zeros((3,))
gripper_grasp = 0

# Global variables for tracked head pose
head_pitch = 0
head_yaw = 0 

# Global variable for surgeon identity
identity_dict = {
    "0013885379" : {"name" : "Dr. Stanley Wang", "color" : "darkgoldenrod", "audio" : "./Q2_Prototypes/FUNCTIONAL_SYSTEM/audio_files/Stanley.mp3"},
    "0013912256" : {"name" : "Dr. Nate Lim", "color" : "mediumslateblue", "audio" : "./Q2_Prototypes/FUNCTIONAL_SYSTEM/audio_files/Nate.mp3"},
    "0013906125" : {"name" : "Dr. Jesus Tejeda", "color" : "darkturquoise", "audio" : "./Q2_Prototypes/FUNCTIONAL_SYSTEM/audio_files/Jesus.mp3"}
}
current_user = ""

# Global variable for input devices
button_engaged = None
touch_engaged = None
relay_engaged = False

lock = threading.Lock()

## ----------------------------------------------------------------------------------------------------
# Luxonis Oak-D S2 Camera Processing
## ----------------------------------------------------------------------------------------------------
def process_camera(device_info, index):
    global cam1_point1, cam2_point1, cam1_point2, cam2_point2, cam1_point3, cam2_point3

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Camera properties
    camRgb.setPreviewSize(640, 360)
    camRgb.setFps(60)
    camRgb.initialControl.setManualExposure(500, 100)
    focus_value = 100
    camRgb.initialControl.setManualFocus(focus_value)
    camRgb.initialControl.setManualWhiteBalance(5500)  # Set to desired color temperature

    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setPreviewKeepAspectRatio(False)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Set up the blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 500
    params.filterByColor = True
    params.blobColor = 255
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    blob_detector = cv2.SimpleBlobDetector_create(params)

    # Connect to device
    with dai.Device(pipeline, device_info) as device:
        print(f'Connected to: {device_info.getMxId()}')

        qRgb = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)

        # FPS counter variables
        frame_count = 0
        start_time = time.time()

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            keypoints = blob_detector.detect(frame)
            hue_vals = np.zeros(3)

            if keypoints:
                coordinates = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints[:3]]
                for i, (x, y) in enumerate(coordinates):
                    # Calculate blob hue
                    # size = int(keypoints[i].size / 2)
                    # roi = frame_hsv[max(0, y - size):min(frame_hsv.shape[0], y + size), max(0, x - size):min(frame_hsv.shape[1], x + size)]
                    # hue = cv2.mean(roi)[0] if roi.size > 0 else 0
                    # hue_vals[i] = hue

                    # Calculate ROI boundaries based on keypoint size
                    size = int(keypoints[i].size / 2)
                    y_start = max(0, y - size)
                    y_end = min(frame.shape[0], y + size)
                    x_start = max(0, x - size)
                    x_end = min(frame.shape[1], x + size)
                    roi = frame[y_start:y_end, x_start:x_end]

                    if roi.size > 0:
                        # Compute the average color in the ROI.
                        # Note: cv2.mean returns (R, G, B, A) if the image has 3 channels.
                        avg_color_rgb = cv2.mean(roi)[:3]
                        # Create a 1x1 image with this average color.
                        avg_color_img = np.uint8([[avg_color_rgb]])
                        # Convert the average color to HSV.
                        # Use COLOR_RGB2HSV if your frame is in RGB.
                        avg_color_hsv = cv2.cvtColor(avg_color_img, cv2.COLOR_RGB2HSV)
                        hue = avg_color_hsv[0, 0, 0]
                        hue_vals[i] = hue

                    # Update corresponding pixel coordinate
                    if hue < 30: # blue
                        if index == 0:
                            with lock:
                               cam1_point1 = np.array([[x], [y]]);
                        else:
                            with lock:
                               cam2_point1 = np.array([[x], [y]]);
                    elif hue < 100:
                        if index == 0:
                            with lock:
                                cam1_point2 = np.array([[x], [y]]);
                        else:
                            with lock:
                                cam2_point2 = np.array([[x], [y]]);
                    else:
                        if index == 0:
                            with lock:
                                cam1_point3 = np.array([[x], [y]]);
                        else:
                            with lock:
                                cam2_point3 = np.array([[x], [y]]);

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps_values[index] = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Annotate frame with FPS and blob coordinates
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"FPS: {fps_values[index]:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw blob annotations
            for i, kp in enumerate(keypoints):
                if i < 3:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(hue_vals[i]), 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
                    cv2.circle(annotated_frame, (x, y), 5, color_bgr, -1)
                    # cv2.putText(annotated_frame, f"({x}, {y}) @ H{hue_vals[i]}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
                    cv2.putText(annotated_frame, f"H = {int(hue_vals[i])}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

            # Store the annotated frame for display
            frames[index] = annotated_frame

## ----------------------------------------------------------------------------------------------------
# Triangulate 3D Coordinates of known markers from 2D pixel coordinates
## ----------------------------------------------------------------------------------------------------
def triangulate():
    global P1, P2, cam1_point1, cam2_point1, cam1_point2, cam2_point2, cam1_point3, cam2_point3, current_3d_points

    while True:
        with lock:
            # Process each 3D point (cyan, yellow, magenta) separately
            points_2d_camera1 = [cam1_point1, cam1_point2, cam1_point3]
            points_2d_camera2 = [cam2_point1, cam2_point2, cam2_point3]
        #     engaged = pedal_engaged
        # if engaged == 1:
        for i in range(3):
            # Step 1: Convert (2, 1) arrays to flat arrays
            c_1_flat = points_2d_camera1[i].flatten()  # Shape becomes (2,)
            c_2_flat = points_2d_camera2[i].flatten()  # Shape becomes (2,)

            # Step 2: Add homogeneous coordinate (1)
            c_1 = np.array([c_1_flat[0], c_1_flat[1], 1.0])
            c_2 = np.array([c_2_flat[0], c_2_flat[1], 1.0])

            # Step 3: Construct the linear system Ax = 0
            A = np.zeros((4, 4))

            # First view constraints (camera 1)
            A[0] = c_1[0] * P1[2] - P1[0]
            A[1] = c_1[1] * P1[2] - P1[1]

            # Second view constraints (camera 2)
            A[2] = c_2[0] * P2[2] - P2[0]
            A[3] = c_2[1] * P2[2] - P2[1]

            # Step 4: Solve the system using SVD
            _, _, V = np.linalg.svd(A)
            X_world_homogeneous = V[-1]  # The solution is the last row of V

            # Step 5: Convert to Cartesian coordinates by normalizing
            X_world = X_world_homogeneous[:3] / X_world_homogeneous[3]

            # Step 6: Store the result in the current 3D points
            with lock:
                current_3d_points[i] = 0.5*current_3d_points[i] + 0.5*X_world.reshape(1, 3) # average with prior point to add slight smoothing
                # current_3d_points[i] = X_world.reshape(1,3)

## ----------------------------------------------------------------------------------------------------
# Triangulate 3D Coordinates of known markers from 2D pixel coordinates
## ----------------------------------------------------------------------------------------------------
def calculate_gripper_pose():
    global current_3d_points, gripper_position, gripper_orientation, gripper_grasp, touch_engaged

    def angle_from_y_axis(gripper_pose):
        x, y = gripper_pose[0], gripper_pose[1]
        angle = np.arctan2(x, y)
        return angle
    
    def angle_from_xy_plane(gripper_pose):
        xy_norm = np.linalg.norm(gripper_pose[:2])
        angle = np.arctan2(gripper_pose[2], xy_norm)
        return angle
    
    while True:
        if touch_engaged:
            with lock:
                left_point   = current_3d_points[0]
                center_point = current_3d_points[1]
                right_point  = current_3d_points[2]

            # Compute gripper pose (bisector of left and right unit vectors)
            gripper_pose = ((left_point - center_point) / np.linalg.norm(left_point - center_point) +
                            (right_point - center_point) / np.linalg.norm(right_point - center_point))
            gripper_pose = gripper_pose / np.linalg.norm(gripper_pose)
            

            left_to_right = (right_point - left_point) / np.linalg.norm(right_point - left_point)

            roll = -angle_from_y_axis(gripper_pose) + np.pi
            yaw = -angle_from_xy_plane(gripper_pose)
            pitch = angle_from_xy_plane(left_to_right)

            # --- Calculate grasp angle ---
            # Angle between left and right vectors (from center)
            left_vec  = left_point - center_point
            right_vec = right_point - center_point
            dot_lr    = np.dot(left_vec, right_vec)
            norm_lr   = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
            full_angle = np.arccos(np.clip(dot_lr / norm_lr, -1.0, 1.0))
            
            with lock:
                # Write computed gripper position, orientation, grasp
                gripper_position = center_point - np.array([0, 0, 0.15])
                gripper_orientation = np.array([roll, yaw, pitch])
                gripper_grasp = 1.5*(full_angle/2 - np.deg2rad(15))
        
        time.sleep(0.01)

## ----------------------------------------------------------------------------------------------------
# Use MuJuCo to visualize the tracked 3d points with head-tracked camera reorientation
## ----------------------------------------------------------------------------------------------------
def visualize_3d():
    global head_pitch, head_yaw, gripper_position, gripper_orientation, gripper_grasp, button_engaged
    initialize_simulation(head_pitch, head_yaw)
    while True:
        # with lock: 
        #     if pedal_engaged == 1:
        #         command_robot_pose(current_3d_points, np.array([0.0, 0.01, 0.0]), head_pitch, head_yaw)
        with lock:
            command_pose(gripper_position, gripper_orientation, gripper_grasp, head_pitch, head_yaw, button_engaged)
        time.sleep(0.005)

## ----------------------------------------------------------------------------------------------------
# Use standard camera feed (webcam) to track pose of user's head
## ----------------------------------------------------------------------------------------------------
def head_track():
    global head_yaw, head_pitch
    yaw_window = deque(maxlen=20)
    pitch_window = deque(maxlen=10)
    for yaw, pitch in head_tracker():
        curr_yaw = 0.75*yaw
        curr_pitch = -1.5*pitch - 30

        yaw_window.append(curr_yaw)
        pitch_window.append(curr_pitch)

        smoothed_yaw = sum(yaw_window) / len(yaw_window)
        smoothed_pitch = sum(pitch_window) / len(pitch_window)
        with lock:
            head_yaw = smoothed_yaw
            head_pitch = smoothed_pitch

## ----------------------------------------------------------------------------------------------------
# USB serial link with microcontroller device
## ----------------------------------------------------------------------------------------------------
def serial_in():
    global button_engaged, touch_engaged, relay_engaged

    ser = serial.Serial('COM5', 9600, timeout=0.25)
    time.sleep(2)  # wait for serial port to open

    while True:
        if ser.in_waiting > 0:
            deviceList = espStatus(ser)
            button = deviceList["button"]
            touch = deviceList["touch"]
        with lock:
            button_engaged = int(button[0])
            touch_engaged = int(touch[0])
            # Send a command message with the "CMD:" marker and newline.
            if relay_engaged:
                # print("engaged")
                ser.write(b"CMD:1\n")
            else:
                ser.write(b"CMD:0\n")
        time.sleep(0.01)

## ----------------------------------------------------------------------------------------------------
# Drive GUI for surgical interface
## ----------------------------------------------------------------------------------------------------
def update_gui():
    """
    Displays a thin horizontal bar with the text "Hello Dr. ____" where the name and bar color
    are specified by the global `identity_dict` using the `current_user` key.
    If the current user is not found, defaults to "Unknown" with a gray background.
    """
    global current_user, identity_dict, lock

    # Create a figure with a thin horizontal bar
    fig, ax = plt.subplots(figsize=(10, 0.5))
    ax.set_position([0, 0, 1, 1])  # Set axes to fill the entire figure
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # manager = plt.get_current_fig_manager()
    # manager.window.overrideredirect(True)

    plt.ion()  # Enable interactive mode
    plt.show()

    while True:
        with lock:
            user = current_user
        
        # Look up user details from identity_dict; default if not found.
        if user in identity_dict:
            name = identity_dict[user]["name"]
            color = identity_dict[user]["color"]
            title_msg = f"Hello {name}"
        else:
            color = "gray"
            title_msg = "System Inactive"

        # Clear the axes and reset limits and appearance.
        ax.cla()
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Draw the horizontal rectangle bar with the background color.
        rect = Rectangle((0, 0), 8, 1, color=color)
        ax.add_patch(rect)
        
        # Add centered text with a greeting.
        ax.text(4, 0.5, title_msg, ha='center', va='center', fontsize=20, color="white")
        
        # Update the figure
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

## ----------------------------------------------------------------------------------------------------
# Drive GUI for surgical interface
## ----------------------------------------------------------------------------------------------------
def read_RFID():
    """
    Listens globally for RFID tag input (emulated as keyboard events).
    The RFID reader types the tag and sends 'enter' at the end.
    When 'enter' is detected, the accumulated tag is stored in current_user.
    """
    global current_user, relay_engaged
    tag = ""
    while True:
        event = keyboard.read_event()  # Blocks until an event occurs
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == "enter":
                with lock:
                    current_user = tag
                    tag = ""
                print("RFID Tag Read:", current_user)
                if current_user in identity_dict:
                    with lock:
                        relay_engaged = True
                    play_audio(identity_dict[current_user]["audio"])
            else:
                # Append single-character key names to tag.
                # This assumes the RFID sends characters like '1','2', etc.
                if len(event.name) == 1:
                    tag += event.name

## ----------------------------------------------------------------------------------------------------
# Monitor touch to turn off relay
## ----------------------------------------------------------------------------------------------------
def touch_timeout():
    global touch_engaged, relay_engaged
    last_touch_time = time.time()
    while True:
        # If the touch sensor is active, update the timestamp.
        if touch_engaged:
            last_touch_time = time.time()
        else:
            # If touch_engaged remains False for 5 seconds, disable the relay.
            if time.time() - last_touch_time >= 5:
                relay_engaged = False
        time.sleep(0.1)  # Check every 100 ms

# Get connected devices
# available_devices = dai.Device.getAllAvailableDevices()
available_devices = sorted(dai.Device.getAllAvailableDevices(), key=lambda d: d.getMxId()) # Sort by ID for consistent order

if len(available_devices) < 2:
    print("Error: Less than two OAK-D cameras connected.")
else:
    threads = []

    # Start camera threads
    for i, device_info in enumerate(available_devices[:2]):
        if device_info.getMxId() == "19443010714E1C1300": # camera 1
            thread = threading.Thread(target=process_camera, args=(device_info, 0))
        elif device_info.getMxId() == "19443010719F181300": # camera 2
            thread = threading.Thread(target=process_camera, args=(device_info, 1))
        threads.append(thread)
        thread.start()

    # Start all other threads
    triangulation_thread = threading.Thread(target=triangulate)
    gripper_pose_thread = threading.Thread(target=calculate_gripper_pose)
    head_tracking_thread = threading.Thread(target=head_track)
    visualization_thread = threading.Thread(target=visualize_3d)
    serial_thread = threading.Thread(target = serial_in)
    GUI_thread = threading.Thread(target = update_gui)
    RFID_thread = threading.Thread(target=read_RFID)
    monitor_thread = threading.Thread(target=touch_timeout)

    gripper_pose_thread.start()
    triangulation_thread.start()
    head_tracking_thread.start()
    visualization_thread.start()
    serial_thread.start()
    GUI_thread.start()
    RFID_thread.start()
    monitor_thread.start()

    # Display loop
    while True:
        if frames[0] is not None and frames[1] is not None:
            combined_display = np.hstack((frames[0], frames[1]))
            cv2.imshow("Dual Camera View", combined_display)

        if cv2.waitKey(1) == ord('q'):
            break

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
