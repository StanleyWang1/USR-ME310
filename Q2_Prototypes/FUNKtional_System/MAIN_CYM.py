import cv2
# import csv
from collections import deque
import depthai as dai
import numpy as np
import serial
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from utils import smooth_keypoints
from camera_config import P1, P2
from esp_link import espStatus
from robot_sim import initialize_simulation, command_sphere_position, close_simulation
from head_tracking import head_tracker

# Shared frame storage for both cameras
frames = [None, None]
fps_values = [0.0, 0.0]

# Global variables to store the most recent blob coordinates
c1_cyan = np.zeros((2,1))
c2_cyan = np.zeros((2,1))
c1_yellow = np.zeros((2,1))
c2_yellow = np.zeros((2,1))
c1_magenta = np.zeros((2,1))
c2_magenta = np.zeros((2,1))

# Global variable for the latest 3D points
current_3d_points = np.zeros((3, 3))
head_pitch = 0 # pose of tracked head
head_yaw = 0 # pose of tracked head

# Global variable for input devices
pedal_engaged = None
touch_engaged = None
pot_value = None

lock = threading.Lock()

# Camera processing function
def process_camera(device_info, index):
    global c1_cyan, c2_cyan, c1_yellow, c2_yellow, c1_magenta, c2_magenta

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Camera properties
    camRgb.setPreviewSize(640, 360)
    camRgb.setFps(60)
    camRgb.initialControl.setManualExposure(1000, 100)
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
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            keypoints = blob_detector.detect(frame)
            hue_vals = np.zeros(3)

            if keypoints:
                coordinates = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints[:3]]
                for i, (x, y) in enumerate(coordinates):
                    # Calculate blob hue
                    size = int(keypoints[i].size / 2)
                    roi = frame_hsv[max(0, y - size):min(frame_hsv.shape[0], y + size), max(0, x - size):min(frame_hsv.shape[1], x + size)]
                    hue = cv2.mean(roi)[0] if roi.size > 0 else 0
                    hue_vals[i] = hue
                    # Update corresponding pixel coordinate
                    if hue < 50: # blue
                        if index == 1:
                            with lock:
                               c1_cyan = np.array([[x], [y]]);
                        else:
                            with lock:
                               c2_cyan = np.array([[x], [y]]);
                    elif hue < 100:
                        if index == 1:
                            with lock:
                                c1_yellow = np.array([[x], [y]]);
                        else:
                            with lock:
                                c2_yellow = np.array([[x], [y]]);
                    else:
                        if index == 1:
                            with lock:
                                c1_magenta = np.array([[x], [y]]);
                        else:
                            with lock:
                                c2_magenta = np.array([[x], [y]]);

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
                    color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(hue_vals[i]), 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0])
                    cv2.circle(annotated_frame, (x, y), 5, color_bgr, -1)
                    # cv2.putText(annotated_frame, f"({x}, {y}) @ H{hue_vals[i]}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
                    cv2.putText(annotated_frame, f"H = {int(hue_vals[i])}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

            # Store the annotated frame for display
            frames[index] = annotated_frame

def triangulate():
    global P1, P2, c1_cyan, c2_cyan, c1_yellow, c2_yellow, c1_magenta, c2_magenta, current_3d_points

    while True:
        with lock:
            # Process each 3D point (cyan, yellow, magenta) separately
            points_2d_camera1 = [c1_cyan, c1_yellow, c1_magenta]
            points_2d_camera2 = [c2_cyan, c2_yellow, c2_magenta]
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
                current_3d_points[i] = 0.5*current_3d_points[i] + 0.5*X_world.reshape(1, 3)
                # current_3d_points[i] = X_world.reshape(1,3)

# Function to visualize the 3D point with fixed axes
def visualize_3d():
    global head_pitch
    initialize_simulation(head_pitch, head_yaw)
    while True:
        with lock: 
            if pedal_engaged == 1:
                command_sphere_position(current_3d_points, head_pitch, head_yaw)
        time.sleep(0.01)

# Head tracking
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

# Serial link to esp
def serial_in():
    global pedal_engaged, touch_engaged, pot_value

    ser = serial.Serial('COM9',9600, timeout = 0.25) # for windows comx is the port but on linux this is /dev/ttyusbx 
    time.sleep(2) #wait for serial port to open and connection to be established 

    while True:
        if ser.in_waiting > 0:
            deviceList = espStatus(ser)
            pedal1 = deviceList["Pedal 1"] 
            # pedal2 = deviceList["Pedal y"]
            touch = deviceList["touch"]
            potentiometer = deviceList["pot"]
        with lock:
            pedal_engaged = int(pedal1[0])
            touch_engaged = int(touch[0])
            pot_value = int(potentiometer[0])
        # print(pedal_engaged)
        time.sleep(0.01)

def update_gui():
    global pedal_engaged, touch_engaged
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Pedal rectangle and label
    pedal_rect = Rectangle((1, 6), 8, 3, color='red', ec='black', lw=2)
    ax.add_patch(pedal_rect)
    ax.text(5, 7.5, "PEDAL", color='white', ha='center', va='center', fontsize=30, weight='bold')

    # Touch rectangle and label
    touch_rect = Rectangle((1, 1), 8, 3, color='red', ec='black', lw=2)
    ax.add_patch(touch_rect)
    ax.text(5, 2.5, "TOUCH", color='white', ha='center', va='center', fontsize=30, weight='bold')

    plt.ion()
    plt.show()

    while True:
        with lock:
            pedal_rect.set_color('green' if pedal_engaged == 1 else 'red')
            touch_rect.set_color('green' if touch_engaged == 1 else 'red')
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

# Get connected devices
available_devices = dai.Device.getAllAvailableDevices()
if len(available_devices) < 2:
    print("Error: Less than two OAK-D cameras connected.")
else:
    threads = []

    # Start camera threads
    for i, device_info in enumerate(available_devices[:2]):
        thread = threading.Thread(target=process_camera, args=(device_info, i))
        threads.append(thread)
        thread.start()

    # Start triangulation and visualization threads
    triangulation_thread = threading.Thread(target=triangulate)
    head_tracking_thread = threading.Thread(target=head_track)
    visualization_thread = threading.Thread(target=visualize_3d)
    serial_thread = threading.Thread(target = serial_in)
    GUI_thread = threading.Thread(target = update_gui)

    triangulation_thread.start()
    head_tracking_thread.start()
    visualization_thread.start()
    serial_thread.start()
    GUI_thread.start()

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
