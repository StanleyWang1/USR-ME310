import cv2
import depthai as dai
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# (Average) camera intrinsic for both Oak-D S-2 cameras (at 360p res)
scale_factor = 6
K = np.array([
    [3077.994/scale_factor, 0.0, 1932.107/scale_factor],
    [0.0, 3076.787/scale_factor, 1088.077/scale_factor],
    [0.0, 0.0, 1.0]
])
K_inv = np.linalg.inv(K)

# Camera extrinsic poses
h = 0.762  # [m]
w = 0.762  # [m]
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

# Shared frame storage for both cameras
frames = [None, None]
fps_values = [0.0, 0.0]

# Global variables to store the most recent blob coordinates
u1, v1 = None, None  # Camera 1 coordinates
u2, v2 = None, None  # Camera 2 coordinates

# Global variable for the latest 3D point
current_3d_point = None

# Camera processing function
def process_camera(device_info, index):
    global u1, v1, u2, v2

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

    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setPreviewKeepAspectRatio(False)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Set up the blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000
    params.filterByColor = True
    params.blobColor = 255

    blob_detector = cv2.SimpleBlobDetector_create(params)

    # Connect to device
    with dai.Device(pipeline, device_info) as device:
        print(f'Connected to: {device_info.getMxId()}')

        qRgb = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()

            # Process the frame to detect blobs
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lower_hsv = np.array([0, 0, 254])
            upper_hsv = np.array([180, 10, 255])
            mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)

            keypoints = blob_detector.detect(mask)

            # Update global coordinates with the first detected blob (if any)
            if keypoints:
                x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
                if index == 0:
                    u1, v1 = x, y
                else:
                    u2, v2 = x, y

            # Store the frame for display
            frames[index] = frame

# Function to perform real-time 3D triangulation
def triangulate():
    global u1, v1, u2, v2, current_3d_point

    while True:
        if u1 is not None and v1 is not None and u2 is not None and v2 is not None:
            # Convert pixel coordinates to normalized image coordinates
            d1 = W_T_C1[:3, :3] @ K_inv @ np.array([[u1], [v1], [1]])
            d2 = W_T_C2[:3, :3] @ K_inv @ np.array([[u2], [v2], [1]])

            # Triangulate
            A = np.hstack((d1, -d2))
            t1 = W_T_C1[:3, 3].reshape(-1, 1)
            t2 = W_T_C2[:3, 3].reshape(-1, 1)
            b = t2 - t1

            lambdas = np.linalg.inv(A.T @ A) @ A.T @ b
            l1, l2 = lambdas[0, 0], lambdas[1, 0]

            # Compute 3D points
            X1 = W_T_C1 @ np.vstack((l1 * K_inv @ np.array([[u1], [v1], [1]]), [1]))
            X2 = W_T_C2 @ np.vstack((l2 * K_inv @ np.array([[u2], [v2], [1]]), [1]))

            # Average the two points
            X_avg = (X1 + X2) / 2

            # Update the current 3D point
            current_3d_point = X_avg[:3].flatten() * 1000  # Convert to millimeters

        # Add a small delay to avoid excessive CPU usage
        time.sleep(0.01)

# Function to visualize the 3D point
# Function to visualize the 3D point with fixed axes
def visualize_3d():
    global current_3d_point

    plt.ion()  # Turn on interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Point Visualization')

    # Set fixed axis limits
    ax.set_xlim(-200, 200)  # X-axis range in mm
    ax.set_ylim(-200, 200)  # Y-axis range in mm
    ax.set_zlim(-50, 350)   # Z-axis range in mm

    while True:
        if current_3d_point is not None:
            ax.clear()

            # Plot the current 3D point
            ax.scatter(current_3d_point[0], current_3d_point[1], current_3d_point[2], color='r', s=50)

            # Reapply axis labels, title, and limits
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('3D Point Visualization')
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_zlim(-50, 350)

            plt.pause(0.05)

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
    # visualization_thread = threading.Thread(target=visualize_3d)

    triangulation_thread.start()
    # visualization_thread.start()

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
