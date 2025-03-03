import cv2
import depthai as dai
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import smooth_keypoints
from camera_config import P1, P2
from robot_sim import initialize_simulation, command_sphere_position, close_simulation

# Shared frame storage for both cameras
frames = [None, None]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
fps_values = [0.0, 0.0]

# Global variables to store the most recent blob coordinates
c1_coords = np.zeros((3,3))  # Camera 1 point coordinates --> each column is (u,v,hue)
c2_coords = np.zeros((3,3))  # Camera 2 point coordinates --> each column is (u,v,hue)

# Global variable for the latest 3D point
current_3d_points = np.zeros((4, 3))

# Camera processing function
def process_camera(device_info, index):
    global c1_coords, c2_coords

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

        # Store previous keypoints to compare with current keypoints
        prev_keypoints = []

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Process the frame to detect blobs
            # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            # lower_hsv = np.array([0, 0, 254])
            # upper_hsv = np.array([180, 50, 255])
            # mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
            # mask = cv2.medianBlur(mask, 5)

            keypoints = blob_detector.detect(frame)

            # Use in the main loop
            keypoints = smooth_keypoints(keypoints, prev_keypoints)
            prev_keypoints = keypoints
            # Update global coordinates with the first detected blob (if any)
            cam_hues = np.zeros(3)
            if keypoints:
                coordinates = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints[:3]]
                c_ref = c1_coords if index == 0 else c2_coords

                for i, (x, y) in enumerate(coordinates):
                    size = int(keypoints[i].size / 2)
                    roi = frame_hsv[max(0, y - size):min(frame_hsv.shape[0], y + size), max(0, x - size):min(frame_hsv.shape[1], x + size)]
                    c_ref[:, i] = [x, y, cv2.mean(roi)[0] if roi.size > 0 else 0]
                    cam_hues[i] = cv2.mean(roi)[0] if roi.size > 0 else 0
            # Sort based on hue
            sorted_indices = np.argsort(cam_hues)
            keypoints = [keypoints[i] for i in sorted_indices]
            if index == 0:
                c1_coords = c1_coords[:, sorted_indices]
            else:
                c2_coords = c2_coords[:, sorted_indices]
            cam_hues.sort()

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
                x, y = int(kp.pt[0]), int(kp.pt[1])
                color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(cam_hues[i]), 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0])
                # color = colors[i % len(colors)]  # Cycle through colors if there are more keypoints than colors
                # Draw the keypoint and annotation
                cv2.circle(annotated_frame, (x, y), 5, color_bgr, -1)
                cv2.putText(annotated_frame, f"({x}, {y})", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

            # Store the annotated frame for display
            frames[index] = annotated_frame

def triangulate():
    global c1_coords, c2_coords, current_3d_points

    while True:
        for i in range(3):
            cam1_point = np.vstack((c1_coords[:2, i].reshape(2, 1), [1]))
            # cam1_hue = c1_coords[2][i]
            # closest_match_idx = np.argmin(np.abs(c2_coords[2, :] - cam1_hue))
            # cam2_point = np.vstack((c2_coords[:2, closest_match_idx].reshape(2, 1), [1]))
            cam2_point = np.vstack((c2_coords[:2, i].reshape(2, 1), [1]))
            A = np.vstack((P1, P2))
            b = np.vstack((cam1_point, cam2_point))
            X_world = np.linalg.inv(A.T @ A) @ A.T @ b
            current_3d_points[i] = 0.3 * (current_3d_points[i]) + 0.7 * (X_world[:3].reshape(1, 3))
        # print(current_3d_points[0])
            # print(current_3d_points[:,i])

# def triangulate():
#     global c1_coords, c2_coords, current_3d_points
    
#     # Establish all 6 possible point correspondences
#     p = np.zeros((18, 6))
#     while True:
#         p[:,0] = np.vstack((
#             np.vstack((c1_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 2].reshape(2, 1), [1])))).flatten()
#         p[:,1] = np.vstack((
#             np.vstack((c1_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 1].reshape(2, 1), [1])))).flatten()
#         p[:,2] = np.vstack((
#             np.vstack((c1_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 2].reshape(2, 1), [1])))).flatten()
#         p[:,3] = np.vstack((
#             np.vstack((c1_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 0].reshape(2, 1), [1])))).flatten()
#         p[:,4] = np.vstack((
#             np.vstack((c1_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 1].reshape(2, 1), [1])))).flatten()
#         p[:,5] = np.vstack((
#             np.vstack((c1_coords[:2, 0].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 1].reshape(2, 1), [1])),
#             np.vstack((c1_coords[:2, 2].reshape(2, 1), [1])),
#             np.vstack((c2_coords[:2, 0].reshape(2, 1), [1])))).flatten()

#         A_diag = np.vstack((P1, P2))
#         A = np.block([
#                     [A_diag, np.zeros((6, 4)), np.zeros((6, 4))],
#                     [np.zeros((6, 4)), A_diag, np.zeros((6, 4))],
#                     [np.zeros((6, 4)), np.zeros((6, 4)), A_diag]
#                 ])
        
#         error_threshold = 1e9
#         for i in range(6):
#             X_world_candidate = np.linalg.inv(A.T @ A) @ A.T @ p[:, i]
#             error = np.linalg.norm(A@X_world_candidate - p[:, i])
#             if error < error_threshold:
#                 error_threshold = error
#                 current_3d_points = X_world_candidate.reshape((3, 4))
#         time.sleep(0.01)
    
# # Function to perform real-time 3D triangulation
# def triangulate():
#     global c1_coords, c2_coords, current_3d_points

#     while True:
#         if c1_coords is not None and c2_coords is not None:
#             skip_indices = [-1] * c1_coords.shape[1]
#             for i in range(c1_coords.shape[1]):
#                 cam1_point = np.vstack((c1_coords[:, i].reshape(2, 1), [1]))

#                 error_threshold = 1e9
#                 for j in range(c2_coords.shape[1]):
#                     if j in skip_indices:
#                         continue
#                     else:
#                         cam2_point = np.vstack((c2_coords[:, j].reshape(2, 1), [1]))
#                         A = np.vstack((P1, P2))
#                         b = np.vstack((cam1_point, cam2_point))
#                         # Solve for x and calculate Ax
#                         X_world_candidate = np.linalg.inv(A.T @ A) @ (A.T @ b)
#                         error = np.linalg.norm(A@X_world_candidate - b)
#                         if error < error_threshold:
#                             skip_indices[i] = j
#                             error_threshold = error
#                             current_3d_points[i] = X_world_candidate[:3].reshape(3)
#         time.sleep(0.01)

# Function to visualize the 3D point
# Function to visualize the 3D point with fixed axes
def visualize_3d():
    global current_3d_points

    initialize_simulation()
    while True:
        command_sphere_position(current_3d_points)
        time.sleep(0.01)

    # plt.ion()  # Turn on interactive mode
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.set_title('3D Point Visualization')

    # # Set fixed axis limits
    # ax.set_xlim(-200, 200)  # X-axis range in mm
    # ax.set_ylim(-200, 200)  # Y-axis range in mm
    # ax.set_zlim(-50, 350)   # Z-axis range in mm

    # while True:
    #     if current_3d_points is not None:
    #         ax.clear()

    #         # Plot each 3D point from current_3d_points
    #         for point in current_3d_points:
    #             ax.scatter(point[0], point[1], point[2], color='r', s=50)

    #         # Reapply axis labels, title, and limits
    #         ax.set_xlabel('X (mm)')
    #         ax.set_ylabel('Y (mm)')
    #         ax.set_zlabel('Z (mm)')
    #         ax.set_title('3D Point Visualization')
    #         ax.set_xlim(-200, 200)
    #         ax.set_ylim(-200, 200)
    #         ax.set_zlim(-50, 350)

    #         plt.pause(0.05)

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
    visualization_thread = threading.Thread(target=visualize_3d)

    triangulation_thread.start()
    visualization_thread.start()

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
