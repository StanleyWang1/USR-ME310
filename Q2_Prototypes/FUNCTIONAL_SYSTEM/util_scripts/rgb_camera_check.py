import cv2
import depthai as dai
import numpy as np
import time
import threading

# Shared frame storage for four cameras
frames = [None, None, None, None]
fps_values = [0.0, 0.0, 0.0, 0.0]

lock = threading.Lock()

def process_camera(device_info, index):
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Camera properties
    camRgb.setPreviewSize(640, 360)
    # camRgb.setPreviewSize(320, 180)
    camRgb.setFps(60)
    # camRgb.initialControl.setManualExposure(1000, 100)
    focus_value = 100
    camRgb.initialControl.setManualFocus(focus_value)
    camRgb.initialControl.setManualWhiteBalance(5500)  # Desired white balance

    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setPreviewKeepAspectRatio(False)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Connect to device (using maxUsbSpeed HIGH)
    with dai.Device(pipeline, device_info, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
        print(f'Connected to: {device_info.getMxId()}')
        qRgb = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)

        # FPS counter variables
        frame_count = 0
        start_time = time.time()

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps_values[index] = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Overlay FPS on the frame
            cv2.putText(frame, f"FPS: {fps_values[index]:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"FPS: {fps_values[index]:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save the frame in the shared list (using lock for thread safety)
            with lock:
                frames[index] = annotated_frame

if __name__ == '__main__':
    available_devices = dai.Device.getAllAvailableDevices()
    if len(available_devices) < 4:
        print("Error: Less than four OAK-D cameras connected.")
    else:
        threads = []
        # Start one thread for each of the first four devices
        for i, device_info in enumerate(available_devices[:4]):
            t = threading.Thread(target=process_camera, args=(device_info, i))
            t.start()
            threads.append(t)

        # Display loop for the 2x2 grid of camera feeds
        while True:
            with lock:
                if all(frame is not None for frame in frames):
                    # Create 2x2 grid: top row (frames 0 & 1), bottom row (frames 2 & 3)
                    top_row = np.hstack((frames[0], frames[1]))
                    bottom_row = np.hstack((frames[2], frames[3]))
                    combined_display = np.vstack((top_row, bottom_row))
                    cv2.imshow("Multi Camera View", combined_display)

            if cv2.waitKey(1) == ord('q'):
                break

        for t in threads:
            t.join()

        cv2.destroyAllWindows()
