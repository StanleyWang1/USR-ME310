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
    pipeline = dai.Pipeline()

    # Create MonoCamera node (OV9282 sensor)
    monoCam = pipeline.create(dai.node.MonoCamera)
    # Create an ImageManip node to downscale the output
    image_manip = pipeline.create(dai.node.ImageManip)
    # Create XLinkOut node to send frames to the host
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("mono")

    # Use only the RIGHT sensor (updated naming: CAM_C)
    monoCam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    # Set the sensor resolution to 400p (lowest available for OV9282)
    monoCam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # Set sensor frame rate to 120 fps
    monoCam.setFps(120)

    monoCam.initialControl.setManualExposure(100, 100)
    focus_value = 100
    monoCam.initialControl.setManualFocus(focus_value)
    monoCam.initialControl.setManualWhiteBalance(5500)  # Desired white balance

    # Configure the ImageManip node to resize frames to 160x120
    # image_manip.initialConfig.setResize(160, 120) 
    image_manip.initialConfig.setResize(320, 200) 
    # Set frame type explicitly to grayscale if desired
    image_manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

    # Link nodes: MonoCamera -> ImageManip -> XLinkOut
    monoCam.out.link(image_manip.inputImage)
    image_manip.out.link(xout.input)

    # Connect to device (using maxUsbSpeed HIGH)
    with dai.Device(pipeline, device_info, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
        print(f"Connected to: {device_info.getMxId()}")
        q = device.getOutputQueue(name="mono", maxSize=8, blocking=False)

        frame_count = 0
        start_time = time.time()

        while True:
            inFrame = q.get()
            frame = inFrame.getCvFrame()  # Frame is now 160x120

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps_values[index] = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Overlay FPS on the frame
            cv2.putText(frame, f"FPS: {fps_values[index]:.2f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            with lock:
                frames[index] = frame

if __name__ == "__main__":
    available_devices = dai.Device.getAllAvailableDevices()
    if len(available_devices) < 4:
        print("Error: Less than four OAK-D cameras connected.")
    else:
        threads = []
        # Start one thread per device (first four devices)
        for i, device_info in enumerate(available_devices[:4]):
            t = threading.Thread(target=process_camera, args=(device_info, i))
            t.start()
            threads.append(t)

        # Display loop: arrange feeds in a 2x2 grid
        while True:
            with lock:
                if all(frame is not None for frame in frames):
                    top_row = np.hstack((frames[0], frames[1]))
                    bottom_row = np.hstack((frames[2], frames[3]))
                    combined_display = np.vstack((top_row, bottom_row))
                    cv2.namedWindow("Multi Camera View", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Multi Camera View", 1280, 720)
                    cv2.imshow("Multi Camera View", combined_display)
            if cv2.waitKey(1) == ord("q"):
                break

        for t in threads:
            t.join()

        cv2.destroyAllWindows()
