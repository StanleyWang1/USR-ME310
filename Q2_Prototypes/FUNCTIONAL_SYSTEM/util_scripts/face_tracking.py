import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for a mirror effect and get dimensions.
    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    
    # Convert the frame color to RGB as required by MediaPipe.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define landmark indices for key facial features:
            # Nose tip: 1, Chin: 152, Left eye left corner: 33, Right eye right corner: 263,
            # Left mouth corner: 61, Right mouth corner: 291
            landmark_indices = [1, 152, 33, 263, 61, 291]
            image_points = []
            for idx in landmark_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                image_points.append((x, y))
            image_points = np.array(image_points, dtype="double")
            
            # 3D model points of a generic face model (in millimeters)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -63.6, -12.5),         # Chin
                (-43.3, 32.7, -26.0),        # Left eye left corner
                (43.3, 32.7, -26.0),         # Right eye right corner
                (-28.9, -28.9, -24.1),       # Left mouth corner
                (28.9, -28.9, -24.1)         # Right mouth corner
            ])
            
            # Camera internals
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            
            # Solve for head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                (nose_end_point2D, _) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 1000.0)]),
                    rotation_vector, translation_vector, camera_matrix, dist_coeffs
                )
                
                # Draw a line from the nose tip to the projected point
                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
                
                # Convert rotation vector to rotation matrix
                rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                
                # Calculate Euler angles (in radians)
                sy = math.sqrt(rotation_mat[0, 0] * rotation_mat[0, 0] +
                               rotation_mat[1, 0] * rotation_mat[1, 0])
                singular = sy < 1e-6
                if not singular:
                    x_angle = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
                    y_angle = math.atan2(-rotation_mat[2, 0], sy)
                    z_angle = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
                else:
                    x_angle = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
                    y_angle = math.atan2(-rotation_mat[2, 0], sy)
                    z_angle = 0
                
                # Convert radians to degrees for display.
                # Convert radians to degrees and shift pitch by adding 180.
                pitch = np.degrees(x_angle) + 180
                if pitch > 180:
                    pitch -= 360  # Convert values above 180 to negative equivalents
                yaw = np.degrees(y_angle)
                roll = np.degrees(z_angle)

                
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # (Optional) Draw all facial landmarks.
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    cv2.imshow("Face Orientation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
