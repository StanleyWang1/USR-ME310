import cv2, mediapipe as mp, numpy as np, math\

def head_tracker():
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmark_indices = [1,152,33,263,61,291]
                    image_points = []
                    for idx in landmark_indices:
                        lm = face_landmarks.landmark[idx]
                        image_points.append((int(lm.x * img_w), int(lm.y * img_h)))
                    image_points = np.array(image_points, dtype="double")
                    model_points = np.array([[0.0,0.0,0.0],[0.0,-63.6,-12.5],[-43.3,32.7,-26.0],[43.3,32.7,-26.0],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]])
                    focal_length = img_w
                    center = (img_w/2, img_h/2)
                    camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]], dtype="double")
                    dist_coeffs = np.zeros((4,1))
                    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    if success:
                        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                        sy = math.sqrt(rotation_mat[0,0]**2 + rotation_mat[1,0]**2)
                        singular = sy < 1e-6
                        if not singular:
                            x_angle = math.atan2(rotation_mat[2,1], rotation_mat[2,2])
                            y_angle = math.atan2(-rotation_mat[2,0], sy)
                            z_angle = math.atan2(rotation_mat[1,0], rotation_mat[0,0])
                        else:
                            x_angle = math.atan2(-rotation_mat[1,2], rotation_mat[1,1])
                            y_angle = math.atan2(-rotation_mat[2,0], sy)
                            z_angle = 0
                            
                        yaw = np.degrees(y_angle)

                        pitch = np.degrees(x_angle) + 180
                        if pitch > 180: pitch -= 360

                        yield yaw, pitch
    finally:
        cap.release()
