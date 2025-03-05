import numpy as np
import cv2

from camera_config import K_inv, W_T_C1, W_T_C2

def smooth_keypoints(keypoints, prev_keypoints, distance_threshold=1000):
    # Ensure `prev_keypoints` always has exactly three keypoints
    if not prev_keypoints or len(prev_keypoints) < 3:
        prev_keypoints = [cv2.KeyPoint(100, 300, 1)] * 3

    # Extract coordinates of current and previous keypoints
    current_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2))
    prev_pts = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32)

    # Match new keypoints to previous keypoints using distance matrix
    if len(current_pts) > 0:
        distances = np.linalg.norm(prev_pts[:, None] - current_pts[None, :], axis=2)

        # Find the closest matches (greedy approach)
        matches = np.full(3, -1, dtype=int)  # -1 indicates no match
        used_current = set()

        for i in range(3):
            if len(current_pts) == 0:
                break

            min_dist = np.inf
            best_match = -1

            for j in range(len(current_pts)):
                if j not in used_current and distances[i, j] < min_dist and distances[i, j] < distance_threshold:
                    min_dist = distances[i, j]
                    best_match = j

            if best_match != -1:
                matches[i] = best_match
                used_current.add(best_match)

        # Update previous keypoints with matched or fallback points
        updated_keypoints = []
        for i in range(3):
            if matches[i] != -1:
                updated_keypoints.append(keypoints[matches[i]])
            else:
                updated_keypoints.append(prev_keypoints[i])
    else:
        # No new keypoints, return previous keypoints
        updated_keypoints = prev_keypoints

    # Ensure exactly three keypoints are returned
    return updated_keypoints
