import cv2
import numpy as np


def estimate_pose_from_matches(kp1, kp2, matches, camera_matrix):
    """Estimate camera pose from matched keypoints"""
    if len(matches) < 8:
        return None, None
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find essential matrix
    E, mask = cv2.findEssentialMat(
        pts1, pts2, camera_matrix, 
        cv2.RANSAC, 0.999, 1.0
    )
    
    if E is None:
        return None, None
    
    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, camera_matrix)
    
    return R, t
