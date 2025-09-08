import numpy as np
from utils.tracker import FeatureTracker
from utils.pose_estimation import estimate_pose_from_matches

class SLAM:
    def __init__(self, camera_matrix, nfeatures=1000):
        self.tracker = FeatureTracker(nfeatures)
        self.camera_matrix = camera_matrix
        
        self.keyframes = []
        self.map_points = []  # 3D points
        self.poses = []       # Camera poses (4x4 matrices)
        
        self.current_pose = np.eye(4)  
        self.previous_frame = None
        self.previous_kp = None
        self.previous_desc = None
        
        self.point_cloud = []  # List of (x, y, z, color) tuples
        
    def process_frame(self, frame, min_points=200):
        kp, desc = self.tracker.extract_features(frame)
        
        if self.previous_frame is not None and self.previous_desc is not None:
            matches = self.tracker.match_features(self.previous_desc, desc)
            
            if len(matches) >= 8:  
                R, t = estimate_pose_from_matches(
                    self.previous_kp, kp, matches, self.camera_matrix
                )
                
                if R is not None and t is not None:
                    self.update_pose(R, t)
                    # Pass current keypoints to triangulate_points
                    self.triangulate_points(matches, kp, min_points=min_points)

                if self.should_create_keyframe(len(matches)):
                    self.create_keyframe(frame, kp, desc)
        
        self.previous_frame = frame.copy()
        self.previous_kp = kp
        self.previous_desc = desc
        
        return self.current_pose.copy(), kp
    
    def update_pose(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        self.current_pose = self.current_pose @ T
        
        self.poses.append(self.current_pose.copy())
    
    def triangulate_points(self, matches, current_kp, min_points=100):
        if len(matches) < 2 or self.previous_kp is None:
            return
            
        pts1 = np.float32([self.previous_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([current_kp[m.trainIdx].pt for m in matches])
        
        # TODO: dont use pseudo 3D points, use real triangulation
        for i in range(min(min_points, len(pts1))):  
            x = (pts1[i][0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
            y = (pts1[i][1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
            z = 1.0  
            
            point_world = self.current_pose[:3, :3] @ np.array([x, y, z]) + self.current_pose[:3, 3]
            
            color = (0, 255, 0)
            self.point_cloud.append((point_world[0], point_world[1], point_world[2], color))
        
    def should_create_keyframe(self, match_count):
        return match_count < 50 or len(self.keyframes) == 0
    
    def create_keyframe(self, frame, kp, desc):
        keyframe = {
            'frame': frame.copy(),
            'keypoints': kp,
            'descriptors': desc,
            'pose': self.current_pose.copy()
        }
        self.keyframes.append(keyframe)
    
    def get_trajectory(self):
        return np.array(self.poses)[:, :3, 3]
    
    def get_poses(self):
        return self.poses
    
    def get_point_cloud(self):
        return self.point_cloud
    
    def get_frame_with_keypoints(self, frame, keypoints):
        return self.tracker.get_frame_with_keypoints(frame, keypoints)
