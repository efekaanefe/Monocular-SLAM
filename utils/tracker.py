import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureTracker:
    def __init__(self, nfeatures):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def extract_features(self, frame):
        """Extract ORB features from frame"""
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []
        matches = self.bf_matcher.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)
    
    def filter_good_matches(self, matches, ratio=0.7):
        """Filter good matches using ratio test"""
        if len(matches) < 2:
            return matches
        return [m for m in matches if m.distance < ratio * 255]

    def get_frame_with_keypoints(self, frame, keypoints):
        return cv2.drawKeypoints(
            frame, 
            keypoints, 
            None, 
            color=(0, 255, 0), # BGR color for green
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT # DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS  DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

def load_image_from_file(image_path):
    return cv2.imread(image_path)

if __name__ == "__main__":
    image_file = './videos/test.png'  
    frame = load_image_from_file(image_file)

    tracker = FeatureTracker(20000)
    keypoints, descriptors = tracker.extract_features(frame)
    frame_with_keypoints = tracker.get_frame_with_keypoints(frame, keypoints)
    
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(frame_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("ORB Features Detected")
    plt.axis('off')  
    plt.show()
            
