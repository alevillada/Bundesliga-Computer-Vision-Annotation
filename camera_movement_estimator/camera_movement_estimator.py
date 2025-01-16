import pickle
import cv2
import numpy as np
import sys 
sys.path.append('../')
from utils import get_euclidean_distance, measure_xy_distance
import os 

class CameraMovementEstimator():
    """
    A class that estimates camera movement in a video by tracking feature points across frames.
    """

    def __init__(self, first_frame):
        """
            first_frame: np.array, the first frame of the video
        """
        # Set a threshold for the minimum number of features to track
        self.min_features_threshold = 7
        
        # Set a minimum distance to consider movement significant
        self.min_distance = 1
        
        # Convert the first frame to grayscale
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask to focus on specific regions (e.g., top and bottom rows)
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:, 0:20] = 1        # First 20 columns of pixels
        mask_features[:, 900:1050] = 1    # Columns from 900 to 1050 pixels
        
        # Parameters for detecting good features to track
        self.features = dict(
            maxCorners=300,
            qualityLevel=0.2,
            minDistance=3,
            blockSize=7,
            # mask=mask_features
        )
        
        # Parameters for optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        
        
        """
        Estimates the camera movement across the video frames. This uses Sparse Optical Flow to track feature points

        Args:
            frames (list): List of frames in the video.
            read_from_stub (bool): If True, read the camera movement from a stub file.
            stub_path (str): Path to the stub file.

        Returns:
            camera_movement (list): List containing camera movement vectors for each frame.
        """
        # Read from stub file if specified
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
                print(f"Camera movement loaded from {stub_path}")
            return camera_movement

        # Initialize camera movement list
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and detect initial features
        previous_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        previous_frame_features = cv2.goodFeaturesToTrack(previous_frame_gray, **self.features)

        # Iterate over each frame starting from the second
        for frame_num in range(1, len(frames)):
            current_frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow to track feature points
            current_frame_features, status, error = cv2.calcOpticalFlowPyrLK(
                previous_frame_gray,
                current_frame_gray,
                previous_frame_features,
                None,
                **self.lk_params
            )

            # Select good points where the tracking was successful
            status = status.reshape(-1)
            good_new = current_frame_features[status == 1]
            good_old = previous_frame_features[status == 1]

            # Ensure the points are in the shape (num_points, 2)
            good_new = good_new.reshape(-1, 2)
            good_old = good_old.reshape(-1, 2)

            # Compute movement vectors for the good points
            movement_vectors = good_new - good_old

            if len(movement_vectors) > 0:
                # Compute the median movement vector to reduce the effect of outliers
                mean_movement = np.mean(movement_vectors, axis=0)

                # Check if the movement is significant
                if np.linalg.norm(mean_movement) > self.min_distance:
                    camera_movement[frame_num] = [mean_movement[0], mean_movement[1]]
                else:
                    camera_movement[frame_num] = [0, 0]

                # Update features based on the number of good points
                if len(good_new) < self.min_features_threshold:
                    # Re-detect features if too few are left
                    previous_frame_features = cv2.goodFeaturesToTrack(current_frame_gray, **self.features)
                else:
                    # Continue tracking the good points
                    previous_frame_features = good_new.reshape(-1, 1, 2)
            else:
                # If no good points, re-detect features
                previous_frame_features = cv2.goodFeaturesToTrack(current_frame_gray, **self.features)
                camera_movement[frame_num] = [0, 0]

            # Update the previous frame to the current one for the next iteration
            previous_frame_gray = current_frame_gray.copy()

        # Save to stub file if specified
        if (stub_path!= None) and (read_from_stub == True):
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)
                print(f"Camera movement saved to {stub_path}")

        print(f"Frame {frame_num}: Median Movement = {mean_movement}, Norm = {np.linalg.norm(mean_movement)}")
        return camera_movement