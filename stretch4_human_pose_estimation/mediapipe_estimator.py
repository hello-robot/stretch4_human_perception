import cv2
import numpy as np
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None

from stretch4_human_pose_estimation.pose_constants import (
    MEDIAPIPE_EDGES as POSE_CONNECTIONS
)

class MediaPipePoseEstimator:
    """
    A wrapper for Google MediaPipe Pose that conforms to the TopDownPosePipeline interface.
    """
    def __init__(self, model_path=None):
        if mp is None:
            raise ImportError("The 'mediapipe' package is not installed. Please install it.")
            
        if model_path is None:
            # Default to the models directory in the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, "models", "pose_landmarker_full.task")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' not found! Please run 'python setup_models.py'.")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def predict_crop(self, crop, offset_x, offset_y):
        """
        Runs MediaPipe on a crop and returns global keypoints in absolute pixels.
        Returns an array of shape (33, 3) where columns are [x, y, visibility].
        """
        crop_h, crop_w = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        
        # Perform inference
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            return None
            
        # The result contains a list of poses, we take the first one (since num_poses=1)
        landmarks = detection_result.pose_landmarks[0]
        
        keypoints = []
        for lm in landmarks:
            # Convert normalized crop coordinates to absolute crop pixels
            px = lm.x * crop_w
            py = lm.y * crop_h
            
            # Shift to global pixels
            global_x = px + offset_x
            global_y = py + offset_y
            
            # Note: lm.visibility may be present. If not, default to 1.0
            visibility = getattr(lm, 'visibility', 1.0)
            keypoints.append([global_x, global_y, visibility])
            
        return np.array(keypoints, dtype=np.float32)

    def draw_keypoints(self, image_bgr, keypoints, threshold=0.2):
        """
        Draws the 33 MediaPipe keypoints and connections onto the image.
        """
        img_draw = image_bgr.copy()
        
        # Draw MediaPipe Connections
        for connection in POSE_CONNECTIONS:
            p1, p2 = connection
            x1, y1, c1 = keypoints[p1]
            x2, y2, c2 = keypoints[p2]
            
            if c1 > threshold and c2 > threshold:
                cv2.line(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
        # Draw joints
        for x, y, c in keypoints:
            if c > threshold:
                cv2.circle(img_draw, (int(x), int(y)), 4, (0, 0, 255), -1)
                
        return img_draw
