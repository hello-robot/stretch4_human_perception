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
    MEDIAPIPE_HAND_EDGES as HAND_CONNECTIONS
)

class MediaPipeHandEstimator:
    """
    A wrapper for Google MediaPipe Hand Landmarker.
    """
    def __init__(self, model_path=None, num_hands=1, min_detection_confidence=0.1, min_presence_confidence=0.1):
        if mp is None:
            raise ImportError("The 'mediapipe' package is not installed. Please install it.")
            
        if model_path is None:
            # Default to the models directory in the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, "models", "hand_landmarker.task")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' not found! Please run 'python setup_models.py'.")

        self.min_presence_confidence = min_presence_confidence
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=0.1)
            
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def predict_crop(self, crop, offset_x, offset_y):
        """
        Runs MediaPipe on a crop and returns global keypoints in absolute pixels.
        Returns an array of shape (N, 21, 3) where columns are [x, y, visibility].
        (N is the number of hands detected, up to num_hands).
        """
        crop_h, crop_w = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        
        # Perform inference
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            return None
            
        all_hands = []
        for hand_landmarks in detection_result.hand_landmarks:
            keypoints = []
            for lm in hand_landmarks:
                # Convert normalized crop coordinates to absolute crop pixels
                px = lm.x * crop_w
                py = lm.y * crop_h
                
                # Shift to global pixels
                global_x = px + offset_x
                global_y = py + offset_y
                
                # Note: Hand landmarks usually have x,y,z (depth relative to wrist) and no distinct visibility. 
                # We'll assign a default visibility of 1.0 since it was detected.
                visibility = getattr(lm, 'visibility', 1.0)
                if visibility is None:
                    visibility = 1.0
                    
                presence = getattr(lm, 'presence', 1.0)
                if presence is not None and presence < self.min_presence_confidence:
                    visibility = 0.0
                    
                keypoints.append([global_x, global_y, visibility])
            all_hands.append(keypoints)
            
        return np.array(all_hands, dtype=np.float32)

    def draw_keypoints(self, image_bgr, keypoints, threshold=0.2):
        """
        Draws the 21 MediaPipe hand keypoints and connections onto the image.
        keypoints here should be shape (21, 3) for a single hand.
        """
        img_draw = image_bgr.copy()
        
        # Draw MediaPipe Connections
        for connection in HAND_CONNECTIONS:
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
