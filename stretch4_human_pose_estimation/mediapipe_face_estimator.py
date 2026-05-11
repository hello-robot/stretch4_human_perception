import cv2
import numpy as np
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None

class MediaPipeFaceEstimator:
    """
    A wrapper for Google MediaPipe Face Landmarker.
    """
    def __init__(self, model_path=None, num_faces=1, min_detection_confidence=0.1, min_presence_confidence=0.1):
        if mp is None:
            raise ImportError("The 'mediapipe' package is not installed. Please install it.")
            
        if model_path is None:
            # Default to the models directory in the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, "models", "face_landmarker.task")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' not found! Please run 'python setup_models.py'.")

        self.min_presence_confidence = min_presence_confidence
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=0.1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
            
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def predict_crop(self, crop, offset_x, offset_y):
        """
        Runs MediaPipe on a crop and returns global keypoints in absolute pixels.
        Returns an array of shape (N, 478, 3) where columns are [x, y, visibility].
        (N is the number of faces detected, up to num_faces).
        """
        crop_h, crop_w = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        
        # Perform inference
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        all_faces = []
        for face_landmarks in detection_result.face_landmarks:
            keypoints = []
            for lm in face_landmarks:
                # Convert normalized crop coordinates to absolute crop pixels
                px = lm.x * crop_w
                py = lm.y * crop_h
                
                # Shift to global pixels
                global_x = px + offset_x
                global_y = py + offset_y
                
                # Note: Face landmarks have x,y,z (depth relative to head center) and no distinct visibility. 
                # We'll assign a default visibility of 1.0 since it was detected.
                visibility = getattr(lm, 'visibility', 1.0)
                if visibility is None:
                    visibility = 1.0
                    
                presence = getattr(lm, 'presence', 1.0)
                if presence is not None and presence < self.min_presence_confidence:
                    visibility = 0.0
                    
                keypoints.append([global_x, global_y, visibility])
            all_faces.append(keypoints)
            
        return np.array(all_faces, dtype=np.float32)
