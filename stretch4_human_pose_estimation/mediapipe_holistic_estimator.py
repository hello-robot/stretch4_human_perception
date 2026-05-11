import cv2
import numpy as np
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None

class MediaPipeHolisticEstimator:
    """
    A wrapper for Google MediaPipe Holistic Landmarker.
    """
    def __init__(self, model_path=None, min_detection_confidence=0.1, min_presence_confidence=0.1):
        if mp is None:
            raise ImportError("The 'mediapipe' package is not installed. Please install it.")
            
        if model_path is None:
            # Default to the models directory in the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, "models", "holistic_landmarker.task")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' not found! Please run 'python setup_models.py'.")

        self.min_presence_confidence = min_presence_confidence
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_confidence,
            min_pose_detection_confidence=min_detection_confidence,
            min_face_landmarks_confidence=min_presence_confidence,
            min_pose_landmarks_confidence=min_presence_confidence,
            min_hand_landmarks_confidence=min_presence_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False)
            
        self.landmarker = mp_vision.HolisticLandmarker.create_from_options(options)

    def predict_crop(self, crop, offset_x, offset_y, outputs=['pose', 'face', 'left_hand', 'right_hand']):
        """
        Runs MediaPipe Holistic on a crop and returns a dict of global keypoints in absolute pixels.
        Dictionary will only contain the keys requested in `outputs` if they are detected.
        Values are numpy arrays of shape (K, 3) where columns are [x, y, visibility] and K is the number of keypoints.
        """
        crop_h, crop_w = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # To bypass MediaPipe's stateful SegmentationSmoothingCalculator bug in IMAGE mode,
        # we must feed a constant sized image. 
        # We scale the crop so its longest edge perfectly fills a 512x512 buffer, preserving aspect ratio.
        # This prevents the pose model from shrinking the person into a tiny corner, while 
        # keeping enough resolution (512px) for the hand and face landmarkers to extract good sub-crops.
        FIXED_SIZE = 512
        
        scale = FIXED_SIZE / max(crop_h, crop_w)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        crop_resized = cv2.resize(crop_rgb, (new_w, new_h))
            
        padded = np.zeros((FIXED_SIZE, FIXED_SIZE, 3), dtype=np.uint8)
        padded[0:new_h, 0:new_w] = crop_resized
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
        
        # Perform inference
        detection_result = self.landmarker.detect(mp_image)
        
        result_dict = {}
        
        def process_landmarks(landmarks, presence_key=True):
            if not landmarks:
                return None
            keypoints = []
            for lm in landmarks:
                # lm.x and lm.y are relative to FIXED_SIZE x FIXED_SIZE
                px = lm.x * FIXED_SIZE
                py = lm.y * FIXED_SIZE
                
                # unscale back to the original crop dimensions
                px = px / scale
                py = py / scale
                
                # Shift to global pixels
                global_x = px + offset_x
                global_y = py + offset_y
                
                visibility = getattr(lm, 'visibility', 1.0)
                if visibility is None:
                    visibility = 1.0
                    
                if presence_key:
                    presence = getattr(lm, 'presence', 1.0)
                    if presence is not None and presence < self.min_presence_confidence:
                        visibility = 0.0
                        
                keypoints.append([global_x, global_y, visibility])
            return np.array(keypoints, dtype=np.float32)

        if 'pose' in outputs:
            # HolisticLandmarker returns a single list for pose (not a list of lists) because it tracks 1 person
            if detection_result.pose_landmarks:
                result_dict['pose'] = process_landmarks(detection_result.pose_landmarks)
                
        if 'face' in outputs:
            if detection_result.face_landmarks:
                # face_landmarks doesn't have presence/visibility, default to 1.0
                result_dict['face'] = process_landmarks(detection_result.face_landmarks, presence_key=False)
                
        if 'left_hand' in outputs:
            if detection_result.left_hand_landmarks:
                result_dict['left_hand'] = process_landmarks(detection_result.left_hand_landmarks)
                
        if 'right_hand' in outputs:
            if detection_result.right_hand_landmarks:
                result_dict['right_hand'] = process_landmarks(detection_result.right_hand_landmarks)
                
        return result_dict
