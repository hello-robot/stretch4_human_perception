import os
import cv2
import numpy as np

from stretch4_human_pose_estimation.pose_constants import (
    RTMO_KEYPOINT_LABELS as KEYPOINT_LABELS,
    RTMO_KEYPOINT_COLORS_RGB as CVPR_KEYPOINT_COLORS_RGB,
    RTMO_EDGE_COLORS_RGB as CVPR_EDGE_COLORS_RGB,
    RTMO_EDGES as EDGES
)

try:
    from openvino import Core
except ImportError:
    Core = None

class RTMOPipeline:
    """
    RTMO Pipeline for single-stage multi-person human pose estimation.
    Uses OpenVINO for hardware-accelerated inference.
    """
    def __init__(self, size="m", device="AUTO"):
        """
        Initialize the RTMO pipeline.
        
        Args:
            size (str): Model size ('t', 's', 'm', 'l'). Default is 'm'.
            device (str): Inference device ('AUTO', 'CPU', 'NPU', 'GPU'). Default is 'AUTO'.
        """
        if Core is None:
            raise ImportError("OpenVINO is not installed. Please install openvino.")
            
        if size not in ['t', 's', 'm', 'l']:
            raise ValueError("Size must be one of 't', 's', 'm', 'l'")
            
        self.size = size
        self.device = device
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        xml_path = os.path.join(models_dir, f"rtmo-{self.size}.xml")
        onnx_path = os.path.join(models_dir, f"rtmo-{self.size}.onnx")
        
        if os.path.exists(xml_path):
            self.model_path = xml_path
        elif os.path.exists(onnx_path):
            self.model_path = onnx_path
        else:
            raise FileNotFoundError(
                f"Model not found. Looked for '{xml_path}' and '{onnx_path}'. "
                f"Please run `human_pose_estimation_setup_models --size {self.size}` or `python3 setup_models.py --size {self.size}` from the root directory to download it."
            )
            
        # Determine input size based on model size variant
        self.input_w = 416 if self.size == 't' else 640
        self.input_h = self.input_w
            
        ie = Core()
        model = ie.read_model(model=self.model_path)
        
        # RTMO has dynamic shapes [?, 3, H, W]. Reshape to static [1, 3, H, W] for optimal execution.
        shapes = {}
        for input_layer in model.inputs:
            shapes[input_layer] = [1, 3, self.input_h, self.input_w]
        model.reshape(shapes)
        
        if self.device == "NPU":
            print("\n[WARNING] The OpenVINO Intel NPU compiler currently has a known bug executing the dynamic post-processing (NMS) operations in RTMO models. This results in empty or garbage bounding boxes. Automatically falling back to GPU for hardware acceleration.\n")
            self.device = "GPU"

        compile_config = {}
        if self.device in ["NPU", "AUTO"]:
            compile_config["NPU_COMPILER_TYPE"] = "DRIVER"

        self.compiled_model = ie.compile_model(model=model, device_name=self.device, config=compile_config)
        self.input_name = self.compiled_model.input(0).any_name

    def preprocess(self, image_bgr):
        img = cv2.resize(image_bgr, (self.input_w, self.input_h))
        # RTMO ONNX graph from bukuroo expects raw 0-255 BGR input
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_bgr, conf_threshold=0.5):
        """
        Run inference on a BGR image.
        
        Args:
            image_bgr: NumPy array of the image (BGR format).
            conf_threshold: Confidence threshold for person detection.
            
        Returns:
            list: List of dictionaries containing 'box' and 'keypoints'.
        """
        orig_h, orig_w = image_bgr.shape[:2]
        img = self.preprocess(image_bgr)
        
        outputs = self.compiled_model({self.input_name: img})
        
        # Find 'dets' and 'keypoints'
        dets = None
        keypoints = None
        for key, val in outputs.items():
            if "dets" in key.any_name:
                dets = val[0] # [num_boxes, 5]
            elif "keypoints" in key.any_name:
                keypoints = val[0] # [num_boxes, 17, 3]
                
        if dets is None or keypoints is None:
            # Fallback by shape if names differ
            vals = list(outputs.values())
            if len(vals[0].shape) == 3 and vals[0].shape[-1] == 5:
                dets = vals[0][0]
                keypoints = vals[1][0]
            else:
                dets = vals[1][0]
                keypoints = vals[0][0]
        
        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h
        
        results = []
        for i in range(len(dets)):
            score = dets[i, 4]
            if score > conf_threshold:
                xmin = dets[i, 0] * scale_x
                ymin = dets[i, 1] * scale_y
                xmax = dets[i, 2] * scale_x
                ymax = dets[i, 3] * scale_y
                
                kpts = []
                for k in range(keypoints.shape[1]):
                    x = keypoints[i, k, 0] * scale_x
                    y = keypoints[i, k, 1] * scale_y
                    conf = keypoints[i, k, 2] if keypoints.shape[-1] > 2 else 1.0
                    kpts.append([float(x), float(y), float(conf)])
                    
                results.append({
                    "box": [float(xmin), float(ymin), float(xmax), float(ymax), float(score)],
                    "keypoints": kpts
                })
        return results

    def visualize(self, image_bgr, results, kpt_thr=0.3, style="cvpr", line_thickness=2, point_radius=3):
        """
        Draw bounding boxes and keypoint skeletons on the image.
        
        Args:
            image_bgr: NumPy array of the image (BGR format).
            results: Results list returned by `predict`.
            kpt_thr: Confidence threshold for drawing a keypoint/edge.
            style: Visualization style ('cvpr' or 'red_green'). Default is 'cvpr'.
            
        Returns:
            NumPy array of the visualized image.
        """
        vis_image = image_bgr.copy()
        

        
        for res in results:
            box = res["box"]
            xmin, ymin, xmax, ymax = map(int, box[:4])
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (255, 0, 255), line_thickness)
            
            kpts = res["keypoints"]
            
            # Draw edges
            for i, (p1, p2) in enumerate(EDGES):
                if p1 < len(kpts) and p2 < len(kpts):
                    x1, y1, c1 = kpts[p1]
                    x2, y2, c2 = kpts[p2]
                    if c1 > kpt_thr and c2 > kpt_thr:
                        if style == "cvpr" and i < len(CVPR_EDGE_COLORS_RGB):
                            color = tuple(CVPR_EDGE_COLORS_RGB[i][::-1]) # RGB to BGR
                        else:
                            color = (0, 255, 0) # Default green
                        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
            
            # Draw keypoints
            for i, kp in enumerate(kpts):
                x, y, conf = kp
                if conf > kpt_thr:
                    if style == "cvpr" and i < len(CVPR_KEYPOINT_COLORS_RGB):
                        color = tuple(CVPR_KEYPOINT_COLORS_RGB[i][::-1]) # RGB to BGR
                    else:
                        color = (0, 0, 255) # Default red
                    cv2.circle(vis_image, (int(x), int(y)), point_radius, color, -1)
        return vis_image

    def convert_to_labeled_dict(self, results):
        """
        Convert pose estimation results to use a dictionary where keypoints are indexed 
        by descriptive text labels standard in the CVPR community.
        
        Args:
            results: Results list returned by `predict`.
            
        Returns:
            list: List of dictionaries with 'box' and 'keypoints' (as a dict).
        """
        labeled_results = []
        for res in results:
            kpts = res["keypoints"]
            kpt_dict = {}
            for i, kp in enumerate(kpts):
                if i < len(KEYPOINT_LABELS):
                    kpt_dict[KEYPOINT_LABELS[i]] = kp
                else:
                    kpt_dict[f"keypoint_{i}"] = kp
            labeled_results.append({
                "box": res["box"],
                "keypoints": kpt_dict
            })
        return labeled_results
