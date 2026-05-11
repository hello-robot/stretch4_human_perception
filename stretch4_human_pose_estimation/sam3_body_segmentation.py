import sys
import os
import cv2
import numpy as np

class SAM3Pipeline:
    """
    SAM 3.1 Pipeline for zero-shot text-prompted segmentation.
    Requires HuggingFace authentication and the sam3 repository installed.
    """
    def __init__(self, prompt="people"):
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("PyTorch and Pillow are required. Please install torch and Pillow.")
            
        # Ensure sam3 is in the python path
        sam3_dir = os.path.expanduser("~/repos/sam3")
        if os.path.exists(sam3_dir) and sam3_dir not in sys.path:
            sys.path.insert(0, sam3_dir)
            
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            raise ImportError("SAM 3.1 is not installed correctly. Please clone https://github.com/facebookresearch/sam3 to ~/repos/sam3 and pip install -e ~/repos/sam3")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading SAM 3.1 model on {self.device_str}...")
        try:
            model = build_sam3_image_model(device=self.device_str)
        except Exception as e:
            raise RuntimeError(f"Failed to build SAM 3.1 model. Ensure you have authenticated via `huggingface-cli login`. Error: {e}")
            
        model = model.to(self.device)
        self.processor = Sam3Processor(model, device=self.device_str)
        self.prompt = prompt
        
    def predict(self, image_bgr, conf_threshold=0.5):
        """
        Run segmentation on a BGR image.
        
        Args:
            image_bgr: NumPy array of the image (BGR format).
            conf_threshold: Confidence threshold for returning masks.
            
        Returns:
            list: List of dictionaries containing 'box', 'mask', and 'keypoints' (empty).
        """
        import torch
        from PIL import Image
        
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        with torch.autocast(device_type=self.device_str, dtype=torch.bfloat16):
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=self.prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        results = []
        for i in range(len(masks)):
            score = float(scores[i])
            if score < conf_threshold:
                continue
                
            mask_data = masks[i]
            if hasattr(mask_data, 'cpu'):
                mask_data = mask_data.cpu().numpy()
            
            mask_data = np.squeeze(mask_data)
            
            mask_bin = (mask_data > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            poly = []
            if contours:
                c = max(contours, key=cv2.contourArea)
                poly = c.flatten().tolist()
                
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
                
            results.append({
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3]), score],
                "keypoints": [],
                "mask": poly
            })
            
        return results

    def visualize(self, image_bgr, results, color=(0, 255, 100), alpha=0.5, line_thickness=2):
        """
        Draw bounding boxes and masks on the image.
        
        Args:
            image_bgr: NumPy array of the image (BGR format).
            results: Results list returned by `predict`.
            color: RGB/BGR tuple for the mask color.
            alpha: Transparency of the mask overlay.
            line_thickness: Thickness of bounding boxes.
            
        Returns:
            NumPy array of the visualized image.
        """
        vis_image = image_bgr.copy()
        
        # Overlay masks
        for r in results:
            mask = r.get('mask', [])
            if len(mask) == 0:
                continue
            pts = np.array(mask, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(vis_image, [pts], color)
            
        vis_image = cv2.addWeighted(vis_image, alpha, image_bgr, 1 - alpha, 0)
        
        # Draw bounding boxes
        for r in results:
            box = [int(x) for x in r['box'][:4]]
            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), line_thickness)
            
        return vis_image


class ContinuousSAM3VideoPipeline:
    def __init__(self, prompt="people"):
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("PyTorch and Pillow are required.")
            
        sam3_dir = os.path.expanduser("~/repos/sam3")
        if os.path.exists(sam3_dir) and sam3_dir not in sys.path:
            sys.path.insert(0, sam3_dir)
            
        try:
            from sam3.model_builder import build_sam3_video_predictor
            from sam3.model.data_misc import FindStage, BatchedDatapoint, convert_my_tensors
            from sam3.model.geometry_encoders import Prompt
        except ImportError:
            raise ImportError("SAM 3.1 is not installed correctly.")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading SAM 3.1 Video Predictor on {self.device_str}...")
        
        # Build predictor
        self.predictor = build_sam3_video_predictor()
        
        # Override SAM 3.1 multiplex heuristics that silently drop unmatched tracklets
        if hasattr(self.predictor, 'model'):
            if hasattr(self.predictor.model, 'hotstart_delay'):
                self.predictor.model.hotstart_delay = 0
            if hasattr(self.predictor.model, 'suppress_unmatched_only_within_hotstart'):
                self.predictor.model.suppress_unmatched_only_within_hotstart = True
        
        self.prompt = prompt
        
        # Local refs to needed sam3 classes for state building
        self.FindStage = FindStage
        self.BatchedDatapoint = BatchedDatapoint
        self.convert_my_tensors = convert_my_tensors
        self.Prompt = Prompt
        
        # Max memory frames to keep references to in lists. SAM3 defaults to num_maskmem=7, but we keep a bit more safely
        self.keep_frames = 15

    def init_state(self, image_size=1008):
        """Initialize an empty continuous streaming inference state."""
        import torch
        with torch.inference_mode():
            device = self.device
            
            inference_state = {}
            inference_state["image_size"] = image_size
            inference_state["offload_state_to_cpu"] = False
            inference_state["num_frames"] = 0
            inference_state["orig_height"] = None
            inference_state["orig_width"] = None
            
            inference_state["constants"] = {}
            bs = 1
            inference_state["constants"]["empty_geometric_prompt"] = self.Prompt(
                box_embeddings=torch.zeros(0, bs, 4, device=device),
                box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
                box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
                point_embeddings=torch.zeros(0, bs, 2, device=device),
                point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
                point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            )
            
            # Batched input, initialized empty
            inference_state["input_batch"] = self.BatchedDatapoint(
                img_batch=[],
                find_text_batch=["<text placeholder>", "visual"],
                find_inputs=[],
                find_targets=[],
                find_metadatas=[],
            )
            
            inference_state["previous_stages_out"] = []
            inference_state["text_prompt"] = None
            inference_state["per_frame_raw_point_input"] = []
            inference_state["per_frame_raw_box_input"] = []
            inference_state["per_frame_visual_prompt"] = []
            inference_state["per_frame_geometric_prompt"] = []
            inference_state["per_frame_cur_step"] = []
            
            inference_state["visual_prompt_embed"] = None
            inference_state["visual_prompt_mask"] = None
            inference_state["tracker_inference_states"] = []
            inference_state["tracker_metadata"] = {}
            inference_state["feature_cache"] = {}
            inference_state["cached_frame_outputs"] = {}
            inference_state["action_history"] = []
            inference_state["is_image_only"] = False
            
            return inference_state

    def add_frame_to_state(self, inference_state, frame_idx, image_bgr):
        """Append the new frame to the state."""
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF
        with torch.inference_mode():
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            if inference_state["orig_height"] is None:
                inference_state["orig_height"], inference_state["orig_width"] = img_rgb.shape[:2]
                
            # Transform image
            pil_img = Image.fromarray(img_rgb)
            img_tensor = TF.resize(pil_img, size=(inference_state["image_size"], inference_state["image_size"]))
            img_tensor = TF.to_tensor(img_tensor).to(self.device, non_blocking=True)
            
            # Normalize
            img_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(3, 1, 1)
            img_std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(3, 1, 1)
            img_tensor -= img_mean
            img_tensor /= img_std
            
            # Append to batch
            batch = inference_state["input_batch"]
            batch.img_batch.append(img_tensor)
            
            # Append to find_inputs
            input_box_embedding_dim = 258
            input_points_embedding_dim = 257
            stage = self.FindStage(
                img_ids=[frame_idx],
                text_ids=[1], # default visual, overridden in add_text_prompt
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            stage = self.convert_my_tensors(stage)
            stage.img_ids = stage.img_ids.to(self.device, non_blocking=True)
            stage.text_ids = stage.text_ids.to(self.device, non_blocking=True)
            stage.input_boxes = [x.to(self.device, non_blocking=True) for x in stage.input_boxes]
            stage.input_boxes_mask = [x.to(self.device, non_blocking=True) for x in stage.input_boxes_mask]
            stage.input_boxes_label = [x.to(self.device, non_blocking=True) for x in stage.input_boxes_label]
            stage.input_points = [x.to(self.device, non_blocking=True) for x in stage.input_points]
            stage.input_points_mask = [x.to(self.device, non_blocking=True) for x in stage.input_points_mask]
            batch.find_inputs.append(stage)
            
            batch.find_targets.append(None)
            batch.find_metadatas.append(None)
            
            # Append to lists
            inference_state["previous_stages_out"].append(None)
            inference_state["per_frame_raw_point_input"].append(None)
            inference_state["per_frame_raw_box_input"].append(None)
            inference_state["per_frame_visual_prompt"].append(None)
            inference_state["per_frame_geometric_prompt"].append(None)
            inference_state["per_frame_cur_step"].append(0)
            
            inference_state["num_frames"] += 1
            for tracker_state in inference_state["tracker_inference_states"]:
                tracker_state["num_frames"] += 1
            
            # Rolling window pruning to save memory
            if frame_idx > self.keep_frames:
                idx_to_prune = frame_idx - self.keep_frames
                batch.img_batch[idx_to_prune] = None
                batch.find_inputs[idx_to_prune] = None
                inference_state["previous_stages_out"][idx_to_prune] = None

    def add_text_prompt(self, inference_state, frame_idx, obj_id, text):
        """Set the text prompt for the video sequence."""
        import torch
        with torch.inference_mode():
            inference_state["text_prompt"] = text
            inference_state["input_batch"].find_text_batch[0] = text
            text_id = 0 # TEXT_ID_FOR_TEXT in Sam3VideoInference
            
            # Set all current and future text_ids to 0
            for t in range(inference_state["num_frames"]):
                if inference_state["input_batch"].find_inputs[t] is not None:
                    inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id
                    
    def track_step(self, inference_state, frame_idx, conf_threshold=0.5):
        import torch
        with torch.inference_mode(), torch.autocast(device_type=self.device_str, dtype=torch.bfloat16):
            out = self.predictor.model._run_single_frame_inference(inference_state, frame_idx, reverse=False)
            
            # Postprocess output
            # out contains obj_id_to_mask, obj_id_to_score
            # we need to mimic the result format of SAM3Pipeline
            results = []
            
            obj_id_to_mask = out.get("obj_id_to_mask", {})
            obj_id_to_score = out.get("obj_id_to_tracker_score", out.get("obj_id_to_score", {}))
            
            H_video = inference_state["orig_height"]
            W_video = inference_state["orig_width"]
            
            for obj_id, mask_t in obj_id_to_mask.items():
                score = float(obj_id_to_score.get(obj_id, 0.0))
                if score < conf_threshold:
                    continue
                    
                # mask_t is (1, H, W) bool tensor usually, or (H, W).
                # We need to extract polygons and bounding box.
                if mask_t.dim() == 3:
                    mask_t = mask_t.squeeze(0)
                
                mask_t = mask_t.unsqueeze(0).unsqueeze(0).float() # (1, 1, H_low, W_low)
                mask_t = torch.nn.functional.interpolate(
                    mask_t, size=(H_video, W_video), mode="bilinear", align_corners=False
                )
                mask_t = mask_t.squeeze() > 0.0
                mask_data = mask_t.cpu().numpy().astype(np.uint8)
                
                contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                poly = []
                box_res = [0, 0, 0, 0, score]
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    poly = c.flatten().tolist()
                    x, y, w, h = cv2.boundingRect(c)
                    box_res = [float(x), float(y), float(x+w), float(y+h), score]
                    
                results.append({
                    "box": box_res,
                    "keypoints": [],
                    "mask": poly
                })
                
        return results
