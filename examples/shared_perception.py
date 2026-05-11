import cv2
import numpy as np
import rerun as rr

import fist_bump_demo_config as config
from stretch4_emulated_rgbd.api import unproject_points

COLORS = [
    [255, 0, 0],   # Red
    [0, 255, 0],   # Green
    [0, 0, 255],   # Blue
    [255, 255, 0], # Yellow
    [0, 255, 255], # Cyan
    [255, 0, 255], # Magenta
    [255, 128, 0], # Orange
    [128, 0, 255], # Purple
]

def remap_coordinate(u, v, c_name, orig_w, orig_h):
    if c_name == "left":
        return orig_w - 1.0 - v, u
    elif c_name == "right":
        return v, orig_h - 1.0 - u
    return u, v

def project_points_to_3d(u_vals, v_vals, z_vals, camera_matrix, dist_coeffs):
    if len(u_vals) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    uv = np.stack([u_vals, v_vals], axis=-1)
    return unproject_points(uv, z_vals, camera_matrix, dist_coeffs, camera_model="fisheye")

class SegmentedObjectTracker:
    def __init__(self, pipeline, prompt):
        self.pipeline = pipeline
        self.prompt = prompt
        self.inference_states = {}
        self.frames_received_per_cam = {}
        self.prev_num_people_cache = {}

    def process_frame(self, frame, depth_mask=None, max_pts_to_disp=5000):
        c_name = frame.camera_type
        
        image_bgr = frame.image.copy()
        orig_h, orig_w = image_bgr.shape[:2]
        
        # Unrotate if left/right to pass upright image to SAM
        if c_name == "left":
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif c_name == "right":
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
            
        if c_name not in self.inference_states:
            self.inference_states[c_name] = self.pipeline.init_state()
            self.frames_received_per_cam[c_name] = 0
            
        frame_idx = self.frames_received_per_cam[c_name]
        inf_state = self.inference_states[c_name]
        
        self.pipeline.add_frame_to_state(inf_state, frame_idx, image_bgr)
        if frame_idx == 0:
            self.pipeline.add_text_prompt(inf_state, frame_idx, obj_id=1, text=self.prompt)
        results = self.pipeline.track_step(inf_state, frame_idx)
        
        self.frames_received_per_cam[c_name] += 1
        
        # Clears old rerun annotations
        prev_num = self.prev_num_people_cache.get(c_name, 0)
        curr_num = len(results)
        for i in range(curr_num, prev_num):
            rr.log(f"SegmentedPeople/{c_name}/person_{i}_pts", rr.Clear(recursive=True))
            rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", rr.Clear(recursive=True))
            rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", rr.Clear(recursive=True))
            rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}", rr.Clear(recursive=True))
        self.prev_num_people_cache[c_name] = curr_num

        camera_matrix = frame.camera_matrix
        dist_coeffs = frame.distortion_coefficients
        
        T_cam_to_base = None
        if frame.T_base_to_cam is not None:
            T_cam_to_base = np.linalg.inv(frame.T_base_to_cam)
        else:
            T_cam_to_base = np.eye(4)

        objects_3d_info = {}

        for i, res in enumerate(results):
            color = COLORS[i % len(COLORS)]
            
            rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", rr.Clear(recursive=True))
            rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", rr.Clear(recursive=True))
            
            # 1. Remap and visualize 2D boxes
            box = res.get("box", None)
            if box is not None:
                xmin, ymin, xmax, ymax = box[:4]
                pt1 = remap_coordinate(xmin, ymin, c_name, orig_w, orig_h)
                pt2 = remap_coordinate(xmax, ymax, c_name, orig_w, orig_h)
                new_xmin = min(pt1[0], pt2[0])
                new_xmax = max(pt1[0], pt2[0])
                new_ymin = min(pt1[1], pt2[1])
                new_ymax = max(pt1[1], pt2[1])
                
                center = [(new_xmin + new_xmax) / 2, (new_ymin + new_ymax) / 2]
                half_sizes = [(new_xmax - new_xmin) / 2, (new_ymax - new_ymin) / 2]
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/bbox", rr.Boxes2D(centers=[center], half_sizes=[half_sizes], colors=[color]))
                    
            mask = np.array(res.get("mask", []))
            if len(mask) > 0:
                pts = np.array(mask, np.float32).reshape((-1, 2))
                pts_orig = np.empty_like(pts)
                for j, (u, v) in enumerate(pts):
                    orig_u, orig_v = remap_coordinate(u, v, c_name, orig_w, orig_h)
                    pts_orig[j] = [orig_u, orig_v]
                    
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/mask", rr.LineStrips2D([pts_orig], colors=[color]))
                
                # 2. Extract 3D Segmented Points
                if frame.depth_image is not None and len(frame.depth_image) > 0:
                    pts_reshaped = pts_orig.astype(np.int32).reshape(-1, 1, 2)
                    
                    binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    cv2.fillPoly(binary_mask, [pts_reshaped], 1)

                    depth_image = frame.depth_image
                    valid_depth_mask = (depth_image > 0) & (depth_image != np.inf) & (~np.isnan(depth_image)) & (binary_mask == 1)
                    if depth_mask is not None:
                        valid_depth_mask &= (depth_mask > 0)
                        
                    v_vals, u_vals = np.where(valid_depth_mask)
                    z_vals = depth_image[valid_depth_mask]
                    
                    pts_3d_cam = project_points_to_3d(u_vals, v_vals, z_vals, camera_matrix, dist_coeffs)
                    if len(pts_3d_cam) > 0:
                        pts_3d_base = np.dot(T_cam_to_base[:3, :3], pts_3d_cam.T).T + T_cam_to_base[:3, 3]
                        
                        # Median of all points
                        median_3d = np.median(pts_3d_base, axis=0)
                        median_cam = np.median(pts_3d_cam, axis=0)
                        
                        # Project back to image
                        med_cam_np = np.array([median_cam], dtype=np.float32)
                        rvec = np.zeros(3)
                        tvec = np.zeros(3)
                        if len(dist_coeffs) == 4:
                            img_pts, _ = cv2.fisheye.projectPoints(med_cam_np.reshape(1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
                        else:
                            img_pts, _ = cv2.projectPoints(med_cam_np, rvec, tvec, camera_matrix, dist_coeffs)
                        u_proj = img_pts[0][0][0]
                        v_proj = img_pts[0][0][1]
                        
                        # Planar distance calculation
                        planar_dists = np.sqrt(pts_3d_base[:, 0]**2 + pts_3d_base[:, 1]**2)
                        
                        num_pts = len(planar_dists)
                        num_closest = max(1, int(0.05 * num_pts))
                        sorted_dists = np.sort(planar_dists)
                        closest_5pct_dists = sorted_dists[:num_closest]
                        closest_median_dist = np.median(closest_5pct_dists)
                        
                        objects_3d_info[i] = {
                            'median_3d': median_3d,
                            'closest_median_dist': closest_median_dist,
                            'u_proj': u_proj
                        }
                        
                        if len(pts_3d_base) > max_pts_to_disp:
                            indices = np.random.choice(len(pts_3d_base), max_pts_to_disp, replace=False)
                            disp_pts = pts_3d_base[indices]
                        else:
                            disp_pts = pts_3d_base
                            
                        rr.log(f"SegmentedPeople/{c_name}/person_{i}_pts", 
                                rr.Points3D(disp_pts, colors=[color], radii=[0.005]))
                                
                        sphere_radius = config.SEGMENTED_REGION_SPHERE_RADIUS_M
                        rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", 
                                rr.Points3D([median_3d], colors=[color], radii=[sphere_radius]))
                        rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", 
                                rr.Points3D([median_3d], colors=[color], radii=[sphere_radius]))
                    else:
                        rr.log(f"SegmentedPeople/{c_name}/person_{i}_pts", rr.Clear(recursive=True))
                        rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                        rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                        
        return objects_3d_info, orig_w, orig_h

