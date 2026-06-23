#!/usr/env/bin python3
import argparse
import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from stretch4_body.core.hello_utils import LoopTimer
try:
    from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream, DenseDepthImage, unproject_points
    from stretch4_emulated_rgbd.shared_utils import RGBDFrame
except ImportError:
    print("Error: stretch4_emulated_rgbd is not installed. Please install it to use optimized RGB-D streams.")
    import sys
    sys.exit(1)
from stretch4_human_pose_estimation import (
    RTMOPipeline,
    CVPR_KEYPOINT_COLORS_RGB,
    CVPR_EDGE_COLORS_RGB
)

# --- CONSTANTS ---
T_CAM_TO_BASE_CACHE = {}
PREV_NUM_PEOPLE_CACHE = {}

HEAD_INDICES = [0, 1, 2, 3, 4]
TORSO_INDICES = [5, 6, 11, 12]
RTMO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]

# --- UTILITIES ---
def remap_coordinate(u, v, c_name, orig_w, orig_h):
    """Remaps a 2D coordinate from the rotated inference frame back to the native camera frame."""
    if c_name == "left":
        return orig_w - 1.0 - v, u
    elif c_name == "right":
        return v, orig_h - 1.0 - u
    return u, v

def extract_valid_depth_patch(depth_image, u, v, radius, min_depth=None, max_depth=None):
    """Safely extracts a valid depth patch around a pixel coordinate."""
    h, w = depth_image.shape
    u_int, v_int = int(round(u)), int(round(v))
    
    min_u = max(0, u_int - radius)
    max_u = min(w, u_int + radius + 1)
    min_v = max(0, v_int - radius)
    max_v = min(h, v_int + radius + 1)
    
    patch = depth_image[min_v:max_v, min_u:max_u]
    valid_mask = (patch > 0) & (patch != np.inf) & (~np.isnan(patch))
    
    if min_depth is not None:
        valid_mask &= (patch >= min_depth)
    if max_depth is not None:
        valid_mask &= (patch <= max_depth)
        
    return patch[valid_mask]

def estimate_person_depth_bounds(depth_image, kpts_2d, kpt_thr=0.3, margin=0.4):
    kp_median_depths = []
    
    for u, v, conf in kpts_2d:
        if conf > kpt_thr:
            valid_depths = extract_valid_depth_patch(depth_image, u, v, radius=10)
            if len(valid_depths) > 0:
                kp_median_depths.append(np.median(valid_depths))
                
    if not kp_median_depths:
        return None, None, None
        
    person_depth = np.median(kp_median_depths)
    min_depth = max(0.1, person_depth - margin)
    max_depth = person_depth + margin
    return min_depth, max_depth, person_depth

def estimate_keypoint_depth(depth_image, u, v, radius=10, min_depth=None, max_depth=None):
    valid_depths = extract_valid_depth_patch(depth_image, u, v, radius, min_depth, max_depth)
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    return None

def project_points_to_3d(u_vals, v_vals, z_vals, camera_matrix, dist_coeffs):
    if len(u_vals) == 0:
        return np.zeros((0, 3), dtype=np.float32)
        
    uv = np.stack([u_vals, v_vals], axis=-1)
    return unproject_points(uv, z_vals, camera_matrix, dist_coeffs, camera_model="fisheye")

def extract_head_pointcloud(depth_image, kpts_2d, kpts_3d_cam, camera_matrix, dist_coeffs, kpt_thr, person_center=None):
    valid_2d = []
    valid_3d = []
    
    min_depth = None
    max_depth = None
    if person_center is not None and person_center.depth is not None:
        min_depth = max(0.1, person_center.depth - 0.4)
        max_depth = person_center.depth + 0.4
        
    for idx in HEAD_INDICES:
        if idx < len(kpts_2d):
            u, v, conf = kpts_2d[idx]
            if conf > kpt_thr:
                if person_center is not None:
                    z = estimate_keypoint_depth(depth_image, u, v, radius=10, min_depth=min_depth, max_depth=max_depth)
                    if z is not None:
                        pt_3d = project_points_to_3d(np.array([u]), np.array([v]), np.array([z]), camera_matrix, dist_coeffs)[0]
                        kpts_3d_cam[idx] = pt_3d
                        valid_2d.append((u, v))
                        valid_3d.append(pt_3d)
                    elif idx in kpts_3d_cam:
                        del kpts_3d_cam[idx]
                else:
                    if idx in kpts_3d_cam:
                        valid_2d.append((u, v))
                        valid_3d.append(kpts_3d_cam[idx])
                
    if len(valid_2d) < 1:
        return None, None, None
        
    valid_2d = np.array(valid_2d)
    valid_3d = np.array(valid_3d)
    
    min_u = max(0, int(np.min(valid_2d[:, 0]) - 50))
    max_u = min(depth_image.shape[1], int(np.max(valid_2d[:, 0]) + 50))
    min_v = max(0, int(np.min(valid_2d[:, 1]) - 50))
    max_v = min(depth_image.shape[0], int(np.max(valid_2d[:, 1]) + 50))
    
    v_grid, u_grid = np.mgrid[min_v:max_v, min_u:max_u]
    u_vals = u_grid.flatten()
    v_vals = v_grid.flatten()
    
    patch = depth_image[min_v:max_v, min_u:max_u]
    z_vals = patch.flatten()
    
    valid_mask = (z_vals > 0) & (z_vals != np.inf) & (~np.isnan(z_vals))
    if min_depth is not None:
        valid_mask &= (z_vals >= min_depth)
    if max_depth is not None:
        valid_mask &= (z_vals <= max_depth)
        
    u_vals = u_vals[valid_mask]
    v_vals = v_vals[valid_mask]
    z_vals = z_vals[valid_mask]
    
    if len(z_vals) == 0:
        return None, None, None
        
    pts_3d = project_points_to_3d(u_vals, v_vals, z_vals, camera_matrix, dist_coeffs)
    
    center_3d = np.median(valid_3d, axis=0) #np.mean(valid_3d, axis=0)
    box_half_size = np.array([0.15, 0.15, 0.15])
    min_3d = center_3d - box_half_size
    max_3d = center_3d + box_half_size
    
    in_box_mask = (pts_3d[:, 0] > min_3d[0]) & (pts_3d[:, 0] < max_3d[0]) & \
                  (pts_3d[:, 1] > min_3d[1]) & (pts_3d[:, 1] < max_3d[1]) & \
                  (pts_3d[:, 2] > min_3d[2]) & (pts_3d[:, 2] < max_3d[2])
                  
    filtered_pts = pts_3d[in_box_mask]
    filtered_uv = np.stack((u_vals[in_box_mask], v_vals[in_box_mask]), axis=-1)
    return filtered_pts, valid_3d, filtered_uv


class PersonCenter3D:
    def __init__(self, kpts_upright, kpt_thr=0.3):
        self.kpts_upright = kpts_upright
        self.kpt_thr = kpt_thr
        self.upright_center_2d = self._compute_2d_center(kpts_upright)
        self.original_center_2d = None
        self.depth = None
        self.center_3d_cam = None
        self.center_3d_base = None

    def _compute_2d_center(self, kpts):
        valid_torso = [kpts[idx] for idx in TORSO_INDICES if idx < len(kpts) and kpts[idx][2] > self.kpt_thr]
        if len(valid_torso) > 0:
            return np.mean([pt[0] for pt in valid_torso]), np.mean([pt[1] for pt in valid_torso])
            
        valid_all = [pt for pt in kpts if pt[2] > self.kpt_thr]
        if len(valid_all) > 0:
            min_u, max_u = min([pt[0] for pt in valid_all]), max([pt[0] for pt in valid_all])
            min_v, max_v = min([pt[1] for pt in valid_all]), max([pt[1] for pt in valid_all])
            return (min_u + max_u) / 2.0, (min_v + max_v) / 2.0
            
        return None

    def map_to_original_frame(self, c_name, orig_w, orig_h):
        if self.upright_center_2d is None:
            return None
        c_u, c_v = self.upright_center_2d
        self.original_center_2d = remap_coordinate(c_u, c_v, c_name, orig_w, orig_h)
        return self.original_center_2d

    def compute_3d_center(self, depth_image, kpts_2d_orig, camera_matrix, dist_coeffs, T_cam_to_base):
        if self.original_center_2d is None:
            return None
        _, _, person_depth = estimate_person_depth_bounds(depth_image, kpts_2d_orig, self.kpt_thr, margin=0.4)
        self.depth = person_depth
        if self.depth is not None:
            c_u, c_v = self.original_center_2d
            pts_3d = project_points_to_3d([c_u], [c_v], [self.depth], camera_matrix, dist_coeffs)
            if len(pts_3d) > 0:
                self.center_3d_cam = tuple(pts_3d[0])
                self.center_3d_base = (T_cam_to_base @ np.array([self.center_3d_cam[0], self.center_3d_cam[1], self.center_3d_cam[2], 1.0]))[:3]
        return self.center_3d_base

    def log_to_rerun(self, entity_path):
        if self.center_3d_base is not None:
            rr.log(entity_path, rr.Points3D([self.center_3d_base], radii=[0.05], colors=[[255, 255, 0]]))
        else:
            rr.log(entity_path, rr.Clear(recursive=True))

# --- PIPELINE LOGIC ---

def _log_2d_overlay(i, res, c_name, kpt_thr, style):
    """Logs 2D overlays (bounding box, keypoints, skeleton) for the given person to ReRun."""
    box = res.get("box", None)
    if box is not None:
        xmin, ymin, xmax, ymax = box[:4]
        center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        half_sizes = [(xmax - xmin) / 2, (ymax - ymin) / 2]
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/bbox", rr.Boxes2D(centers=[center], half_sizes=[half_sizes], colors=[[255, 0, 255]]))
        
    kpts = res.get("keypoints", [])
    valid_kpts = []
    valid_colors = []
    for k, kp in enumerate(kpts):
        x, y, conf = kp
        if conf > kpt_thr:
            valid_kpts.append([x, y])
            if style == "cvpr" and k < len(CVPR_KEYPOINT_COLORS_RGB):
                valid_colors.append(CVPR_KEYPOINT_COLORS_RGB[k])
            else:
                valid_colors.append([255, 0, 0])
                
    if valid_kpts:
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/keypoints", rr.Points2D(valid_kpts, colors=valid_colors, radii=3.0))
        
    lines = []
    colors = []
    for edge_idx, (p1, p2) in enumerate(RTMO_EDGES):
        if p1 < len(kpts) and p2 < len(kpts):
            if kpts[p1][2] > kpt_thr and kpts[p2][2] > kpt_thr:
                lines.append([kpts[p1][:2], kpts[p2][:2]])
                if style == "cvpr" and edge_idx < len(CVPR_EDGE_COLORS_RGB):
                    colors.append(CVPR_EDGE_COLORS_RGB[edge_idx])
                else:
                    colors.append([0, 255, 0])
                    
    if lines:
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/skeleton", rr.LineStrips2D(lines, colors=colors))

    if "person_center_obj" in res and res["person_center_obj"].upright_center_2d is not None:
        c_u, c_v = res["person_center_obj"].upright_center_2d
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/center", rr.Points2D([c_u, c_v], colors=[[0, 255, 255]], radii=5.0))

def _remap_rtmo_results(results, c_name, orig_w, orig_h):
    """Remaps 2D RTMO results from the upright rotated image back to the native camera orientation."""
    mapped_results = []
    for res in results:
        mapped_res = res.copy()
        kpts_rot = np.array(res["keypoints"])
        
        kpts_orig = np.empty_like(kpts_rot)
        for i, (u, v, conf) in enumerate(kpts_rot):
            orig_u, orig_v = remap_coordinate(u, v, c_name, orig_w, orig_h)
            kpts_orig[i] = [orig_u, orig_v, conf]
            
        if "person_center_obj" in res:
            person_center = res["person_center_obj"]
            person_center.map_to_original_frame(c_name, orig_w, orig_h)
            mapped_res["person_center_obj"] = person_center
            
        mapped_res["keypoints"] = kpts_orig
        mapped_results.append(mapped_res)
    return mapped_results

def _log_3d_skeleton(i, res, c_name, frame, camera_matrix, dist_coeffs, T_cam_to_base, kpt_thr, style, orig_w, orig_h):
    """Estimates depth, projects to 3D, constructs the skeleton, extracts head PC, and logs all to ReRun."""
    kpts_2d = res["keypoints"]
    
    if "person_center_obj" in res:
        person_center = res["person_center_obj"]
        person_center.compute_3d_center(frame.depth_image, kpts_2d, camera_matrix, dist_coeffs, T_cam_to_base)
        person_center.log_to_rerun(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_center")
    else:
        person_center = None
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_center", rr.Clear(recursive=True))
        
    valid_kpts_idx = []
    valid_u = []
    valid_v = []
    valid_z = []
    
    for k, (u, v, conf) in enumerate(kpts_2d):
        if conf > kpt_thr:
            z = estimate_keypoint_depth(frame.depth_image, u, v)
            if z is not None:
                valid_kpts_idx.append(k)
                valid_u.append(u)
                valid_v.append(v)
                valid_z.append(z)
                    
    pts_3d_cam = project_points_to_3d(valid_u, valid_v, valid_z, camera_matrix, dist_coeffs)
    
    kpts_3d_cam_dict = {}
    for idx, pt_cam in zip(valid_kpts_idx, pts_3d_cam):
        kpts_3d_cam_dict[idx] = pt_cam
        
    head_pts, head_kpts_3d, head_uvs = extract_head_pointcloud(frame.depth_image, kpts_2d, kpts_3d_cam_dict, camera_matrix, dist_coeffs, kpt_thr, person_center)
    
    kpts_3d_base_dict = {}
    pts_base_list = []
    pts_colors = []
    
    for idx, pt_cam in kpts_3d_cam_dict.items():
        p_base = (T_cam_to_base @ np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0]))[:3]
        kpts_3d_base_dict[idx] = p_base
        pts_base_list.append(p_base)
        if style == "cvpr" and idx < len(CVPR_KEYPOINT_COLORS_RGB):
            pts_colors.append(CVPR_KEYPOINT_COLORS_RGB[idx])
        else:
            pts_colors.append([255, 0, 0])
                
    lines_base = []
    colors = []
    
    for edge_idx, (p1, p2) in enumerate(RTMO_EDGES):
        if p1 in kpts_3d_base_dict and p2 in kpts_3d_base_dict:
            lines_base.append([kpts_3d_base_dict[p1], kpts_3d_base_dict[p2]])
            if style == "cvpr" and edge_idx < len(CVPR_EDGE_COLORS_RGB):
                colors.append(CVPR_EDGE_COLORS_RGB[edge_idx])
            else:
                colors.append([0, 255, 0])
            
    if lines_base:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}", 
               rr.LineStrips3D(lines_base, colors=colors, radii=[0.005]))
    else:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}", rr.Clear(recursive=True))
        
    if pts_base_list:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", 
               rr.Points3D(pts_base_list, colors=pts_colors, radii=[0.015]))
    else:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", rr.Clear(recursive=True))
    if head_pts is not None:
        path = f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head"
        rr.log(path, rr.Transform3D(translation=T_cam_to_base[:3, 3], mat3x3=T_cam_to_base[:3, :3]))
        rr.log(f"{path}/points", rr.Points3D(head_pts, colors=[[255, 0, 255]], radii=[0.002]))
        
        if len(head_pts) > 0:
            head_center_cam = np.median(head_pts, axis=0)
            head_center_base = (T_cam_to_base @ np.array([head_center_cam[0], head_center_cam[1], head_center_cam[2], 1.0]))[:3]
            rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head_center", rr.Points3D([head_center_base], radii=[0.05], colors=[[255, 0, 255]]))
        else:
            rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head_center", rr.Clear(recursive=True))
            
        if head_uvs is not None:
            up_uvs = []
            for u_orig, v_orig in head_uvs:
                # head_uvs is in original frame. overlay is in upright frame.
                if c_name == "left":
                    up_u, up_v = v_orig, orig_w - 1.0 - u_orig
                elif c_name == "right":
                    up_u, up_v = orig_h - 1.0 - v_orig, u_orig
                else:
                    up_u, up_v = u_orig, v_orig
                up_uvs.append([up_u, up_v])
            if up_uvs:
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/head_pts", rr.Points2D(up_uvs, colors=[[255, 0, 255]], radii=2.0))
            else:
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/head_pts", rr.Clear(recursive=True))
        else:
            rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/head_pts", rr.Clear(recursive=True))
    else:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head", rr.Clear(recursive=True))
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head_center", rr.Clear(recursive=True))
        rr.log(f"Cameras/{c_name}/Overlay/person_{i}/head_pts", rr.Clear(recursive=True))

def render_rgbd_rtmo(c_name: str, frame: RGBDFrame, pipeline: RTMOPipeline, style="cvpr", kpt_thr=0.3):
    rr.set_time("timestamp", timestamp=frame.timestamp)
    image_bgr = frame.image_frame.image.copy()
    orig_h, orig_w = image_bgr.shape[:2]
    
    if c_name == "left":
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif c_name == "right":
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    
    # 1. Inference and base computations
    results = pipeline.predict(image_bgr)
    
    for i, res in enumerate(results):
        kpts_upright = res["keypoints"]
        person_center = PersonCenter3D(kpts_upright, kpt_thr)
        res["person_center_obj"] = person_center

    # 2. Dense depth computations
    depth_vis = None
    dense_processor = None
    if frame.depth_image is not None and frame.depth_image.shape[0] > 0:
        depth_vis = frame.depth_image
        if c_name == "left":
            depth_vis = cv2.rotate(depth_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif c_name == "right":
            depth_vis = cv2.rotate(depth_vis, cv2.ROTATE_90_CLOCKWISE)
            
        dense_processor = DenseDepthImage(
            frame.image_frame.image, 
            frame.depth_image, 
            apply_validity_mask=True, 
            camera_name=c_name, 
            lidar_name=getattr(frame, "lidars_used", "both_lidar")
        )
        dense_processor.compute_dense_depth()
        
    # 3. 3D mapping computations
    camera_matrix = frame.camera_matrix
    dist_coeffs = frame.distortion_coefficients
    
    mapped_results = None
    T_cam_to_base = None
    if camera_matrix is not None and frame.depth_image is not None:
        if c_name not in T_CAM_TO_BASE_CACHE:
            if frame.T_base_to_cam is not None:
                T_CAM_TO_BASE_CACHE[c_name] = np.linalg.inv(frame.T_base_to_cam)
            else:
                T_CAM_TO_BASE_CACHE[c_name] = np.eye(4)
        T_cam_to_base = T_CAM_TO_BASE_CACHE[c_name]
        mapped_results = _remap_rtmo_results(results, c_name, orig_w, orig_h)

    # === LOGGING ALL DATA IN QUICK SUCCESSION ===
    
    # Clears
    prev_num = PREV_NUM_PEOPLE_CACHE.get(c_name, 0)
    curr_num = len(results)
    for i in range(curr_num, prev_num):
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}", rr.Clear(recursive=True))
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", rr.Clear(recursive=True))
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head", rr.Clear(recursive=True))
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_head_center", rr.Clear(recursive=True))
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_center", rr.Clear(recursive=True))
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}", rr.Clear(recursive=True))
    PREV_NUM_PEOPLE_CACHE[c_name] = curr_num

    # 2D Data
    if dense_processor is not None and dense_processor.dense_depth_image is not None:
        dd = dense_processor.dense_depth_image.copy()
        if c_name == "left":
            dd = cv2.rotate(dd, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif c_name == "right":
            dd = cv2.rotate(dd, cv2.ROTATE_90_CLOCKWISE)
        rr.log(f"Cameras/{c_name}/dense_depth", rr.DepthImage(dd, meter=1.0))

    if depth_vis is not None:
        rr.log(f"Cameras/{c_name}/depth", rr.DepthImage(depth_vis, meter=1.0))

    rr.log(f"Cameras/{c_name}/rgb", rr.Image(image_bgr, color_model="BGR").compress())

    for i, res in enumerate(results):
        _log_2d_overlay(i, res, c_name, kpt_thr, style)
                
    # 3D Data
    if len(frame.point_cloud_base) > 0:
        rr.log(
            f"Pointclouds/base_frame/{c_name}",
            rr.Points3D(frame.point_cloud_base, colors=frame.point_colors, radii=[0.0025]),
        )
        
    if mapped_results is not None:
        for i, res in enumerate(mapped_results):
            _log_3d_skeleton(i, res, c_name, frame, camera_matrix, dist_coeffs, T_cam_to_base, kpt_thr, style, orig_w, orig_h)

def _parse_args():
    from stretch4_emulated_rgbd.shared_utils import get_arg_parser
    parser = get_arg_parser("Visualize colored point clouds and 3D RTMO human poses from Stretch lidars and cameras in rerun.")
    parser.add_argument("--use_ros_for_lidars", action="store_true", help="Use ros2 to subscribe to lidar points. (Default: False)")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l'], default='m', help="Size of the RTMO model to run (default: m)")
    parser.add_argument("--device", type=str, choices=["AUTO", "CPU", "NPU", "GPU"], default="AUTO", help="Inference device (default: AUTO)")
    parser.add_argument("--style", type=str, choices=["cvpr", "red_green"], default="cvpr", help="Visualization color style (default: cvpr)")

    return parser.parse_args()

def main():
    args = _parse_args()
    show_fps = args.show_fps
    use_left = args.left
    use_right = args.right
    use_center = args.center
    use_left_right = args.left_right
    use_left_right_center = args.left_right_center
    use_both_lidars_default = not (args.lidar_left or args.lidar_right)
    use_left_lidar = args.lidar_left or use_both_lidars_default
    use_right_lidar = args.lidar_right or use_both_lidars_default

    print(f"Initializing RTMO Pipeline (Size: {args.size}, Device: {args.device})...")
    try:
        pipeline = RTMOPipeline(size=args.size, device=args.device)
    except Exception as e:
        print(f"Error initializing RTMO pipeline: {e}")
        return

    print("Initializing RGBD Streamer with Lidars...")
    rr.init("Stretch RGBD RTMO Show", spawn=False)
    rr.spawn(memory_limit="2GiB")

    blueprint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rgbd_rtmo_pose_estimation.rbl")
    if os.path.exists(blueprint_path):
        print(f"Loading ReRun blueprint from {blueprint_path}")
        rr.log_file_from_path(blueprint_path)
    else:
        print("Blueprint file not found. Falling back to default layout.")
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ Pointclouds/base_frame/**"]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Left Camera", origin="Cameras/left"),
                    rrb.Spatial2DView(name="Center Camera", origin="Cameras/center"),
                    rrb.Spatial2DView(name="Right Camera", origin="Cameras/right"),
                ),
                row_shares=[3, 1]
            ),
            rrb.BlueprintPanel(expanded=True),
        )
        rr.send_blueprint(blueprint)

    print("Streaming started. Ctrl+C to exit.")

    loop_timer = LoopTimer()
    loop_timer.start_of_iteration()
    def print_loop_timer():
        if not show_fps:
            return
        loop_timer.end_of_iteration()
        loop_timer.pretty_print(minimum=True)
        loop_timer.start_of_iteration()
        
    calibration = None
    if hasattr(args, 'opt_yaml') and args.opt_yaml:
        from stretch4_emulated_rgbd.shared_utils import ExtrinsicsCalibration
        calibration = ExtrinsicsCalibration.load_from_yaml(args.opt_yaml)
        if calibration is None:
            return

    try:
        streamer, generator = get_emulated_rgbd_stream(
            use_left=use_left,
            use_right=use_right,
            use_center=use_center,
            use_left_right=use_left_right,
            use_left_right_center=use_left_right_center,
            use_left_lidar=use_left_lidar,
            use_right_lidar=use_right_lidar,
            emulated_rgbd_fps=args.emulated_rgbd_fps,
            camera_fps=args.camera_fps,
            resolution_height=args.resolution,
            compress=not args.disable_compression,
            oak_buffer_size=args.oak_buffer_size,
            calibration=calibration
        )
        
        for frame_data in generator:
            print_loop_timer()
            if frame_data is None: return
            
            # frame_data can be a SyncedRGBDFrame (has .left, .right, .center) or a single RGBDFrame
            if hasattr(frame_data, "left") or hasattr(frame_data, "right") or hasattr(frame_data, "center"):
                if hasattr(frame_data, "left") and frame_data.left: render_rgbd_rtmo("left", frame_data.left, pipeline, style=args.style)
                if hasattr(frame_data, "right") and frame_data.right: render_rgbd_rtmo("right", frame_data.right, pipeline, style=args.style)
                if hasattr(frame_data, "center") and frame_data.center: render_rgbd_rtmo("center", frame_data.center, pipeline, style=args.style)
            else:
                render_rgbd_rtmo(frame_data.camera_type, frame_data, pipeline, style=args.style)


    except KeyboardInterrupt:
        print("Stopping... (Force quitting due to background threads)")
        os._exit(0)
    finally:
        if 'streamer' in locals() and streamer is not None:
            streamer.stop()

if __name__ == "__main__":
    main()
