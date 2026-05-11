#!/usr/env/bin python3
import argparse
import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from stretch_body_ii.core.hello_utils import LoopTimer
try:
    from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream, DenseDepthImage, unproject_points
    from stretch4_emulated_rgbd.shared_utils import RGBDFrame
except ImportError:
    print("Error: stretch4_emulated_rgbd is not installed. Please install it to use optimized RGB-D streams.")
    import sys
    sys.exit(1)

from stretch4_human_pose_estimation import SAM3Pipeline

# --- CONSTANTS ---
T_CAM_TO_BASE_CACHE = {}
PREV_NUM_PEOPLE_CACHE = {}

# --- UTILITIES ---
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

class PersonCenter3D:
    def __init__(self, box):
        self.box = box
        self.upright_center_2d = self._compute_2d_center(box)
        self.original_center_2d = None
        self.depth = None
        self.center_3d_cam = None
        self.center_3d_base = None

    def _compute_2d_center(self, box):
        xmin, ymin, xmax, ymax = box[:4]
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    def map_to_original_frame(self, c_name, orig_w, orig_h):
        if self.upright_center_2d is None:
            return None
        c_u, c_v = self.upright_center_2d
        self.original_center_2d = remap_coordinate(c_u, c_v, c_name, orig_w, orig_h)
        return self.original_center_2d

    def compute_3d_center(self, depth_image, camera_matrix, dist_coeffs, T_cam_to_base):
        if self.original_center_2d is None:
            return None
        c_u, c_v = self.original_center_2d
        
        # Estimate depth near the center
        u_int, v_int = int(round(c_u)), int(round(c_v))
        h, w = depth_image.shape
        radius = 10
        min_u = max(0, u_int - radius)
        max_u = min(w, u_int + radius + 1)
        min_v = max(0, v_int - radius)
        max_v = min(h, v_int + radius + 1)
        patch = depth_image[min_v:max_v, min_u:max_u]
        valid_mask = (patch > 0) & (patch != np.inf) & (~np.isnan(patch))
        valid_depths = patch[valid_mask]
        
        if len(valid_depths) > 0:
            self.depth = np.median(valid_depths)
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

def _log_2d_overlay(i, res, c_name):
    box = res.get("box", None)
    if box is not None:
        xmin, ymin, xmax, ymax = box[:4]
        center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        half_sizes = [(xmax - xmin) / 2, (ymax - ymin) / 2]
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/bbox", rr.Boxes2D(centers=[center], half_sizes=[half_sizes], colors=[[255, 0, 255]]))
        
    mask = res.get("mask", [])
    if len(mask) > 0:
        pts = np.array(mask, np.int32).reshape((-1, 2))
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/mask", rr.LineStrips2D([pts], colors=[[0, 255, 100]]))

    if "person_center_obj" in res and res["person_center_obj"].upright_center_2d is not None:
        c_u, c_v = res["person_center_obj"].upright_center_2d
        rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/center", rr.Points2D([c_u, c_v], colors=[[0, 255, 255]], radii=5.0))

def _remap_sam3_results(results, c_name, orig_w, orig_h):
    mapped_results = []
    for res in results:
        mapped_res = res.copy()
        
        # Remap polygon mask
        mask = np.array(res.get("mask", []))
        if len(mask) > 0:
            mask_orig = np.empty_like(mask)
            pts = mask.reshape(-1, 2)
            pts_orig = np.empty_like(pts)
            for i, (u, v) in enumerate(pts):
                orig_u, orig_v = remap_coordinate(u, v, c_name, orig_w, orig_h)
                pts_orig[i] = [orig_u, orig_v]
            mapped_res["mask"] = pts_orig.flatten().tolist()
            
        if "person_center_obj" in res:
            person_center = res["person_center_obj"]
            person_center.map_to_original_frame(c_name, orig_w, orig_h)
            mapped_res["person_center_obj"] = person_center
            
        mapped_results.append(mapped_res)
    return mapped_results

def _log_3d_pointcloud(i, res, c_name, frame, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
    mask = res.get("mask", [])
    
    if "person_center_obj" in res:
        person_center = res["person_center_obj"]
        person_center.compute_3d_center(frame.depth_image, camera_matrix, dist_coeffs, T_cam_to_base)
        person_center.log_to_rerun(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_center")
    else:
        person_center = None
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_center", rr.Clear(recursive=True))
        
    if len(mask) == 0:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", rr.Clear(recursive=True))
        return
        
    pts = np.array(mask, np.int32).reshape((-1, 1, 2))
    
    # Create a full resolution binary mask
    h, w = frame.depth_image.shape
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [pts], 1)
    
    # Get depth values for the mask
    depth_image = frame.depth_image
    valid_depth_mask = (depth_image > 0) & (depth_image != np.inf) & (~np.isnan(depth_image)) & (binary_mask == 1)
    
    if person_center is not None and person_center.depth is not None:
        valid_depth_mask &= (depth_image >= max(0.1, person_center.depth - 0.4))
        valid_depth_mask &= (depth_image <= person_center.depth + 0.4)
        
    v_vals, u_vals = np.where(valid_depth_mask)
    z_vals = depth_image[valid_depth_mask]
    
    pts_3d_cam = project_points_to_3d(u_vals, v_vals, z_vals, camera_matrix, dist_coeffs)
    
    if len(pts_3d_cam) > 0:
        pts_3d_base = np.dot(T_cam_to_base[:3, :3], pts_3d_cam.T).T + T_cam_to_base[:3, 3]
        
        # Subsample if too many points to avoid slowing down rerun
        if len(pts_3d_base) > 2000:
            indices = np.random.choice(len(pts_3d_base), 2000, replace=False)
            pts_3d_base = pts_3d_base[indices]
            
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", 
               rr.Points3D(pts_3d_base, colors=[[0, 255, 100]], radii=[0.005]))
    else:
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", rr.Clear(recursive=True))

def render_rgbd_sam3(c_name: str, frame: RGBDFrame, pipeline: SAM3Pipeline):
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
        box = res["box"]
        person_center = PersonCenter3D(box)
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
        mapped_results = _remap_sam3_results(results, c_name, orig_w, orig_h)

    # === LOGGING ALL DATA IN QUICK SUCCESSION ===
    
    # Clears
    prev_num = PREV_NUM_PEOPLE_CACHE.get(c_name, 0)
    curr_num = len(results)
    for i in range(curr_num, prev_num):
        rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", rr.Clear(recursive=True))
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

    # To visualize masks as translucent colored areas we can create an annotation context.
    # However, since polygon overlaps can happen and to keep it simple and analogous, 
    # we just log the image.
    rr.log(f"Cameras/{c_name}/rgb", rr.Image(image_bgr, color_model="BGR").compress())

    for i, res in enumerate(results):
        _log_2d_overlay(i, res, c_name)
                
    # 3D Data
    if len(frame.point_cloud_base) > 0:
        rr.log(
            f"Pointclouds/base_frame/{c_name}",
            rr.Points3D(frame.point_cloud_base, colors=frame.point_colors, radii=[0.0025]),
        )
        
    if mapped_results is not None:
        for i, res in enumerate(mapped_results):
            _log_3d_pointcloud(i, res, c_name, frame, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h)

def _parse_args():
    from stretch4_emulated_rgbd.shared_utils import get_arg_parser
    parser = get_arg_parser("Visualize colored point clouds and 3D SAM 3.1 segmentations from Stretch lidars and cameras in rerun.")
    parser.add_argument("--use_ros_for_lidars", action="store_true", help="Use ros2 to subscribe to lidar points. (Default: False)")
    parser.add_argument("--prompt", type=str, default='people', help="Text prompt for SAM 3.1 segmentation (default: people)")

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

    print(f"Initializing SAM 3.1 Pipeline (Prompt: {args.prompt})...")
    try:
        pipeline = SAM3Pipeline(prompt=args.prompt)
    except Exception as e:
        print(f"Error initializing SAM 3.1 pipeline: {e}")
        return

    print("Initializing RGBD Streamer with Lidars...")
    rr.init("Stretch RGBD SAM 3.1 Show", spawn=False)
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
            
            if hasattr(frame_data, "left") or hasattr(frame_data, "right") or hasattr(frame_data, "center"):
                if hasattr(frame_data, "left") and frame_data.left: render_rgbd_sam3("left", frame_data.left, pipeline)
                if hasattr(frame_data, "right") and frame_data.right: render_rgbd_sam3("right", frame_data.right, pipeline)
                if hasattr(frame_data, "center") and frame_data.center: render_rgbd_sam3("center", frame_data.center, pipeline)
            else:
                render_rgbd_sam3(frame_data.camera_type, frame_data, pipeline)


    except KeyboardInterrupt:
        print("Stopping... (Force quitting due to background threads)")
        os._exit(0)
    finally:
        if 'streamer' in locals() and streamer is not None:
            streamer.stop()

if __name__ == "__main__":
    main()
