#!/usr/bin/env python3
"""
Minimized RGB-D RTMO pose estimation script.

This script performs a minimal loop:
1. Receives an RGB-D image
2. Rotates the RGB component to be upright for RTMO
3. Uses RTMO to detect humans and estimate pose skeletons in the upright image
4. Transforms RTMO results back to native RGB-D orientation
5. Creates a dense depth version of the sparse depth image
6. Creates a 3D version of the RTMO pose skeleton by looking up dense depth at keypoints
7. Visualizes the RTMO results overlaid on the RGB-D image in its native orientation
8. Visualizes the 3D RTMO results in the RGB-D point cloud
"""

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
    print("Error: stretch4_emulated_rgbd is not installed.")
    import sys
    sys.exit(1)
from stretch4_human_pose_estimation import (
    RTMOPipeline,
    CVPR_KEYPOINT_COLORS_RGB,
    CVPR_EDGE_COLORS_RGB
)

# Definition of the skeletal edges according to COCO 17-keypoint format used by RTMO
RTMO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]

def rotate_to_upright(image_bgr, c_name):
    """
    Rotates the camera image to an upright orientation based on its name.
    The Stretch cameras (left and right) are mounted rotated.
    """
    if c_name == "left":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif c_name == "right":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr

def remap_coordinate(u, v, c_name, orig_w, orig_h):
    """
    Remaps a 2D pixel coordinate from the upright orientation back to the native camera orientation.
    """
    if c_name == "left":
        return orig_w - 1.0 - v, u
    elif c_name == "right":
        return v, orig_h - 1.0 - u
    return u, v

def process_and_visualize(c_name: str, frame: RGBDFrame, pipeline: RTMOPipeline, kpt_thr: float = 0.3):
    """
    Main loop iteration processing an RGBDFrame.
    """
    rr.set_time("timestamp", timestamp=frame.timestamp)
    
    # 1. Receive an RGB-D image
    image_bgr = frame.image.copy()
    orig_h, orig_w = image_bgr.shape[:2]
    
    # 2. Rotate the RGB image component of the RGB-D image to be upright
    upright_image = rotate_to_upright(image_bgr, c_name)
    
    # 3. Use RTMO to detect humans and estimate their pose skeletons in the upright RGB image
    results = pipeline.predict(upright_image)
    
    # 4. Transform the RTMO results back to the native RGB-D orientation
    native_results = []
    for res in results:
        kpts_rot = np.array(res["keypoints"])
        kpts_orig = np.empty_like(kpts_rot)
        for i, (u, v, conf) in enumerate(kpts_rot):
            orig_u, orig_v = remap_coordinate(u, v, c_name, orig_w, orig_h)
            kpts_orig[i] = [orig_u, orig_v, conf]
        native_results.append({
            "keypoints": kpts_orig,
            "box": res.get("box", None)
        })

    # 5. Create a dense depth version of the sparse depth image component of the RGB-D image
    dense_depth = None
    if frame.depth_image is not None and frame.depth_image.shape[0] > 0:
        dense_processor = DenseDepthImage(
            frame.image, 
            frame.depth_image, 
            apply_validity_mask=True, 
            camera_name=c_name, 
            lidar_name=getattr(frame, "lidars_used", "both_lidar")
        )
        dense_processor.compute_dense_depth()
        dense_depth = dense_processor.dense_depth_image

    # Prepare for 3D projection
    camera_matrix = frame.camera_matrix
    dist_coeffs = frame.distortion_coefficients
    T_base_to_cam = frame.T_base_to_cam
    T_cam_to_base = np.linalg.inv(T_base_to_cam) if T_base_to_cam is not None else np.eye(4)

    # 7. Visualize the RTMO results on an RGB-D image visualization
    # We log everything in its upright orientation to Cameras/{c_name} to leverage ReRun's natural overlaying
    rr.log(f"Cameras/{c_name}/rgb", rr.Image(upright_image, color_model="BGR").compress())
    
    # Semi-transparent overlays for dense and sparse depth (rotated upright)
    if dense_depth is not None:
        upright_dense_depth = rotate_to_upright(dense_depth, c_name)
        rr.log(f"Cameras/{c_name}/dense_depth", rr.DepthImage(upright_dense_depth, meter=1.0))
    if frame.depth_image is not None:
        upright_depth = rotate_to_upright(frame.depth_image, c_name)
        rr.log(f"Cameras/{c_name}/sparse_depth", rr.DepthImage(upright_depth, meter=1.0))

    # Clear previously logged skeletons to avoid ghosting (we clear the whole parent entity)
    rr.log(f"Cameras/{c_name}/skeletal_overlay", rr.Clear(recursive=True))
    rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons", rr.Clear(recursive=True))

    # 8. Visualize the 3D RTMO results on the RGB-D point cloud
    if len(frame.point_cloud_base) > 0:
        rr.log(
            f"Pointclouds/base_frame/{c_name}",
            rr.Points3D(frame.point_cloud_base, colors=frame.point_colors, radii=[0.0025]),
        )

    for i, res in enumerate(results):
        kpts_2d_upright = res["keypoints"]
        kpts_2d_native = native_results[i]["keypoints"]
        
        valid_kpts_2d_upright = []
        valid_colors_2d = []
        valid_3d_indices = []
        valid_u = []
        valid_v = []
        valid_z = []
        
        # 6. Create a 3D version of the RTMO pose skeleton by using the dense depth
        for k, (u_native, v_native, conf) in enumerate(kpts_2d_native):
            if conf > kpt_thr:
                valid_kpts_2d_upright.append(kpts_2d_upright[k][:2])
                
                # Check depth at the keypoint location in the native frame
                u_int, v_int = int(round(u_native)), int(round(v_native))
                if dense_depth is not None and 0 <= u_int < orig_w and 0 <= v_int < orig_h:
                    z = dense_depth[v_int, u_int]
                    # If valid depth is at the location, store for unprojection
                    if z > 0 and z != np.inf and not np.isnan(z):
                        valid_3d_indices.append(k)
                        valid_u.append(u_native)
                        valid_v.append(v_native)
                        valid_z.append(z)
                        
                # 2D Colors
                if k < len(CVPR_KEYPOINT_COLORS_RGB):
                    valid_colors_2d.append(CVPR_KEYPOINT_COLORS_RGB[k])
                else:
                    valid_colors_2d.append([0, 255, 0])
                        
        # Visualize the 2D keypoints and skeleton lines
        if valid_kpts_2d_upright:
            rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/keypoints", rr.Points2D(valid_kpts_2d_upright, colors=valid_colors_2d, radii=3.0))
            
        lines_2d = []
        lines_2d_colors = []
        for edge_idx, (p1, p2) in enumerate(RTMO_EDGES):
            if p1 < len(kpts_2d_upright) and p2 < len(kpts_2d_upright):
                if kpts_2d_upright[p1][2] > kpt_thr and kpts_2d_upright[p2][2] > kpt_thr:
                    lines_2d.append([kpts_2d_upright[p1][:2], kpts_2d_upright[p2][:2]])
                    if edge_idx < len(CVPR_EDGE_COLORS_RGB):
                        lines_2d_colors.append(CVPR_EDGE_COLORS_RGB[edge_idx])
                    else:
                        lines_2d_colors.append([0, 255, 0])
        if lines_2d:
            rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/skeleton", rr.LineStrips2D(lines_2d, colors=lines_2d_colors))
            
        # Create and visualize the 3D skeleton points and lines
        kpts_3d_base_dict = {}
        if valid_3d_indices and camera_matrix is not None:
            # Unproject the valid points to 3D camera coordinates
            pts_3d_cam = unproject_points(
                np.stack([valid_u, valid_v], axis=-1), 
                np.array(valid_z), 
                camera_matrix, 
                dist_coeffs, 
                camera_model="fisheye"
            )
            
            pts_base_list = []
            pts_base_colors = []
            for idx, pt_cam in zip(valid_3d_indices, pts_3d_cam):
                # Transform 3D point to base frame
                p_base = (T_cam_to_base @ np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0]))[:3]
                kpts_3d_base_dict[idx] = p_base
                pts_base_list.append(p_base)
                if idx < len(CVPR_KEYPOINT_COLORS_RGB):
                    pts_base_colors.append(CVPR_KEYPOINT_COLORS_RGB[idx])
                else:
                    pts_base_colors.append([0, 255, 0])
                
            # If a 3D keypoint is undefined, it is not in pts_base_list and won't be displayed
            if pts_base_list:
                rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}_pts", 
                       rr.Points3D(pts_base_list, colors=pts_base_colors, radii=[0.015]))
                       
            lines_base = []
            lines_base_colors = []
            for edge_idx, (p1, p2) in enumerate(RTMO_EDGES):
                if p1 in kpts_3d_base_dict and p2 in kpts_3d_base_dict:
                    lines_base.append([kpts_3d_base_dict[p1], kpts_3d_base_dict[p2]])
                    if edge_idx < len(CVPR_EDGE_COLORS_RGB):
                        lines_base_colors.append(CVPR_EDGE_COLORS_RGB[edge_idx])
                    else:
                        lines_base_colors.append([0, 255, 0])
                        
            if lines_base:
                rr.log(f"Pointclouds/base_frame/{c_name}/Skeletons/person_{i}", 
                       rr.LineStrips3D(lines_base, colors=lines_base_colors, radii=[0.005]))

def _parse_args():
    from stretch4_emulated_rgbd.shared_utils import get_arg_parser
    parser = get_arg_parser("Minimized version of RTMO pose estimation with RGBD streams.")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l'], default='m', help="Size of the RTMO model to run (default: m)")
    parser.add_argument("--device", type=str, choices=["AUTO", "CPU", "NPU", "GPU"], default="AUTO", help="Inference device (default: AUTO)")

    return parser.parse_args()

def main():
    args = _parse_args()
    
    # Initialize RTMO pipeline
    print(f"Initializing RTMO Pipeline (Size: {args.size}, Device: {args.device})...")
    try:
        pipeline = RTMOPipeline(size=args.size, device=args.device)
    except Exception as e:
        print(f"Error initializing RTMO pipeline: {e}")
        return

    # Initialize Rerun
    print("Initializing RGBD Streamer with Lidars...")
    rr.init("Stretch RGBD RTMO Minimal", spawn=False)
    rr.spawn(memory_limit="2GiB")

    # Load the blueprint
    blueprint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_rgbd_rtmo_pose_estimation.rbl")
    if os.path.exists(blueprint_path):
        print(f"Loading ReRun blueprint from {blueprint_path}")
        rr.log_file_from_path(blueprint_path)
    else:
        print("Blueprint file not found. Falling back to default layout.")
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ Pointclouds/base_frame/**"]),
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
    
    use_left = args.camera == "left"
    use_right = args.camera == "right"
    use_center = args.camera == "center"
    use_left_right = args.camera == "left_right"
    use_left_right_center = args.camera == "all"
    use_both_lidars_default = args.lidar == "both"
    use_left_lidar = args.lidar == "left" or use_both_lidars_default
    use_right_lidar = args.lidar == "right" or use_both_lidars_default
    
    calibration = None
    if hasattr(args, 'opt_yaml') and args.opt_yaml:
        from stretch4_emulated_rgbd.shared_utils import ExtrinsicsCalibration
        calibration = ExtrinsicsCalibration.load_from_yaml(args.opt_yaml)
        if calibration is None:
            return

    try:
        # Request stream from stretch4_emulated_rgbd
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
        
        # Main processing loop
        for frame_data in generator:
            if args.show_fps:
                loop_timer.end_of_iteration()
                loop_timer.pretty_print(minimum=True)
                loop_timer.start_of_iteration()
                
            if frame_data is None: 
                continue
            
            # Extract frames based on which streams are active
            if hasattr(frame_data, "left") or hasattr(frame_data, "right") or hasattr(frame_data, "center"):
                if hasattr(frame_data, "left") and frame_data.left:
                    process_and_visualize("left", frame_data.left, pipeline)
                if hasattr(frame_data, "right") and frame_data.right:
                    process_and_visualize("right", frame_data.right, pipeline)
                if hasattr(frame_data, "center") and frame_data.center:
                    process_and_visualize("center", frame_data.center, pipeline)
            else:
                process_and_visualize(frame_data.camera_type, frame_data, pipeline)

    except KeyboardInterrupt:
        print("Stopping...")
        os._exit(0)
    finally:
        if 'streamer' in locals() and streamer is not None:
            streamer.stop()

if __name__ == "__main__":
    main()
