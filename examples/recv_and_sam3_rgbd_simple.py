#!/usr/bin/env python3
import argparse
import time
import zmq
import sys
import os
import cv2
import numpy as np

import rerun as rr
import rerun.blueprint as rrb

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stretch4_emulated_rgbd.shared_utils import RGBDFrame
    from stretch4_emulated_rgbd.api import visualize_rgbd_frame, ValidityMaskManager, unproject_points
    from stretch4_emulated_rgbd import rgbd_networking as gn
except ImportError:
    print("Error: stretch4_emulated_rgbd is not installed or not in python path.")
    sys.exit(1)

from stretch4_human_pose_estimation.sam3_body_segmentation import SAM3Pipeline, ContinuousSAM3VideoPipeline

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

def main():
    parser = argparse.ArgumentParser(description="Receive Emulated RGB-D stream and Joint States over PyZMQ with SAM 3.1 Segmentation.")
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure rgbd_networking.py first.')
    parser.add_argument('--disable-rate-print', action='store_true', help='Disable printing of the receiving rate and dropped messages.')
    parser.add_argument('--prompt', type=str, default='people', help="Text prompt for SAM 3.1 segmentation (default: people)")
    parser.add_argument('--tracking', action='store_true', help='Use SAM 3.1 continuous video tracking instead of per-frame segmentation.')
    args = parser.parse_args()
    
    print(f"Initializing SAM 3.1 Pipeline (Prompt: {args.prompt}, Tracking: {args.tracking})...")
    try:
        if args.tracking:
            pipeline = ContinuousSAM3VideoPipeline(prompt=args.prompt)
            inference_states = {}
            frames_received_per_cam = {}
        else:
            pipeline = SAM3Pipeline(prompt=args.prompt)
            inference_states = None
    except Exception as e:
        print(f"Error initializing SAM 3.1 pipeline: {e}")
        return
        
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    if args.remote:
        address = f"tcp://{gn.robot_ip}:{gn.rgbd_and_joints_port}"
    else:
        address = f"tcp://127.0.0.1:{gn.rgbd_and_joints_port}"
        
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)
    
    print("Initializing ReRun...")
    rr.init("Stretch SAM 3.1 Emulated RGBD Live", spawn=False)
    rr.spawn(memory_limit="2GiB")

    camera_views = [
        rrb.Spatial2DView(name="Left Camera", origin="Cameras/left"),
        rrb.Spatial2DView(name="Right Camera", origin="Cameras/right"),
        rrb.Spatial2DView(name="Center Camera", origin="Cameras/center")
    ]
    
    timeseries_views = [
        rrb.TimeSeriesView(name="Lift & Arm", origin="Telemetry/LiftArm"),
        rrb.TimeSeriesView(name="Wrist", origin="Telemetry/Wrist"),
    ]

    view_layout = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ Pointclouds/base_frame/**"]),
            rrb.Spatial3DView(name="Segmented People", origin="/", contents=["+ SegmentedPeople/**"])
        ),
        rrb.Vertical(
            rrb.Horizontal(*camera_views),
            rrb.Horizontal(*timeseries_views)
        ),
        column_shares=[2, 3]
    )

    blueprint = rrb.Blueprint(
        view_layout,
        rrb.BlueprintPanel(expanded=False),
        rrb.TimePanel(play_state="following"),
    )
    rr.send_blueprint(blueprint)

    mask_manager = None
    PREV_NUM_PEOPLE_CACHE = {}

    print("Receiving stream... Press 'Ctrl+C' to exit.")
    
    last_print_time = time.time()
    frames_received = 0
    last_seq_num = None
    dropped_messages = 0
    
    try:
        while True:
            output_dict = socket.recv_pyobj()
            frame = RGBDFrame.from_dict(output_dict)
            
            if mask_manager is None:
                robot_id = output_dict.get('robot_id')
                if robot_id and robot_id != 'unknown':
                    fleet_path = os.environ.get("HELLO_FLEET_PATH", os.path.expanduser('~/stretch_user'))
                    masks_dir = os.path.join(fleet_path, robot_id, "calibration_cameras")
                    mask_manager = ValidityMaskManager(masks_dir=masks_dir)
                else:
                    mask_manager = ValidityMaskManager()
                    
            closest_joint_state = output_dict.get('closest_joint_state', None)
            
            frames_received += 1
            img_seq = frame.image_frame.frame_number
            if img_seq is not None:
                if last_seq_num is not None:
                    dropped = img_seq - last_seq_num - 1
                    if dropped > 0:
                        dropped_messages += dropped
                last_seq_num = img_seq
                
            # Log Joint States
            if closest_joint_state is not None:
                rr.set_time("timestamp", timestamp=closest_joint_state['monotonic_timestamp'])
                
                rr.log("Telemetry/LiftArm/Lift", rr.Scalars(closest_joint_state['lift']['height']))
                rr.log("Telemetry/LiftArm/Arm", rr.Scalars(closest_joint_state['arm']['extension']))
                rr.log("Telemetry/LiftArm/Gripper", rr.Scalars(closest_joint_state['gripper']['pos_pct']))
                
                rr.log("Telemetry/Wrist/Yaw", rr.Scalars(closest_joint_state['wrist_yaw']['angle']))
                rr.log("Telemetry/Wrist/Pitch", rr.Scalars(closest_joint_state['wrist_pitch']['angle']))
                rr.log("Telemetry/Wrist/Roll", rr.Scalars(closest_joint_state['wrist_roll']['angle']))

            c_name = frame.camera_type
            lidar_str = frame.lidars_used if frame.lidars_used else "no_lidar"
            
            # --- SAM 3.1 Segmentation ---
            image_bgr = frame.image.copy()
            orig_h, orig_w = image_bgr.shape[:2]
            
            # Unrotate if left/right to pass upright image to SAM
            if c_name == "left":
                image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif c_name == "right":
                image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
                
            if args.tracking:
                if c_name not in inference_states:
                    inference_states[c_name] = pipeline.init_state()
                    frames_received_per_cam[c_name] = 0
                    
                frame_idx = frames_received_per_cam[c_name]
                inf_state = inference_states[c_name]
                
                pipeline.add_frame_to_state(inf_state, frame_idx, image_bgr)
                if frame_idx == 0:
                    pipeline.add_text_prompt(inf_state, frame_idx, obj_id=1, text=args.prompt)
                results = pipeline.track_step(inf_state, frame_idx)
                
                frames_received_per_cam[c_name] += 1
            else:
                results = pipeline.predict(image_bgr)
            
            # Clears old rerun annotations
            prev_num = PREV_NUM_PEOPLE_CACHE.get(c_name, 0)
            curr_num = len(results)
            for i in range(curr_num, prev_num):
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_pts", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}", rr.Clear(recursive=True))
            PREV_NUM_PEOPLE_CACHE[c_name] = curr_num

            # Log RGBD frames to rerun
            vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, frame.image.shape)
            visualize_rgbd_frame(c_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)

            # 3D Variables
            camera_matrix = frame.camera_matrix
            dist_coeffs = frame.distortion_coefficients
            
            T_cam_to_base = None
            if frame.T_base_to_cam is not None:
                T_cam_to_base = np.linalg.inv(frame.T_base_to_cam)
            else:
                T_cam_to_base = np.eye(4)

            for i, res in enumerate(results):
                color = COLORS[i % len(COLORS)]
                
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                
                # 1. Remap and visualize 2D boxes
                box = res.get("box", None)
                if box is not None:
                    # Bounding boxes might be tricky to remap simply via corners if rotated by 90 degrees.
                    # A rotated bounding box might require remapping min/max.
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
                        
                        # Create full resolution binary mask of the segmented person in unrotated coordinates
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
                            
                            # Compute median before subsampling
                            median_pt = np.median(pts_3d_base, axis=0)
                            
                            # Subsample if too many points to avoid slowing down rerun
                            if len(pts_3d_base) > 5000:
                                indices = np.random.choice(len(pts_3d_base), 5000, replace=False)
                                pts_3d_base = pts_3d_base[indices]
                                
                            rr.log(f"SegmentedPeople/{c_name}/person_{i}_pts", 
                                   rr.Points3D(pts_3d_base, colors=[color], radii=[0.005]))
                                   
                            sphere_radius = 0.05
                            rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", 
                                   rr.Points3D([median_pt], colors=[color], radii=[sphere_radius]))
                            rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", 
                                   rr.Points3D([median_pt], colors=[color], radii=[sphere_radius]))
                        else:
                            rr.log(f"SegmentedPeople/{c_name}/person_{i}_pts", rr.Clear(recursive=True))
                            rr.log(f"SegmentedPeople/{c_name}/person_{i}_center", rr.Clear(recursive=True))
                            rr.log(f"Pointclouds/base_frame/{c_name}/person_{i}_center", rr.Clear(recursive=True))
            
            # Print stats
            current_time = time.time()
            elapsed = current_time - last_print_time
            if elapsed >= 1.0:
                if not args.disable_rate_print:
                    hz = frames_received / elapsed
                    print(f"Rate: {hz:.2f} Hz | Estimated dropped messages in last {elapsed:.1f}s: {dropped_messages}")
                frames_received = 0
                dropped_messages = 0
                last_print_time = current_time
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped receiving.")

if __name__ == "__main__":
    main()
