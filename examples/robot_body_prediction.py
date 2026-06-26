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

import pinocchio as pin

import robot_body_predictions_config as config

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stretch4_emulated_rgbd.shared_utils import RGBDFrame
    from stretch4_emulated_rgbd.api import visualize_rgbd_frame, ValidityMaskManager, project_base_link_points_to_image
    from stretch4_emulated_rgbd import rgbd_networking as gn
except ImportError:
    print("Error: stretch4_emulated_rgbd is not installed or not in python path.")
    sys.exit(1)

def main():
    """
    Main function to initialize ZMQ streaming, configure Rerun visualization, 
    and continuously predict and visualize the 3D positions of the robot's body links 
    using the Pinocchio kinematics library and a URDF model.
    """
    parser = argparse.ArgumentParser(description="Visualize robot body prediction using Pinocchio.")
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer.')
    args = parser.parse_args()
    
    # Set up ZeroMQ subscriber to receive RGBDFrame and joint state data
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
    sub_socket.setsockopt(zmq.RCVHWM, 1)
    sub_socket.setsockopt(zmq.CONFLATE, 1)
    
    if args.remote:
        sub_address = f"tcp://{gn.robot_ip}:{gn.rgbd_and_joints_port}"
    else:
        sub_address = f"tcp://127.0.0.1:{gn.rgbd_and_joints_port}"
        
    print(f"Connecting ZMQ Subscriber to {sub_address}")
    sub_socket.connect(sub_address)
    
    print("Initializing ReRun...")
    rr.init("Robot Body Prediction", spawn=False)
    rr.spawn(memory_limit="2GiB")

    # Load URDF model into Pinocchio to compute forward kinematics
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "urdfs", "calder_ik.urdf")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q = pin.neutral(model) # Initialize the joint configuration vector to neutral

    # Validate the configuration
    missing_links = []
    config_errors = []
    
    if not hasattr(config, "VISUALIZED_LINKS") or not isinstance(config.VISUALIZED_LINKS, dict):
        config_errors.append("VISUALIZED_LINKS dictionary is missing or invalid in config file.")
    else:
        for link_name, params in config.VISUALIZED_LINKS.items():
            if not model.existFrame(link_name):
                missing_links.append(link_name)
            if not isinstance(params, dict):
                config_errors.append(f"Parameters for link '{link_name}' must be a dictionary.")
                
    if missing_links or config_errors:
        error_msg = "Invalid configuration in robot_body_predictions_config.py:\n"
        if missing_links:
            error_msg += f"  - The following links were not found in the URDF: {', '.join(missing_links)}\n"
        if config_errors:
            error_msg += f"  - Errors: {'; '.join(config_errors)}\n"
        print(error_msg)
        sys.exit(1)

    # Define the visual layout for the Rerun viewer (3D viewer and camera views)
    view_layout = rrb.Horizontal(
        rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ BodyPredictions/**", "+ Pointclouds/base_frame/**"]),
        rrb.Vertical(
            rrb.Spatial2DView(name="Left Camera", origin="Cameras/left"),
            rrb.Spatial2DView(name="Right Camera", origin="Cameras/right"),
            rrb.Spatial2DView(name="Center Camera", origin="Cameras/center")
        ),
        column_shares=[3, 1]
    )

    blueprint = rrb.Blueprint(
        view_layout,
        rrb.BlueprintPanel(expanded=False),
        rrb.TimePanel(play_state="following"),
    )
    rr.send_blueprint(blueprint)

    mask_manager = None
    print("Receiving stream... Press 'Ctrl+C' to exit.")
    

    
    try:
        while True:
            output_dict = sub_socket.recv_pyobj()
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
            
            if closest_joint_state is not None:
                rr.set_time("timestamp", timestamp=closest_joint_state['monotonic_timestamp'])
                
                # Helper function to map a joint value to its correct index in the Pinocchio 'q' vector
                def update_joint(joint_name, val):
                    if model.existJointName(joint_name):
                        idx = model.joints[model.getJointId(joint_name)].idx_q
                        q[idx] = val
                
                # Map Stretch joint states to Pinocchio model joints.
                # Note: For the arm, only joint_arm_l0 is actuated in this specific URDF.
                update_joint("joint_lift", closest_joint_state['lift']['height'])
                update_joint("joint_arm_l0", closest_joint_state['arm']['extension'])
                update_joint("joint_wrist_yaw", closest_joint_state['wrist_yaw']['angle'])

                ############################################################################
                # SIGN FLIP CORRECTIONS
                # Once Stretch 4 and the Stretch 4 URDF is finalized, these sign flips 
                # should no longer be required, although updates to the IK URDF and code 
                # will be needed.  
                # 
                # wrist pitch direction needs to be flipped when run on Calder 4010
                update_joint("joint_wrist_pitch", -closest_joint_state['wrist_pitch']['angle'])
                # wrist roll direction needs to be flipped when run on Calder 4010
                update_joint("joint_wrist_roll", -closest_joint_state['wrist_roll']['angle'])
                ############################################################################

                # Compute forward kinematics and update the 3D placements of all frames
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                
                # Iterate through the requested links, fetch their 3D translations, and visualize them
                for link_name, params in config.VISUALIZED_LINKS.items():
                    frame_id = model.getFrameId(link_name)
                    translation = data.oMf[frame_id].translation
                    
                    # Apply defaults if not specified
                    color = params.get("sphere_color", [255, 255, 255])
                    radius = params.get("sphere_radius_m", 0.05)
                    show_frame = params.get("show_frame", False)
                    axes_length = params.get("frame_axis_length_m", 0.1)
                    axes_radius = params.get("frame_axis_radius_m", 0.005)
                    
                    # Render the link origin as a colored sphere
                    rr.log(f"BodyPredictions/{link_name}", rr.Points3D([translation], colors=[color], radii=[radius]))
                    
                    # Additionally render a coordinate frame if configured
                    if show_frame:
                        rotation = data.oMf[frame_id].rotation
                        rr.log(f"BodyPredictions/{link_name}_frame", rr.Arrows3D(
                            origins=[translation, translation, translation],
                            vectors=[
                                rotation[:, 0] * axes_length,
                                rotation[:, 1] * axes_length,
                                rotation[:, 2] * axes_length
                            ],
                            colors=[
                                [255, 0, 0],   # Red x-axis
                                [0, 255, 0],   # Green y-axis
                                [0, 0, 255]    # Blue z-axis
                            ],
                            radii=[axes_radius, axes_radius, axes_radius]
                        ))
            
            c_name = frame.camera_type
            lidar_str = frame.lidars_used if frame.lidars_used else "no_lidar"
            
            vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, frame.image.shape)
            visualize_rgbd_frame(c_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)
            
            # Also render the link origins and frames as an overlay on the camera image
            if closest_joint_state is not None and frame.T_base_to_cam is not None:
                for link_name, params in config.VISUALIZED_LINKS.items():
                    if model.existFrame(link_name):
                        frame_id = model.getFrameId(link_name)
                        translation_base = data.oMf[frame_id].translation
                        rotation_base = data.oMf[frame_id].rotation
                        
                        show_frame = params.get("show_frame", False)
                        axes_length = params.get("frame_axis_length_m", 0.1)
                        
                        # Prepare points to project: [origin, x_end, y_end, z_end]
                        pts_to_project = [translation_base]
                        if show_frame:
                            pts_to_project.append(translation_base + rotation_base[:, 0] * axes_length)
                            pts_to_project.append(translation_base + rotation_base[:, 1] * axes_length)
                            pts_to_project.append(translation_base + rotation_base[:, 2] * axes_length)
                            
                        pts_to_project_np = np.array(pts_to_project)
                        
                        # Use the 2D projection function which accurately handles fisheye distortion
                        img_pts, valid_mask, pts_cam = project_base_link_points_to_image(
                            pts_to_project_np, 
                            frame.T_base_to_cam, 
                            frame.camera_matrix, 
                            frame.distortion_coefficients
                        )
                        
                        if valid_mask[0]:  # Only render if the origin is in front of the camera
                            color = params.get("sphere_color", [255, 255, 255])
                            
                            # Convert physical radius to pixel radius based on depth (Z) and focal length (f_x)
                            Z = pts_cam[0, 2]
                            f_x = frame.camera_matrix[0, 0]
                            
                            physical_radius = params.get("sphere_radius_m", 0.05)
                            pixel_radius = max(1.0, (physical_radius * f_x) / Z)
                            
                            # Log 2D Point
                            rr.log(f"Cameras/{c_name}/predicted_robot_body_overlay/{link_name}", rr.Points2D(img_pts[0:1], colors=[color], radii=[pixel_radius]))
                            
                            if show_frame and np.all(valid_mask):
                                physical_axis_radius = params.get("frame_axis_radius_m", 0.005)
                                pixel_axis_radius = max(1.0, (physical_axis_radius * f_x) / Z)
                                
                                strips = [
                                    [img_pts[0], img_pts[1]], # Origin to X
                                    [img_pts[0], img_pts[2]], # Origin to Y
                                    [img_pts[0], img_pts[3]]  # Origin to Z
                                ]
                                colors = [
                                    [255, 0, 0],
                                    [0, 255, 0],
                                    [0, 0, 255]
                                ]
                                
                                # Log 2D Line Strips
                                rr.log(f"Cameras/{c_name}/predicted_robot_body_overlay/{link_name}_frame", rr.LineStrips2D(
                                    strips,
                                    colors=colors,
                                    radii=[pixel_axis_radius, pixel_axis_radius, pixel_axis_radius]
                                ))
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")

if __name__ == "__main__":
    main()
