#!/usr/bin/env python3
import argparse
import time
import zmq
import sys
import os
import cv2
import numpy as np
from enum import Enum

import rerun as rr
import rerun.blueprint as rrb
import pinocchio as pin

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stretch4_emulated_rgbd.shared_utils import RGBDFrame
    from stretch4_emulated_rgbd.api import visualize_rgbd_frame, ValidityMaskManager, project_base_link_points_to_image
    from stretch4_emulated_rgbd import rgbd_networking as gn
except ImportError:
    print("Error: stretch4_emulated_rgbd is not installed or not in python path.")
    sys.exit(1)

from stretch4_human_pose_estimation.sam3_body_segmentation import ContinuousSAM3VideoPipeline
from shared_perception import SegmentedObjectTracker

try:
    from stretch4_gripper_modeling_and_control import gripper_networking as gcmd_net
except ImportError:
    print("Error: stretch4_gripper_modeling_and_control not in path.")
    sys.exit(1)

import fist_bump_demo_config as config

class State(Enum):
    INITIALIZE = 0
    FIND_HAND = 1
    GET_READY_TO_FIST_BUMP = 2
    FIST_BUMP = 3
    EXPLODE = 4
    STOPPED = 5

class FistBumpFSM:
    def __init__(self, cmd_socket, max_speed):
        self.state = State.INITIALIZE
        self.cmd_socket = cmd_socket
        self.target_hand_id = None
        self.retract_start_time = 0.0
        self.fist_bump_start_time = 0.0
        self.max_speed = max_speed
        
        # Approach velocity tracking
        self.last_dist_to_gripper = None
        self.last_dist_time = None
        self.smoothed_approach_velocity = 0.0
        
        # Gesture tracking
        self.ema_hand_height = None

    def send_zero_command(self):
        cmd = {
            'control_mode': 3,
            'joint_velocity_commands': {
                'base_x': 0.0,
                'base_y': 0.0,
                'base_theta': 0.0
            }
        }
        self.cmd_socket.send_pyobj(cmd)

    def send_velocity_command(self, j_cmds):
        cmd = {
            'control_mode': 3,
            'joint_velocity_commands': j_cmds
        }
        self.cmd_socket.send_pyobj(cmd)

    def send_velocity_and_grip_command(self, j_cmds, grip_pct):
        cmd = {
            'control_mode': 3,
            'joint_velocity_commands': j_cmds,
            'grip': {
                'pos_pct': grip_pct,
                'speed': config.GRIPPER_SPEED,
                'accel': config.GRIPPER_ACCEL
            }
        }
        self.cmd_socket.send_pyobj(cmd)
        
    def send_explode_command(self):
        cmd = {
            'control_mode': 1,
            'v_desired': config.EXPLODE_VELOCITY_CMD,
            'rot_change': [0.0, 0.0, 0.0],
            'grip': {
                'pos_pct': config.GRIPPER_OPEN_POS_PCT,
                'speed': config.GRIPPER_SPEED,
                'accel': config.GRIPPER_ACCEL
            }
        }
        self.cmd_socket.send_pyobj(cmd)

    def send_reach_command(self, vx, vy, vz):
        cmd = {
            'control_mode': 4,
            'v_desired': [float(vx), float(vy), float(vz)]
        }
        print(f"Sending Mode 4 reach command: {cmd}")
        self.cmd_socket.send_pyobj(cmd)

    def update(self, hands_3d_info, grasp_center_pos, grasp_center_rot, joint_state):
        if self.state == State.INITIALIZE:
            if joint_state is None:
                return None
                
            # Removed standalone send_close_gripper_command() to prevent ZMQ dropping messages
            
            # Targets: Lift = 0.80m, Arm = 0.01m, Wrist Pitch = 0 deg, Wrist Roll = 0 deg, Wrist Yaw = 45 deg
            target_lift = config.INIT_TARGET_LIFT_M
            target_arm = config.INIT_TARGET_ARM_M
            target_yaw = np.radians(config.INIT_TARGET_WRIST_YAW_DEG)
            target_pitch = np.radians(config.INIT_TARGET_WRIST_PITCH_DEG)
            target_roll = np.radians(config.INIT_TARGET_WRIST_ROLL_DEG)
            
            curr_lift = joint_state['lift']['height']
            curr_arm = joint_state['arm']['extension']
            curr_yaw = joint_state['wrist_yaw']['angle']
            curr_pitch = joint_state['wrist_pitch']['angle']
            curr_roll = joint_state['wrist_roll']['angle']
            
            err_lift = target_lift - curr_lift
            err_arm = target_arm - curr_arm
            err_yaw = target_yaw - curr_yaw
            err_pitch = target_pitch - curr_pitch
            err_roll = target_roll - curr_roll
            
            # Check if within tolerance
            if (abs(err_lift) < config.INIT_TOLERANCE_M and 
                abs(err_arm) < config.INIT_TOLERANCE_M and 
                abs(err_yaw) < np.radians(config.INIT_TOLERANCE_DEG) and 
                abs(err_pitch) < np.radians(config.INIT_TOLERANCE_DEG) and 
                abs(err_roll) < np.radians(config.INIT_TOLERANCE_DEG)):
                
                print("INITIALIZE complete. Transitioning to FIND_HAND.")
                self.send_zero_command()
                self.state = State.FIND_HAND
            else:
                kp = config.PROPORTIONAL_GAIN_INIT
                j_cmds = {
                    'lift': kp * err_lift,
                    'arm': kp * err_arm,
                    'wrist_yaw': kp * err_yaw,
                    'wrist_pitch': -kp * err_pitch, # FIX FOR FUTURE ROBOTS
                    'wrist_roll': -kp * err_roll # FIX FOR FUTURE ROBOTS
                }
                self.send_velocity_and_grip_command(j_cmds, config.GRIPPER_CLOSE_POS_PCT)
                
        if self.state == State.FIND_HAND:
            if grasp_center_pos is None:
                return None
                
            closest_dist = float('inf')
            closest_id = None
            for hid, info in hands_3d_info.items():
                hand_pos = info['median_3d']
                dist_to_gripper = np.linalg.norm(hand_pos - grasp_center_pos)
                if dist_to_gripper < closest_dist:
                    closest_dist = dist_to_gripper
                    closest_id = hid
            
            if closest_id is not None:
                self.target_hand_id = closest_id
                self.ema_hand_height = hands_3d_info[closest_id]['median_3d'][2]
                print(f"Targeting hand {closest_id} at distance {closest_dist:.2f}m from gripper (Initial Height: {self.ema_hand_height:.2f}m)")
                self.state = State.GET_READY_TO_FIST_BUMP
                
        if self.state == State.GET_READY_TO_FIST_BUMP:
            if self.target_hand_id not in hands_3d_info or grasp_center_pos is None:
                print("Target hand lost! Re-evaluating...")
                self.send_zero_command()
                self.state = State.FIND_HAND
                return None
                
            target_info = hands_3d_info[self.target_hand_id]
            current_hand_pos = target_info['median_3d']
            
            dist_to_gripper = np.linalg.norm(current_hand_pos - grasp_center_pos)
            
            vec = current_hand_pos - grasp_center_pos
            error_y = np.dot(vec, grasp_center_rot[:, 1])
            error_z = np.dot(vec, grasp_center_rot[:, 2])
            
            vec_mag = np.linalg.norm(vec)
            if vec_mag > 0:
                vec_norm = vec / vec_mag
                cos_theta = np.clip(np.dot(vec_norm, grasp_center_rot[:, 0]), -1.0, 1.0)
                angle_to_hand = np.arccos(cos_theta)
            else:
                angle_to_hand = 0.0
                
            # Approach velocity calculation
            current_time = time.time()
            if self.last_dist_to_gripper is not None and self.last_dist_time is not None:
                dt = current_time - self.last_dist_time
                if dt > 0:
                    v_app = (self.last_dist_to_gripper - dist_to_gripper) / dt
                    self.smoothed_approach_velocity = 0.8 * self.smoothed_approach_velocity + 0.2 * v_app
                    
            self.last_dist_to_gripper = dist_to_gripper
            self.last_dist_time = current_time
            
            is_close_enough = dist_to_gripper < config.START_FIST_BUMP_MAX_DIST_M
            is_moving_toward = config.ENABLE_APPROACH_VELOCITY_TRIGGER and \
                               (dist_to_gripper < config.START_FIST_BUMP_MAX_TRACKING_DIST_M) and \
                               (self.smoothed_approach_velocity > config.START_FIST_BUMP_APPROACH_VEL_M_S)
                               
            # Adaptive baseline for hand height gesture
            current_height = current_hand_pos[2]
            if self.ema_hand_height is None:
                self.ema_hand_height = current_height
            else:
                alpha = config.HAND_HEIGHT_EMA_ALPHA
                self.ema_hand_height = (1.0 - alpha) * self.ema_hand_height + alpha * current_height
                
            is_raised_enough = current_height >= (self.ema_hand_height + config.START_FIST_BUMP_RAISE_HAND_M)
            
            if angle_to_hand < np.radians(config.START_FIST_BUMP_MAX_ANGLE_DEG) and is_raised_enough and (is_close_enough or is_moving_toward):
                if is_moving_toward and not is_close_enough:
                    print(f"Proactive tracking aligned! (Approach vel: {self.smoothed_approach_velocity:.2f} m/s). Ready to Fist Bump.")
                else:
                    print("Tracking aligned! Ready to Fist Bump.")
                self.state = State.FIST_BUMP
                self.fist_bump_start_time = time.time()
                # DO NOT RETURN, ALLOW FALL THROUGH TO FIST_BUMP
            else:
                kp_lift = config.PROPORTIONAL_GAIN_TRACKING_LIFT
                kp_base = config.PROPORTIONAL_GAIN_TRACKING_BASE
                v_lift = -kp_lift * error_z # Negated to fix tracking direction
                v_theta = -kp_base * error_y # Negated to fix tracking direction
                
                # Cap speeds for safety
                v_lift = max(-self.max_speed, min(self.max_speed, v_lift))
                v_theta = max(-self.max_speed, min(self.max_speed, v_theta))
                
                j_cmds = {
                    'lift': v_lift,
                    'base_theta': v_theta
                }
                self.send_velocity_command(j_cmds)
                return self.target_hand_id
                
        if self.state == State.FIST_BUMP:
            if time.time() - self.fist_bump_start_time > config.FIST_BUMP_TIMEOUT_S:
                print("Fist bump timed out. Returning to INITIALIZE.")
                self.send_zero_command()
                self.state = State.INITIALIZE
                return None
                
            if self.target_hand_id not in hands_3d_info or grasp_center_pos is None:
                print("Target hand lost! Re-evaluating...")
                self.send_zero_command()
                self.state = State.FIND_HAND
                return None
                
            target_info = hands_3d_info[self.target_hand_id]
            current_hand_pos = target_info['median_3d']
            
            dist_to_gripper = np.linalg.norm(current_hand_pos - grasp_center_pos)
            
            if dist_to_gripper <= config.EXPLODE_TRIGGER_DISTANCE_M:
                print("EXPLODING FIST BUMP! Retracting...")
                self.state = State.EXPLODE
                self.retract_start_time = time.time()
                self.send_explode_command()
                return self.target_hand_id
                
            # Mode 4 Translation in Projected Gripper Frame
            yaw = np.arctan2(grasp_center_rot[1, 0], grasp_center_rot[0, 0])
            R_proj = np.array([
                [np.cos(yaw), np.sin(yaw), 0],
                [-np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            error_proj = R_proj @ (current_hand_pos - grasp_center_pos)
            
            kp = config.PROPORTIONAL_GAIN_REACH
            vx = kp * error_proj[0]
            vy = kp * error_proj[1]
            vz = kp * error_proj[2]
            
            mag = np.linalg.norm([vx, vy, vz])
            if mag > self.max_speed:
                vx = (vx / mag) * self.max_speed
                vy = (vy / mag) * self.max_speed
                vz = (vz / mag) * self.max_speed
                
            self.send_reach_command(vx, vy, vz)
            return self.target_hand_id
            
        if self.state == State.EXPLODE:
            time_in_explode = time.time() - self.retract_start_time
            if time_in_explode >= (config.EXPLODE_DURATION_S + config.EXPLODE_WAIT_S):
                print("Explode and wait complete. Returning to INITIALIZE.")
                self.state = State.INITIALIZE
                self.send_zero_command()
            elif time_in_explode >= config.EXPLODE_DURATION_S:
                # Still in EXPLODE but waiting
                self.send_zero_command()
            else:
                self.send_explode_command()
                
        return self.target_hand_id

def main():
    parser = argparse.ArgumentParser(description="Fist bump demo using SAM 3.1 and Control Mode 4.")
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer.')
    parser.add_argument('--disable-rate-print', action='store_true', help='Disable printing of the receiving rate.')
    parser.add_argument('--prompt', type=str, default=config.DEFAULT_PROMPT, help=f'Prompt for SAM 3.1 segmentation (default: {config.DEFAULT_PROMPT}).')
    parser.add_argument('--speed', type=str, choices=list(config.SPEED_MAP.keys()), default='low', help='Speeds of the joints')
    args = parser.parse_args()
    
    print(f"Initializing SAM 3.1 Pipeline (Tracking: True, Prompt: {args.prompt})...")
    try:
        pipeline = ContinuousSAM3VideoPipeline(prompt=args.prompt)
        tracker = SegmentedObjectTracker(pipeline, args.prompt)
    except Exception as e:
        print(f"Error initializing SAM 3.1 pipeline: {e}")
        return
        
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
    sub_socket.setsockopt(zmq.RCVHWM, 1)
    sub_socket.setsockopt(zmq.CONFLATE, 1)
    
    pub_socket = context.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1)
    pub_socket.setsockopt(zmq.RCVHWM, 1)
    
    if args.remote:
        sub_address = f"tcp://{gn.robot_ip}:{gn.rgbd_and_joints_port}"
        pub_address = f"tcp://*:{gcmd_net.gripper_cmd_port}"
    else:
        sub_address = f"tcp://127.0.0.1:{gn.rgbd_and_joints_port}"
        pub_address = f"tcp://127.0.0.1:{gcmd_net.gripper_cmd_port}"
        
    print(f"Connecting ZMQ Subscriber to {sub_address}")
    sub_socket.connect(sub_address)
    
    print(f"Binding ZMQ Publisher to {pub_address}")
    pub_socket.bind(pub_address)
    
    # The max_speed represents the max 'stick' input percentage sent to the hardware receiver
    fsm = FistBumpFSM(pub_socket, max_speed=config.SPEED_MAP[args.speed])
    
    print("Initializing ReRun...")
    rr.init("Stretch Fist Bump Demo", spawn=False)
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
            rrb.Spatial3DView(name="Base Frame", origin="/", contents=["+ Pointclouds/base_frame/**", "+ BodyPredictions/**"]),
            rrb.Spatial3DView(name="Segmented Hands", origin="/", contents=["+ SegmentedPeople/**", "+ BodyPredictions/link_grasp_center", "+ BodyPredictions/link_grasp_center_frame"])
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

    # Load URDF model into Pinocchio to compute forward kinematics
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "urdfs", "calder_ik.urdf")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q = pin.neutral(model) # Initialize the joint configuration vector to neutral
    
    grasp_center_frame_id = model.getFrameId("link_grasp_center") if model.existFrame("link_grasp_center") else None

    print("Receiving stream... Press 'Ctrl+C' to exit.")
    
    last_print_time = time.time()
    frames_received = 0
    last_seq_num = None
    dropped_messages = 0
    
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
            
            frames_received += 1
            img_seq = frame.image_frame.frame_number
            if img_seq is not None:
                if last_seq_num is not None:
                    dropped = img_seq - last_seq_num - 1
                    if dropped > 0:
                        dropped_messages += dropped
                last_seq_num = img_seq
                
            grasp_center_pos = None
            grasp_center_rot = None
            
            # Log Joint States and calculate kinematics
            if closest_joint_state is not None:
                rr.set_time("timestamp", timestamp=closest_joint_state['monotonic_timestamp'])
                
                rr.log("Telemetry/LiftArm/Lift", rr.Scalars(closest_joint_state['lift']['height']))
                rr.log("Telemetry/LiftArm/Arm", rr.Scalars(closest_joint_state['arm']['extension']))
                rr.log("Telemetry/LiftArm/Gripper", rr.Scalars(closest_joint_state['gripper']['pos_pct']))
                
                rr.log("Telemetry/Wrist/Yaw", rr.Scalars(closest_joint_state['wrist_yaw']['angle']))
                rr.log("Telemetry/Wrist/Pitch", rr.Scalars(closest_joint_state['wrist_pitch']['angle']))
                rr.log("Telemetry/Wrist/Roll", rr.Scalars(closest_joint_state['wrist_roll']['angle']))
                
                # Update kinematics
                def update_joint(joint_name, val):
                    if model.existJointName(joint_name):
                        idx = model.joints[model.getJointId(joint_name)].idx_q
                        q[idx] = val
                
                update_joint("joint_lift", closest_joint_state['lift']['height'])
                update_joint("joint_arm_l0", closest_joint_state['arm']['extension'])
                update_joint("joint_wrist_yaw", closest_joint_state['wrist_yaw']['angle'])
                update_joint("joint_wrist_pitch", -closest_joint_state['wrist_pitch']['angle'])
                update_joint("joint_wrist_roll", -closest_joint_state['wrist_roll']['angle'])
                
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                
                if grasp_center_frame_id is not None:
                    grasp_center_pos = data.oMf[grasp_center_frame_id].translation
                    grasp_center_rot = data.oMf[grasp_center_frame_id].rotation
                    rotation = grasp_center_rot
                    
                    rr.log("BodyPredictions/link_grasp_center", rr.Points3D([grasp_center_pos], colors=[[255, 255, 0]], radii=[config.GRASP_CENTER_SPHERE_RADIUS_M]))
                    axes_length = config.GRASP_CENTER_FRAME_AXIS_LENGTH_M
                    axes_radius = config.GRASP_CENTER_FRAME_AXIS_RADIUS_M
                    rr.log("BodyPredictions/link_grasp_center_frame", rr.Arrows3D(
                        origins=[grasp_center_pos, grasp_center_pos, grasp_center_pos],
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
                    if frame.T_base_to_cam is not None:
                        pts_to_project = np.array([grasp_center_pos,
                                                   grasp_center_pos + rotation[:, 0] * axes_length,
                                                   grasp_center_pos + rotation[:, 1] * axes_length,
                                                   grasp_center_pos + rotation[:, 2] * axes_length])
                        img_pts, valid_mask, pts_cam = project_base_link_points_to_image(
                            pts_to_project, 
                            frame.T_base_to_cam, 
                            frame.camera_matrix, 
                            frame.distortion_coefficients
                        )
                        
                        if valid_mask[0]:
                            Z = pts_cam[0, 2]
                            f_x = frame.camera_matrix[0, 0]
                            pixel_radius = max(1.0, (config.GRASP_CENTER_SPHERE_RADIUS_M * f_x) / Z)
                            rr.log(f"Cameras/{c_name}/predicted_robot_body_overlay/link_grasp_center", rr.Points2D(img_pts[0:1], colors=[[255, 255, 0]], radii=[pixel_radius]))
                            
                            if np.all(valid_mask):
                                pixel_axis_radius = max(1.0, (config.GRASP_CENTER_FRAME_AXIS_RADIUS_M * f_x) / Z)
                                strips = [
                                    [img_pts[0], img_pts[1]],
                                    [img_pts[0], img_pts[2]],
                                    [img_pts[0], img_pts[3]]
                                ]
                                colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                                rr.log(f"Cameras/{c_name}/predicted_robot_body_overlay/link_grasp_center_frame", rr.LineStrips2D(
                                    strips, colors=colors, radii=[pixel_axis_radius, pixel_axis_radius, pixel_axis_radius]
                                ))

            c_name = frame.camera_type
            lidar_str = frame.lidars_used if frame.lidars_used else "no_lidar"
            
            # Log RGBD frames to rerun
            vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, frame.image.shape)
            visualize_rgbd_frame(c_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)

            hands_3d_info, orig_w, orig_h = tracker.process_frame(frame, depth_mask=depth_mask)
            
            # --- FSM Update ---
            target_id = fsm.update(hands_3d_info, grasp_center_pos, grasp_center_rot, closest_joint_state)
            
            if target_id is not None:
                rr.log(f"SegmentedPeople/{c_name}/person_{target_id}_center/status", rr.TextDocument(f"TARGET ({fsm.state.name})"))
            
            # Print stats
            current_time = time.time()
            elapsed = current_time - last_print_time
            if elapsed >= 1.0:
                if not args.disable_rate_print:
                    hz = frames_received / elapsed
                    print(f"Rate: {hz:.2f} Hz | Estimated dropped messages in last {elapsed:.1f}s: {dropped_messages} | FSM State: {fsm.state.name}")
                frames_received = 0
                dropped_messages = 0
                last_print_time = current_time
                
    except KeyboardInterrupt:
        pass
    finally:
        fsm.send_zero_command()
        print("\nStopped.")

if __name__ == "__main__":
    main()
