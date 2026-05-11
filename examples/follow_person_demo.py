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

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stretch4_emulated_rgbd.shared_utils import RGBDFrame
    from stretch4_emulated_rgbd.api import visualize_rgbd_frame, ValidityMaskManager, unproject_points
    from stretch4_emulated_rgbd import rgbd_networking as gn
except ImportError:
    print("Error: stretch4_emulated_rgbd is not installed or not in python path.")
    sys.exit(1)

from stretch4_human_pose_estimation.sam3_body_segmentation import ContinuousSAM3VideoPipeline

try:
    from stretch4_gripper_modeling_and_control import gripper_networking as gcmd_net
except ImportError:
    print("Error: stretch4_gripper_modeling_and_control not in path.")
    sys.exit(1)

import follow_person_demo_config as config

from shared_perception import SegmentedObjectTracker

class State(Enum):
    INITIALIZE = 0
    FIND_PERSON = 1
    FOLLOW_PERSON = 2
    STOPPED = 3

class FollowPersonFSM:
    def __init__(self, cmd_socket):
        self.state = State.INITIALIZE
        self.cmd_socket = cmd_socket
        self.target_person_id = None
        self.last_target_pos = None

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

    def send_velocity_command(self, vx, vy, vtheta):
        cmd = {
            'control_mode': 3,
            'joint_velocity_commands': {
                'base_x': float(vx),
                'base_y': float(vy),
                'base_theta': float(vtheta)
            }
        }
        self.cmd_socket.send_pyobj(cmd)

    def update(self, people_3d_info, orig_w):
        if self.state == State.INITIALIZE:
            if len(people_3d_info) > 0:
                self.state = State.FIND_PERSON
                
        if self.state == State.FIND_PERSON:
            closest_dist = float('inf')
            closest_id = None
            for pid, info in people_3d_info.items():
                if info['closest_median_dist'] < closest_dist:
                    closest_dist = info['closest_median_dist']
                    closest_id = pid
            
            if closest_id is not None:
                self.target_person_id = closest_id
                self.last_target_pos = people_3d_info[closest_id]['median_3d']
                print(f"Targeting person {closest_id} at planar distance {closest_dist:.2f}m")
                self.state = State.FOLLOW_PERSON
                
        if self.state in [State.FOLLOW_PERSON, State.STOPPED]:
            if self.target_person_id not in people_3d_info:
                print("Target person lost! Re-evaluating...")
                self.send_zero_command()
                self.state = State.FIND_PERSON
                return
                
            target_info = people_3d_info[self.target_person_id]
            current_pos = target_info['median_3d']
            closest_dist = target_info['closest_median_dist']
            u_proj = target_info['u_proj']
            
            # Jump warning
            if self.last_target_pos is not None:
                jump_dist = np.linalg.norm(current_pos - self.last_target_pos)
                if jump_dist > config.TARGET_JUMP_WARN_THRESH_M:
                    print(f"WARNING: Target jumped by {jump_dist:.2f}m!")
            self.last_target_pos = current_pos
            
            # Rotation
            # Rotate the robot to keep the person in the center of the image.
            # horizontal_error is the normalized position of the person in the image. 0 means center, -1 means left, 1 means right.
            horizontal_error = (u_proj - (orig_w / 2.0)) / (orig_w / 2.0)
            vtheta = -config.KP_ROT * horizontal_error
            
            # State transition and Translation
            if closest_dist <= config.TARGET_STOP_DIST_M:
                if self.state != State.STOPPED:
                    print(f"Reached target (planar distance {closest_dist:.2f}m <= {config.TARGET_STOP_DIST_M}m). Stopping.")
                    self.state = State.STOPPED
                vx = 0.0
                vy = 0.0
            else:
                if self.state == State.STOPPED:
                    print(f"Target moved away (planar distance {closest_dist:.2f}m > {config.TARGET_STOP_DIST_M}m). Following.")
                    self.state = State.FOLLOW_PERSON
                
                # Compute translation
                # vector from base origin to current_pos in base XY
                dx = current_pos[0]
                dy = current_pos[1]
                
                dist = np.sqrt(dx**2 + dy**2)
                
                unit_dx = dx/dist
                unit_dy = dy/dist

                # At this distance, the maximum speed of 1.0 should be achieved.
                max_distance = config.DISTANCE_THAT_RESULTS_IN_MAX_SPEED_M
                if dist > 0.01:
                    scale = min(1.0, dist / max_distance)
                    dir_x = unit_dx * scale
                    dir_y = unit_dy * scale
                    
                    vx = config.KP_TRANS * dir_x
                    vy = config.KP_TRANS * dir_y
                    
                else:
                    vx = 0.0
                    vy = 0.0
            
            # This controller outputs a normalized angular velocity between -1 and 1, and a translation velocity 
            # vector with a magnitude between 0 and 1.
            if config.DEBUG:
                print(f"Sending command: vx={vx:.2f}, vy={vy:.2f}, vtheta={vtheta:.2f}")
                print(f"Magnitude of the translational velocity command: {np.sqrt(vx**2 + vy**2):.2f}")
            self.send_velocity_command(vx, vy, vtheta)
            
            # Return targeted state so visualization can reflect it
            return self.target_person_id
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Follow a person using SAM 3.1 and Omnidirectional Base.")
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer.')
    parser.add_argument('--disable-rate-print', action='store_true', help='Disable printing of the receiving rate.')
    parser.add_argument('--prompt', type=str, default='people', help='Prompt for SAM 3.1 segmentation (default: people).')
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
    
    fsm = FollowPersonFSM(pub_socket)
    
    print("Initializing ReRun...")
    rr.init("Stretch Follow Person Demo", spawn=False)
    rr.spawn(memory_limit="2GiB")

    camera_views = [
        rrb.Spatial2DView(name="Left Camera", origin="Cameras/left"),
        rrb.Spatial2DView(name="Right Camera", origin="Cameras/right"),
        rrb.Spatial2DView(name="Center Camera", origin="Cameras/center")
    ]
    
    timeseries_views = [
        rrb.TimeSeriesView(name="Lift & Arm", origin="Telemetry/LiftArm"),
        rrb.TimeSeriesView(name="Wrist", origin="Telemetry/Wrist"),
        rrb.TimeSeriesView(name="Head", origin="Telemetry/Head"),
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
                
            # Log Joint States
            if closest_joint_state is not None:
                rr.set_time("timestamp", timestamp=closest_joint_state['monotonic_timestamp'])
                
                rr.log("Telemetry/LiftArm/Lift", rr.Scalars(closest_joint_state['lift']['height']))
                rr.log("Telemetry/LiftArm/Arm", rr.Scalars(closest_joint_state['arm']['extension']))
                rr.log("Telemetry/LiftArm/Gripper", rr.Scalars(closest_joint_state['gripper']['pos_pct']))
                
                rr.log("Telemetry/Wrist/Yaw", rr.Scalars(closest_joint_state['wrist_yaw']['angle']))
                rr.log("Telemetry/Wrist/Pitch", rr.Scalars(closest_joint_state['wrist_pitch']['angle']))
                rr.log("Telemetry/Wrist/Roll", rr.Scalars(closest_joint_state['wrist_roll']['angle']))
                
                rr.log("Telemetry/Head/Pan", rr.Scalars(closest_joint_state['head_pan']['angle']))
                rr.log("Telemetry/Head/Tilt", rr.Scalars(closest_joint_state['head_tilt']['angle']))

            c_name = frame.camera_type
            lidar_str = frame.lidars_used if frame.lidars_used else "no_lidar"
            
            # Log RGBD frames to rerun
            vig_mask, depth_mask = mask_manager.get_masks(c_name, lidar_str, frame.image.shape)
            visualize_rgbd_frame(c_name, frame, vig_mask=vig_mask, depth_mask=depth_mask)

            people_3d_info, orig_w, orig_h = tracker.process_frame(frame, depth_mask=depth_mask)
            
            # --- FSM Update ---
            target_id = fsm.update(people_3d_info, orig_w)
            
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
