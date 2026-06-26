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
from stretch4_human_pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from stretch4_human_pose_estimation.mediapipe_hand_estimator import MediaPipeHandEstimator
from stretch4_human_pose_estimation.mediapipe_face_estimator import MediaPipeFaceEstimator
from stretch4_human_pose_estimation.mediapipe_holistic_estimator import MediaPipeHolisticEstimator
from stretch4_human_pose_estimation.pose_constants import (
    MEDIAPIPE_EDGES as POSE_CONNECTIONS,
    MEDIAPIPE_KEYPOINT_COLORS_RGB,
    MEDIAPIPE_EDGE_COLORS_RGB,
    MEDIAPIPE_HAND_EDGES as HAND_CONNECTIONS,
    MEDIAPIPE_HAND_KEYPOINT_COLORS_RGB,
    MEDIAPIPE_HAND_EDGE_COLORS_RGB,
    MEDIAPIPE_FACE_COLOR_RGB
)

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
except ImportError:
    mp = None

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


class BaseEstimatorRunner:
    """
    Base class for running and visualizing MediaPipe estimators within the SAM3 tracking pipeline.
    """
    def __init__(self, name, estimator, valid_prompts, margin_factor):
        """
        :param name: Identifier name for the estimator (e.g., 'MediaPipe_Hands').
        :param estimator: The initialized MediaPipe estimator object.
        :param valid_prompts: List of optimal SAM prompts for this estimator. Empty list means no warnings.
        :param margin_factor: Float determining how much context padding is added around the SAM bounding box.
        """
        self.name = name
        self.estimator = estimator
        self.valid_prompts = valid_prompts
        self.margin_factor = margin_factor

    def check_prompt(self, current_prompt):
        """
        Emits a standard warning if the user-provided prompt is not optimal for this specific estimator.
        """
        if self.valid_prompts and current_prompt not in self.valid_prompts:
            print("*" * 80)
            print(f"* WARNING: --use_{self.name.lower()} is specified but the prompt is not '{self.valid_prompts}'. *")
            print(f"* You are using the prompt '{current_prompt}'. Results may not be optimal.      *")
            print("*" * 80)

    def _get_crop(self, image, new_xmin, new_ymin, new_xmax, new_ymax):
        """
        Calculates and extracts an image crop based on the bounding box and configured margin factor.
        Returns the cropped image, and the global X and Y offset of the crop.
        """
        margin_x = (new_xmax - new_xmin) * self.margin_factor
        margin_y = (new_ymax - new_ymin) * self.margin_factor
        
        orig_h, orig_w = image.shape[:2]
        crop_xmin = max(0, int(new_xmin - margin_x))
        crop_ymin = max(0, int(new_ymin - margin_y))
        crop_xmax = min(orig_w - 1, int(new_xmax + margin_x))
        crop_ymax = min(orig_h - 1, int(new_ymax + margin_y))
        
        if crop_xmax > crop_xmin and crop_ymax > crop_ymin:
            return image[crop_ymin:crop_ymax, crop_xmin:crop_xmax], crop_xmin, crop_ymin
        return None, 0, 0

    def _render_to_rerun(self, frame, kpts, conn_list, kp_colors, edge_colors, prefix_2d, prefix_3d, z_radius, is_face, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
        """
        Shared rendering logic that converts 2D estimator keypoints into 2D and 3D Rerun visualizations.
        """
        pts_2d = kpts[:, :2]
        confs = kpts[:, 2]
        
        lines = []
        line_colors = []
        if is_face and mp is not None:
            for edge in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                p1, p2 = edge.start, edge.end
                if confs[p1] > 0.2 and confs[p2] > 0.2:
                    lines.append([pts_2d[p1], pts_2d[p2]])
        elif conn_list is not None:
            for edge_i, (p1, p2) in enumerate(conn_list):
                if confs[p1] > 0.2 and confs[p2] > 0.2:
                    lines.append([pts_2d[p1], pts_2d[p2]])
                    line_colors.append(edge_colors[edge_i])
                    
        valid_colors = [kp_colors[k] for k, conf in enumerate(confs) if conf > 0.2] if not is_face else [kp_colors]
        valid_pts = [pt for pt, conf in zip(pts_2d, confs) if conf > 0.2]
        
        if len(valid_pts) > 0:
            rr.log(f"{prefix_2d}/keypoints", rr.Points2D(valid_pts, colors=valid_colors, radii=1.0 if is_face else 2.0))
        if len(lines) > 0:
            rr.log(f"{prefix_2d}/skeleton", rr.LineStrips2D(lines, colors=[edge_colors] if is_face else line_colors))
            
        if frame.depth_image is not None and len(frame.depth_image) > 0:
            valid_kpts = []
            for kp_idx, (x, y, conf) in enumerate(kpts):
                if conf > 0.2:
                    xi, yi = int(x), int(y)
                    if 0 <= xi < orig_w and 0 <= yi < orig_h:
                        z = frame.depth_image[yi, xi]
                        if z > 0 and z != np.inf and not np.isnan(z):
                            if depth_mask is None or depth_mask[yi, xi] > 0:
                                valid_kpts.append((kp_idx, xi, yi, z))
                                
            if len(valid_kpts) > 0:
                kpt_indices = [v[0] for v in valid_kpts]
                u_vals = np.array([v[1] for v in valid_kpts])
                v_vals = np.array([v[2] for v in valid_kpts])
                z_vals = np.array([v[3] for v in valid_kpts])
                
                pts_3d_cam = project_points_to_3d(u_vals, v_vals, z_vals, camera_matrix, dist_coeffs)
                pts_3d_base = np.dot(T_cam_to_base[:3, :3], pts_3d_cam.T).T + T_cam_to_base[:3, 3]
                idx_to_3d = {kp_idx: pt for kp_idx, pt in zip(kpt_indices, pts_3d_base)}
                
                joint_colors = [kp_colors[idx] for idx in kpt_indices] if not is_face else [kp_colors]
                rr.log(f"{prefix_3d}_joints", rr.Points3D(pts_3d_base, colors=joint_colors, radii=[z_radius]))
                
                lines_3d = []
                lines_3d_colors = []
                if is_face and mp is not None:
                    for edge in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                        p1, p2 = edge.start, edge.end
                        if p1 in idx_to_3d and p2 in idx_to_3d:
                            lines_3d.append([idx_to_3d[p1], idx_to_3d[p2]])
                elif conn_list is not None:
                    for edge_i, (p1, p2) in enumerate(conn_list):
                        if p1 in idx_to_3d and p2 in idx_to_3d:
                            lines_3d.append([idx_to_3d[p1], idx_to_3d[p2]])
                            lines_3d_colors.append(edge_colors[edge_i])
                            
                if len(lines_3d) > 0:
                    rr.log(f"{prefix_3d}_skeleton", rr.LineStrips3D(lines_3d, colors=[edge_colors] if is_face else lines_3d_colors, radii=[z_radius/2.0]))
                else:
                    rr.log(f"{prefix_3d}_skeleton", rr.Clear(recursive=True))
            else:
                rr.log(f"{prefix_3d}_joints", rr.Clear(recursive=True))
                rr.log(f"{prefix_3d}_skeleton", rr.Clear(recursive=True))
        else:
            rr.log(f"{prefix_3d}_joints", rr.Clear(recursive=True))
            rr.log(f"{prefix_3d}_skeleton", rr.Clear(recursive=True))

    def process_and_visualize(self, frame, new_xmin, new_ymin, new_xmax, new_ymax, c_name, person_i, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
        """
        Must be implemented by subclasses to handle specific prediction and rendering workflows.
        """
        raise NotImplementedError


class PoseRunner(BaseEstimatorRunner):
    def process_and_visualize(self, frame, new_xmin, new_ymin, new_xmax, new_ymax, c_name, person_i, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
        crop_img, crop_xmin, crop_ymin = self._get_crop(frame.image, new_xmin, new_ymin, new_xmax, new_ymax)
        if crop_img is not None:
            keypoints = self.estimator.predict_crop(crop_img, crop_xmin, crop_ymin)
            if keypoints is not None:
                self._render_to_rerun(
                    frame, keypoints, POSE_CONNECTIONS, MEDIAPIPE_KEYPOINT_COLORS_RGB, MEDIAPIPE_EDGE_COLORS_RGB,
                    f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/pose",
                    f"SegmentedPeople/{c_name}/person_{person_i}_pose",
                    0.01, False, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                )


class HandRunner(BaseEstimatorRunner):
    def process_and_visualize(self, frame, new_xmin, new_ymin, new_xmax, new_ymax, c_name, person_i, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
        crop_img, crop_xmin, crop_ymin = self._get_crop(frame.image, new_xmin, new_ymin, new_xmax, new_ymax)
        if crop_img is not None:
            hands_kpts = self.estimator.predict_crop(crop_img, crop_xmin, crop_ymin)
            if hands_kpts is not None:
                for hand_idx, hand_kpts in enumerate(hands_kpts):
                    self._render_to_rerun(
                        frame, hand_kpts, HAND_CONNECTIONS, MEDIAPIPE_HAND_KEYPOINT_COLORS_RGB, MEDIAPIPE_HAND_EDGE_COLORS_RGB,
                        f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/hand_{hand_idx}",
                        f"SegmentedPeople/{c_name}/person_{person_i}_hand_{hand_idx}",
                        0.005, False, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                    )


class FaceRunner(BaseEstimatorRunner):
    def process_and_visualize(self, frame, new_xmin, new_ymin, new_xmax, new_ymax, c_name, person_i, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
        crop_img, crop_xmin, crop_ymin = self._get_crop(frame.image, new_xmin, new_ymin, new_xmax, new_ymax)
        if crop_img is not None:
            faces_kpts = self.estimator.predict_crop(crop_img, crop_xmin, crop_ymin)
            if faces_kpts is not None:
                for face_idx, face_kpts in enumerate(faces_kpts):
                    self._render_to_rerun(
                        frame, face_kpts, None, MEDIAPIPE_FACE_COLOR_RGB, MEDIAPIPE_FACE_COLOR_RGB,
                        f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/face_{face_idx}",
                        f"SegmentedPeople/{c_name}/person_{person_i}_face_{face_idx}",
                        0.002, True, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                    )


class HolisticRunner(BaseEstimatorRunner):
    def __init__(self, name, estimator, valid_prompts, margin_factor, holistic_outputs_list):
        super().__init__(name, estimator, valid_prompts, margin_factor)
        self.holistic_outputs_list = holistic_outputs_list

    def process_and_visualize(self, frame, new_xmin, new_ymin, new_xmax, new_ymax, c_name, person_i, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h):
        crop_img, crop_xmin, crop_ymin = self._get_crop(frame.image, new_xmin, new_ymin, new_xmax, new_ymax)
        if crop_img is not None:
            holistic_res = self.estimator.predict_crop(crop_img, crop_xmin, crop_ymin, self.holistic_outputs_list)
            
            if 'pose' in holistic_res:
                self._render_to_rerun(
                    frame, holistic_res['pose'], POSE_CONNECTIONS, MEDIAPIPE_KEYPOINT_COLORS_RGB, MEDIAPIPE_EDGE_COLORS_RGB,
                    f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/pose",
                    f"SegmentedPeople/{c_name}/person_{person_i}_pose",
                    0.01, False, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                )
            if 'face' in holistic_res:
                self._render_to_rerun(
                    frame, holistic_res['face'], None, MEDIAPIPE_FACE_COLOR_RGB, MEDIAPIPE_FACE_COLOR_RGB,
                    f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/face_0",
                    f"SegmentedPeople/{c_name}/person_{person_i}_face_0",
                    0.002, True, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                )
            if 'left_hand' in holistic_res:
                self._render_to_rerun(
                    frame, holistic_res['left_hand'], HAND_CONNECTIONS, MEDIAPIPE_HAND_KEYPOINT_COLORS_RGB, MEDIAPIPE_HAND_EDGE_COLORS_RGB,
                    f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/left_hand",
                    f"SegmentedPeople/{c_name}/person_{person_i}_left_hand",
                    0.005, False, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                )
            if 'right_hand' in holistic_res:
                self._render_to_rerun(
                    frame, holistic_res['right_hand'], HAND_CONNECTIONS, MEDIAPIPE_HAND_KEYPOINT_COLORS_RGB, MEDIAPIPE_HAND_EDGE_COLORS_RGB,
                    f"Cameras/{c_name}/skeletal_overlay/person_{person_i}/right_hand",
                    f"SegmentedPeople/{c_name}/person_{person_i}_right_hand",
                    0.005, False, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h
                )


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
    parser.add_argument('--mediapipe_body', action='store_true', help='Enable MediaPipe Pose Estimator on SAM 3.1 bounding boxes. Only works with a body segmentation prompt (e.g., "people").')
    parser.add_argument('--mediapipe_hands', action='store_true', help='Enable MediaPipe Hand Landmarker on SAM 3.1 bounding boxes. Only works with a hands segmentation prompt (e.g., "hands", "left hands", "right hands").')
    parser.add_argument('--mediapipe_faces', action='store_true', help='Enable MediaPipe Face Landmarker on SAM 3.1 bounding boxes. Only works with a face segmentation prompt (e.g., "head", "face").')
    parser.add_argument('--mediapipe_holistic', nargs='?', const='pose,face,left_hand,right_hand', default=None, help='Enable MediaPipe Holistic Landmarker on SAM 3.1 bounding boxes. Only works with a body segmentation prompt (e.g., "people"). Optionally specify comma-separated list of outputs (e.g., pose,face,left_hand,right_hand).')
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
        
    runners = []
    
    if args.mediapipe_body:
        print("Initializing MediaPipe Pose Estimator...")
        try:
            runners.append(PoseRunner("mediapipe_body", MediaPipePoseEstimator(), ["people"], 0.125))
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            return
            
    if args.mediapipe_hands:
        print("Initializing MediaPipe Hand Landmarker...")
        try:
            runners.append(HandRunner("mediapipe_hands", MediaPipeHandEstimator(num_hands=2), ["hands", "left hands", "right hands"], 0.3))
        except Exception as e:
            print(f"Error initializing MediaPipe Hand Landmarker: {e}")
            return
            
    if args.mediapipe_faces:
        print("Initializing MediaPipe Face Landmarker...")
        try:
            runners.append(FaceRunner("mediapipe_faces", MediaPipeFaceEstimator(num_faces=1), ["head", "face"], 0.3))
        except Exception as e:
            print(f"Error initializing MediaPipe Face Landmarker: {e}")
            return
            
    if args.mediapipe_holistic is not None:
        print("Initializing MediaPipe Holistic Landmarker...")
        holistic_outputs_list = [x.strip() for x in args.mediapipe_holistic.split(',')]
        try:
            runners.append(HolisticRunner("mediapipe_holistic", MediaPipeHolisticEstimator(), ["people"], 0.25, holistic_outputs_list))
        except Exception as e:
            print(f"Error initializing MediaPipe Holistic Landmarker: {e}")
            return
            
    for runner in runners:
        runner.check_prompt(args.prompt)
            
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
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_pose_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_pose_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_0_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_0_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_1_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_1_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_face_0_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_face_0_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_left_hand_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_left_hand_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_right_hand_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_right_hand_skeleton", rr.Clear(recursive=True))
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
                
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/hand_0", rr.Clear(recursive=True))
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/hand_1", rr.Clear(recursive=True))
                rr.log(f"Cameras/{c_name}/skeletal_overlay/person_{i}/face_0", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_0_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_0_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_1_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_hand_1_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_face_0_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_face_0_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_left_hand_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_left_hand_skeleton", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_right_hand_joints", rr.Clear(recursive=True))
                rr.log(f"SegmentedPeople/{c_name}/person_{i}_right_hand_skeleton", rr.Clear(recursive=True))
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
                    
                    for runner in runners:
                        runner.process_and_visualize(frame, new_xmin, new_ymin, new_xmax, new_ymax, c_name, i, depth_mask, camera_matrix, dist_coeffs, T_cam_to_base, orig_w, orig_h)
                        
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
