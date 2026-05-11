# pose_constants.py

# ==============================================================================
# RTMO / COCO Constants (17 Keypoints)
# ==============================================================================

RTMO_KEYPOINT_LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# RGB colors for typical CVPR visualization (MMPose / RTMO style)
RTMO_KEYPOINT_COLORS_RGB = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]
]

RTMO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Lower body
    (5, 11), (6, 12) # Torso
]

RTMO_EDGE_COLORS_RGB = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], # Head edges
    [51, 153, 255], [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], # Upper body
    [51, 153, 255], [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], # Lower body
    [0, 255, 0], [255, 128, 0] # Torso
]


# ==============================================================================
# MediaPipe Constants (33 Keypoints)
# ==============================================================================

MEDIAPIPE_KEYPOINT_LABELS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face
    (9, 10), # Mouth
    (11, 12), # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Left arm & hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Right arm & hand
    (11, 23), (12, 24), # Torso sides
    (23, 24), # Hips
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Left leg & foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # Right leg & foot
]

# Map MediaPipe keypoints to CVPR-style colors where possible:
# Face/Head: [51, 153, 255] (Light Blue)
# Left side (Upper & Lower): [0, 255, 0] (Green)
# Right side (Upper & Lower): [255, 128, 0] (Orange)

MEDIAPIPE_KEYPOINT_COLORS_RGB = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], # 0-6: face
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], # 7-10: ears and mouth
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], # 11-14: shoulders and elbows
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], # 15-22: wrists and hands
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], # 23-28: hips, knees, ankles
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0] # 29-32: heels and feet
]

MEDIAPIPE_EDGE_COLORS_RGB = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], # Face
    [51, 153, 255], # Mouth
    [51, 153, 255], # Shoulders
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], # Left arm & hand
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], # Right arm & hand
    [0, 255, 0], [255, 128, 0], # Torso sides
    [51, 153, 255], # Hips
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], # Left leg & foot
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0] # Right leg & foot
]

# ==============================================================================
# MediaPipe Hand Constants (21 Keypoints)
# ==============================================================================

MEDIAPIPE_HAND_KEYPOINT_LABELS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

MEDIAPIPE_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky and palm base
]

# CVPR-style colors for hand points (using a gradient-like scheme for different fingers or just simple colors)
MEDIAPIPE_HAND_KEYPOINT_COLORS_RGB = [
    [51, 153, 255], # 0: Wrist
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], # 1-4: Thumb
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], # 5-8: Index
    [255, 0, 255], [255, 0, 255], [255, 0, 255], [255, 0, 255], # 9-12: Middle
    [0, 255, 255], [0, 255, 255], [0, 255, 255], [0, 255, 255], # 13-16: Ring
    [255, 255, 0], [255, 255, 0], [255, 255, 0], [255, 255, 0] # 17-20: Pinky
]

MEDIAPIPE_HAND_EDGE_COLORS_RGB = [
    [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], # Thumb
    [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], # Index finger
    [255, 0, 255], [255, 0, 255], [255, 0, 255], [255, 0, 255], # Middle finger
    [0, 255, 255], [0, 255, 255], [0, 255, 255], [0, 255, 255], # Ring finger
    [255, 255, 0], [255, 255, 0], [255, 255, 0], [255, 255, 0], [255, 255, 0] # Pinky and palm base
]

# ==============================================================================
# MediaPipe Face Constants (478 Keypoints)
# ==============================================================================

# Using a single color since the mesh is dense
MEDIAPIPE_FACE_COLOR_RGB = [51, 153, 255] # Light Blue
