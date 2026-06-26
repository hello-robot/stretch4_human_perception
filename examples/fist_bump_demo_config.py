# fist_bump_demo_config.py
#
# This file contains the tunable parameters for the fist_bump_demo.py script.
# Adjust these values to modify the perception, reaching, and retraction behavior.
#
# FSM State Diagram:
# 
#  +------------------+
#  |                  |
#  v                  | (Timeout / Explode complete)
# [INITIALIZE] ----> [FIND_HAND] <------------------------+
#                          |                              |
#                          v                              |
#              [GET_READY_TO_FIST_BUMP] ---(Hand Lost)--->|
#                          |                              |
#                 (Tracking Aligned)                      |
#                          |                              |
#                          v                              |
#                     [FIST_BUMP] ---------(Hand Lost)--->+
#                          |
#                    (Hand Reached)
#                          |
#                          v
#                      [EXPLODE]
#

# ==============================================================================
# PERCEPTION PARAMETERS
# ==============================================================================

# The default text prompt used by SAM 3.1 to segment the target object in the RGB stream.
# Changing this allows the robot to track and fist bump other objects.
DEFAULT_PROMPT = 'human hands'


# ==============================================================================
# RERUN VISUALIZATION PARAMETERS
# ==============================================================================

# Radius (in meters) of the sphere used to represent the origin of link_grasp_center
GRASP_CENTER_SPHERE_RADIUS_M = 0.02

# Length (in meters) of the axes used to represent the coordinate frame of link_grasp_center
GRASP_CENTER_FRAME_AXIS_LENGTH_M = 0.1

# Radius (in meters) of the axes used to represent the coordinate frame of link_grasp_center
GRASP_CENTER_FRAME_AXIS_RADIUS_M = 0.005

# Radius (in meters) of the sphere used to represent the 3D position of segmented regions
SEGMENTED_REGION_SPHERE_RADIUS_M = 0.05


# ==============================================================================
# STATE: INITIALIZE
# Description: The robot closes its gripper and moves its arm and wrist to a 
# predefined starting pose (0.01m extension, 45 degree yaw).
# ==============================================================================

# Target joint positions for the INITIALIZE state
INIT_TARGET_LIFT_M = 0.80               # meters
INIT_TARGET_ARM_M = 0.01                # meters
INIT_TARGET_WRIST_YAW_DEG = 30.0        # degrees
INIT_TARGET_WRIST_PITCH_DEG = 0.0       # degrees
INIT_TARGET_WRIST_ROLL_DEG = 0.0        # degrees

# Position tolerances for the INITIALIZE state to consider the target reached.
INIT_TOLERANCE_DEG = 2.0  # degrees
INIT_TOLERANCE_M = 0.01    # meters (10 mm)

# Proportional gain for Joint-Space velocity control during INITIALIZE state.
PROPORTIONAL_GAIN_LIFT_ARM_INIT = 2.0
PROPORTIONAL_GAIN_WRIST_INIT = 0.05

# Position (percentage) to set the gripper to during the initialization and 
# reaching phases. 0.0 is fully closed, 100.0 is fully open.
GRIPPER_CLOSE_POS_PCT = 0.0


# ==============================================================================
# STATE: FIND_HAND
# Description: The robot waits until a hand is segmented and identified by the 
# perception pipeline, selecting the closest hand as the target.
# ==============================================================================

# (No specific tuning parameters currently required for this state)


# ==============================================================================
# STATE: GET_READY_TO_FIST_BUMP
# Description: The robot uses its mobile base and lift to perfectly align the 
# gripper's forward vector (X-axis) with the tracked hand's 3D position.
# ==============================================================================

# Proportional gain for Cartesian Tracking (mobile base rotation) during GET_READY_TO_FIST_BUMP state.
PROPORTIONAL_GAIN_TRACKING_BASE = 0.5

# Proportional gain for Cartesian Tracking (lift) during GET_READY_TO_FIST_BUMP state.
PROPORTIONAL_GAIN_TRACKING_LIFT = 4.0

# The maximum angle (in degrees) between the gripper's forward vector (X-axis)
# and the vector pointing from the gripper to the tracked hand required to start the fist bump.
START_FIST_BUMP_MAX_ANGLE_DEG = 30.0

# The maximum distance (in meters) from the gripper's origin to the hand 
# required to start the fist bump. This ensures it doesn't reach from too far away.
START_FIST_BUMP_MAX_DIST_M = 0.6

# If True, the robot will proactively trigger the fist bump if the hand is rapidly
# approaching the gripper, even if it hasn't crossed the START_FIST_BUMP_MAX_DIST_M.
# If False, the robot will only trigger the fist bump based on the static distance threshold.
ENABLE_APPROACH_VELOCITY_TRIGGER = True

# The minimum approach velocity (in meters per second) required to trigger a proactive fist bump.
# A higher value means you must thrust your fist towards the robot faster.
START_FIST_BUMP_APPROACH_VEL_M_S = 0.05

# The absolute maximum distance (in meters) at which an approach velocity can trigger a fist bump.
# This prevents the robot from triggering on someone walking quickly towards it from far away.
START_FIST_BUMP_MAX_TRACKING_DIST_M = 0.75

# The minimum height (in meters) the hand must rapidly rise above its exponentially 
# smoothed baseline height before the robot will initiate the fist bump.
START_FIST_BUMP_RAISE_HAND_M = 0.20

# The alpha blending factor for the hand height exponential moving average. 
# A smaller value means the baseline adapts more slowly, making it easier to trigger 
# the fist bump with a rapid upward movement. A larger value adapts faster.
HAND_HEIGHT_EMA_ALPHA = 0.05


# ==============================================================================
# STATE: FIST_BUMP
# Description: The robot uses Control Mode #4 to linearly translate the gripper
# toward the hand while keeping its orientation fixed.
# ==============================================================================

# The proportional gain multiplier applied to the distance error vector (meters)
# to compute the unscaled velocity vector toward the hand.
# A higher value makes the robot reach its max speed even for small distance errors.
# Example: If gain is 5.0, a 20cm error will result in a fully saturated (1.0) stick command.
PROPORTIONAL_GAIN_REACH = 6.0

# Maximum time (in seconds) allowed for the FIST_BUMP state before timing out
# and returning to the INITIALIZE state.
FIST_BUMP_TIMEOUT_S = 3.0


# ==============================================================================
# STATE: EXPLODE
# Description: When the hand is reached, the robot rapidly retracts its arm 
# while opening the gripper to simulate an "exploding" fist bump.
# ==============================================================================

# Distance threshold (in meters) between the `link_grasp_center` and the median
# 3D position of the target hand mask. When the distance drops below this value,
# the robot transitions from reaching to the explosive retraction sequence.
EXPLODE_TRIGGER_DISTANCE_M = 0.15

# The duration (in seconds) that the retraction sequence should last before
# the robot stops all movement and enters the STOPPED state.
EXPLODE_DURATION_S = 1.8

# The duration (in seconds) to wait after the explosion before transitioning
# back to the INITIALIZE state to look for another hand.
EXPLODE_WAIT_S = 3.0

# The target velocity vector (in the projected gripper frame) to command during the 
# retraction sequence. It's scaled by the receiver's max velocity limit.
# The format is [vx, vy, vz].
# Default is [-2.0, 0.0, 0.0] which commands 200% speed in reverse (away from the hand).
EXPLODE_VELOCITY_CMD = [-3.0, 0.0, 0.0]

# Position (percentage) to set the gripper to during the retraction sequence.
# Opening the gripper playfully simulates an "exploding" fist bump.
GRIPPER_OPEN_POS_PCT = 100.0


# ==============================================================================
# GLOBAL & HARDWARE PARAMETERS
# ==============================================================================

# Actuation speed (percentage) for the gripper joints during state transitions.
GRIPPER_SPEED = 100.0

# Actuation acceleration (percentage) for the gripper joints during state transitions.
GRIPPER_ACCEL = 100.0

# Maps the user-provided `--speed` command-line argument to a maximum "stick input"
# percentage (0.0 to 1.0). This value caps the magnitude of the translational 
# velocity command sent to the lower-level hardware controller.
# 
# Note: The absolute physical speed is ultimately determined by the `gamepad_speed_trans`
# limit configured in the receiver (recv_and_execute_gripper_commands.py).
SPEED_MAP = {
    'low': 0.5,      # 50% of maximum configured receiver speed
    'medium': 1.0,   # 100% of maximum configured receiver speed
    'high': 2.0,     # 200% of maximum configured receiver speed
    'max': 3.0       # 300% of maximum configured receiver speed
}
