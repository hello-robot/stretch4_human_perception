# Parameters for follow_person_demo.py

DEBUG = False

# ==========================================
# PROPORTIONAL GAINS    
# ==========================================

# Proportional gain for translation (velocity per meter of error)
KP_TRANS = 1.0

if not (0.0 <= KP_TRANS <= 1.0):
    raise ValueError(f"KP_TRANS ({KP_TRANS}) must be between 0.0 and 1.0 inclusive.")

# Proportional gain for rotation based on normalized image error (-1.0 to 1.0)
KP_ROT = 0.4

if not (0.0 <= KP_ROT <= 1.0):
    raise ValueError(f"KP_ROT ({KP_ROT}) must be between 0.0 and 1.0 inclusive.")

# ==========================================
# KEY DISTANCES
# ==========================================

# When the person is at this distance (in meters), the robot will move at its maximum translational speed.
DISTANCE_THAT_RESULTS_IN_MAX_SPEED_M = 2.0

# Planar distance (ignoring height) from the robot to stop at (meters)
TARGET_STOP_DIST_M = 0.75

# If the 3D target centroid moves by more than this threshold between frames, print a warning (meters)
TARGET_JUMP_WARN_THRESH_M = 0.5
