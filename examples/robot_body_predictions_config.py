"""
Configuration parameters for robot_body_prediction.py

This file contains visualization parameters that can be adjusted
to change how the robot's predicted body parts are rendered in Rerun.

The VISUALIZED_LINKS dictionary specifies which URDF links to track and display.
Key: The exact name of the link in the URDF.
Value: A dictionary containing visualization parameters.
    - 'sphere_color': List of RGB values [R, G, B]. Defaults to [255, 255, 255] (white).
    - 'sphere_radius_m': Radius of the sphere in meters. Defaults to 0.05 (10cm diameter).
    - 'show_frame': Boolean indicating whether to draw the coordinate axes. Defaults to False.
    - 'frame_axis_length_m': Length of the axes arrows in meters. Defaults to 0.1.
    - 'frame_axis_radius_m': Thickness (radius) of the axes arrows in meters. Defaults to 0.005.

If a link is listed with an empty dictionary `{}`, it will use all default values.
"""

VISUALIZED_LINKS = {
    #"base_link": {
    #    "sphere_color": [255, 255, 255],
    #    "sphere_radius_m": 0.02,
    #},
    #"link_lift": {
    #    "sphere_color": [0, 0, 255],
    #    "sphere_radius_m": 0.02,
    #},
    "link_arm_l4": {
        "sphere_color": [0, 0, 255],
        "sphere_radius_m": 0.02,
    },
    "link_arm_l0": {
        "sphere_color": [0, 0, 255],
        "sphere_radius_m": 0.02,
    },
    "link_wrist": {
        "sphere_color": [0, 0, 255],
        "sphere_radius_m": 0.02,
    },
    #"link_wrist_yaw": {
    #    "sphere_color": [255, 255, 0],
    #    "sphere_radius_m": 0.02,
    #},
    "link_wrist_pitch": {
        "sphere_color": [255, 255, 0],
        "sphere_radius_m": 0.02,
    },
    "link_wrist_roll": {
        "sphere_color": [255, 255, 0],
        "sphere_radius_m": 0.02,
    },
    "link_gripper_s4_body": {
        "sphere_color": [0, 255, 0],
        "sphere_radius_m": 0.02,
    },
    "link_grasp_center": {
        "sphere_color": [0, 255, 0],
        "sphere_radius_m": 0.02,
        "show_frame": True,
        "frame_axis_length_m": 0.1,
        "frame_axis_radius_m": 0.005,
    }
}
