# Stretch 4 Human Perception

This repository provides tools and examples for performing real-time human pose estimation on the Stretch 4 mobile manipulator. It primarily uses the RTMO series of models via OpenVINO for inference on edge devices (including CPU, GPU, and NPU), and SAM 3.1 for segmentation.

> [!NOTE]
> OpenVINO runs on any x86_64 compatible CPU. Hardware-specific execution like `--device NPU` or `--device GPU` within standard OpenVINO builds requires compatible Intel hardware. On desktop setups with AMD CPUs or NVIDIA GPUs, use the default `--device AUTO` or `--device CPU`.
> 
> **Important NPU Notice:** Due to a known OpenVINO compiler bug with the dynamic Non-Max Suppression (NMS) operators used in RTMO, running these models on the Intel NPU currently produces empty or garbage bounding boxes. If you specify `--device NPU`, the pipeline will automatically detect this limitation and safely fall back to the **GPU**, which executes the model with hardware acceleration.

## Installation

### Installing this package directly
Run the installation script to setup the system hardware drivers (NPU and GPU), create a virtual environment, install the python package, and download all models:

```bash
./install_dependencies.sh
```

> [!CAUTION]
> **MANDATORY REBOOT / LOGOUT:** You may be prompted for your `sudo` password to install the required Intel NPU and GPU system drivers. After the script completes, **you must log out and log back in** (or reboot) for hardware permissions to fully take effect. If you do not do this, the system will not detect the NPU/GPU hardware.

Activate the virtual environment before using the tools:

```bash
source venv/bin/activate
```

### Other Dependencies

Please clone and install the following dependencies in the virtual environment associated with this package:

Robot and External Desktop:
- `stretch4_flying_gripper`: https://github.com/hello-robot/stretch4_flying_gripper
- `stretch4_compliant_gripper` : https://github.com/hello-robot/stretch4_compliant_gripper
- `stretch4_rgbd` : https://github.com/hello-robot/stretch4_rgbd

Robot only:
- `stretch4_pyhesai_wrapper` : https://github.com/hello-robot/stretch4_pyhesai_wrapper

### Installing this package as a dependency in another project

When adding this repository as a depdendency in another package, we added console scripts (such as `install_dependencies.sh` and `setup_models.py`) that install as part of pip installing this repository that you can run from your project's python environment.

1. Add this package to your project's dependencies list: `"stretch4-human-pose-estimation @ git+ssh://git@github.com/hello-robot/stretch4_human_perception.git"`
2. Install your package package: `pip install -e .`
3. Run the install script to setup the system hardware drivers (NPU and GPU), create a virtual environment, install the python package, and download all models: `install_dependencies.sh` or `python3 -m stretch4_human_pose_estimation.utils.install_deps`
4. You can also download models using `setup_models.py` or `python3 -m stretch4_human_pose_estimation.utils.install_deps --size all`


## Downloading Models

The installation script automatically downloads all models by default. If you need to manually download them later, use the provided setup script (make sure your virtual environment is active):

```bash
# Download the medium model (default)
python3 setup_models.py
```
```bash
# Download a specific model size
python3 setup_models.py --size t
```
```bash
# Print setup instructions for SAM 3.1
python3 setup_models.py --sam3
```
```bash
# Download all models and print SAM 3.1 setup instructions
python3 setup_models.py --size all
```

### SAM3 Setup

Please clone the SAM 3.1 repository and install it as described in its README: https://github.com/facebookresearch/sam3. You will need to request access for the SAM 3.1 model weights on [Hugging Face](https://huggingface.co/facebook/sam3) and authenticate using the Hugging Face CLI:

```bash
hf auth login
```


## Running Examples

Example scripts are provided to test the pose estimation pipeline with camera streams, image directories, and 3D RGB-D projection. **Ensure your virtual environment is active** before running the examples.

Run any demos using RTMO directly on the robot.

Run any demos using SAM3 on a remote desktop. These require a high bandwidth connection between the robot and the desktop.

### Key Scripts for Desktop & Robot Communication

For any demos that use the external desktop computer, be sure to update the IP addresses in `stretch4_rgbd/rgbd_networking.py` and `stretch4_compliant_gripper/gripper_networking.py`. Run the following commands from the `stretch4_compliant_gripper` and `stretch4_rgbd` repositories on the robot:

```bash
# robot control interface
python3 stretch4_compliant_gripper/recv_and_execute_gripper_commands.py --remote

# stream RGBD data
python3 stretch4_rgbd/examples/send_rgbd_images_and_joint_states.py --remote
```

### 2D Pose Estimation: Robot Local

Terminal 1 (robot):
```bash
# stream RGBD data locally
python3 stretch4_rgbd/examples/send_rgbd_images_and_joint_states.py
```

Terminal 2 (robot): choose 1 of the following scripts:
```bash
# Run with default settings (Medium model, Left camera, AUTO device)
python3 examples/rtmo_pose_estimation.py
```
```bash
# Run the large model on the GPU using the right camera
python3 examples/rtmo_pose_estimation.py --size l --device GPU --camera right
```
```bash
# Run on a directory of images
python3 examples/rtmo_pose_estimation.py --dir /path/to/images
```

<!-- NOTE the below does not work yet? -->
```bash
# Run the stereo camera example (combines left and right camera streams side-by-side)
python3 examples/stereo_rtmo_pose_estimation.py
```

### 3D RGB-D Pose Estimation (ReRun): Robot Local

We also provide an advanced example that uses the RGB-D camera streams to infer and visualize 3D human pose keypoints alongside the point cloud in ReRun.

Terminal 1 (robot):
```bash
# Run with left camera stream and visualize 3D skeletons
python3 examples/rgbd_rtmo_pose_estimation.py --camera left --lidar left
```

### 3D RGB-D SAM 3.1 Body Segmentation (ReRun): Robot + Desktop

We provide an example that uses SAM 3.1 to segment people in the RGB-D camera streams and visualize the 2D masks and 3D point clouds in ReRun.

Terminal 1 (robot):
```bash
# stream RGBD data remotely
python3 stretch4_rgbd/examples/send_rgbd_images_and_joint_states.py --remote
```

Terminal 2 (desktop):
```bash
# Run with left camera stream and visualize SAM 3.1 segmentations
python3 recv_and_sam3_rgbd.py --remote --tracking --prompt "people"
```

The script on the desktop has many options for further processing the segmented humans. For example, mediapipe can be used to infer human skeleton, face, and hand keypoints by adding the following flags:

```bash
# 2D pose estimation
--mediapipe_body
# 2D hand pose estimation
--mediapipe_hands
# 2D face pose estimation
--mediapipe_faces
# simultaneously estimates pose, face, and hands
--mediapipe_holistic  
```

The script `recv_and_sam3_rgbd_simple.py` strips away the additional options for pose processing as a minimal example.

### 3D Robot Body Prediction (ReRun): Robot Local or Robot + Desktop

We also provide an example that tracks the robot's physical links using time-synchronized joint states, computing exact 3D Cartesian coordinates with Pinocchio and visualizing them against the RGB-D point cloud in ReRun.

#### Example usage on the robot:

Terminal 1 (robot):
```bash
# stream RGBD data locally
python3 stretch4_rgbd/examples/send_rgbd_images_and_joint_states.py
```

Terminal 2 (robot):
```bash
# Run to visualize the robot links and coordinate frames inside the point cloud
python3 examples/robot_body_prediction.py
```

#### Example usage on the robot and desktop:

Terminal 1 (robot):
```bash
# stream RGBD data remotely
python3 stretch4_rgbd/examples/send_rgbd_images_and_joint_states.py --remote
```

Terminal 2 (desktop):
```bash
# Run to visualize the robot links and coordinate frames inside the point cloud
python3 examples/robot_body_prediction.py --remote
```

> [!NOTE]
> This script natively integrates with the `stretch4_emulated_rgbd` package to provide high-performance, temporally synchronized streams. It will automatically load and apply any optimized Extrinsics calibration present for the current robot without requiring any additional command line arguments.

### Moving Relative to Humans: Robot + Desktop

We provide two examples of moving Stretch 4 relative to human pose estimates from SAM 3.1. Be sure to run both of the scripts in the "Key Scripts for Desktop & Robot Communication" section above on the robot, and then run one of the following on a desktop computer:

1. Follow a human around the room:

```bash
python3 examples/follow_person_demo.py --remote
```

The human following demonstation commands Stretch 4 to move its omnibase to follow a human around a room. **Note:** the robot will run into obstacles, but should stop when near the human.

2. Give a human a fist bump:

```bash
python3 examples/fist_bump_demo.py --remote
```

The fist bump demonstration commands Stretch 4 to track a human's hand and to perform a "fist bump" motion when the user moves their hand upwards and towards the robot. In order to trigger the fist bump with the robot, the human must move their hand upwards and towards the robot. Detection of this gesture may not work consistently for different sized users; view the parameters beginning with `START_FIST_BUMP_` in `examples/fist_bump_demo_config.py` to adjust the detection thresholds.

## Python API

You can easily integrate RTMO into your own Python code:

```python
import cv2
from stretch4_human_pose_estimation import RTMOPipeline

# Initialize the pipeline (downloads model if needed)
pipeline = RTMOPipeline(size='m', device='AUTO')

# Load an image
image = cv2.imread('test_image.jpg')

# Predict
results = pipeline.predict(image, conf_threshold=0.5)

# Visualize
vis_image = pipeline.visualize(image, results)
cv2.imshow("RTMO", vis_image)
cv2.waitKey(0)
```

You can similarly use SAM 3.1:

```python
import cv2
from stretch4_human_pose_estimation import SAM3Pipeline

# Initialize the SAM 3.1 pipeline (requires HuggingFace login and sam3 installed)
pipeline = SAM3Pipeline(prompt='people')

# Load an image
image = cv2.imread('test_image.jpg')

# Predict
results = pipeline.predict(image, conf_threshold=0.5)

# Visualize
vis_image = pipeline.visualize(image, results)
cv2.imshow("SAM 3.1", vis_image)
cv2.waitKey(0)
```
