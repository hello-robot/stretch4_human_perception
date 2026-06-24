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


### Installing this package as a dependency in another project

When adding this repository as a depdendency in another package, we added console scripts (such as `install_dependencies.sh` and `setup_models.py`) that install as part of pip installing this repository that you can run from your project's python environment.

1. Add this package to your project's dependencies list: `"stretch4-human-pose-estimation @ git+ssh://git@github.com/hello-robot/stretch4_human_pose_estimation.git"`
2. Install your package package: `pip install -e .`
3. Run the install script to setup the system hardware drivers (NPU and GPU), create a virtual environment, install the python package, and download all models: `human_pose_estimation_install_dependencies` or `python3 -m stretch4_human_pose_estimation.install_deps`
4. You can also download models using `human_pose_estimation_setup_models` or `python3 -m stretch4_human_pose_estimation.install_deps --size all`


## Downloading Models

The installation script automatically downloads all models by default. If you need to manually download them later, use the provided setup script (make sure your virtual environment is active):

```bash
# Download the medium model (default)
python3 setup_models.py

# Download a specific model size
python3 setup_models.py --size t

# Print setup instructions for SAM 3.1
python3 setup_models.py --sam3

# Download all models and print SAM 3.1 setup instructions
python3 setup_models.py --size all
```

## Running the Examples

Example scripts are provided to test the pose estimation pipeline with camera streams, image directories, and 3D RGB-D projection. **Ensure your virtual environment is active** before running the examples.

### 2D Pose Estimation

```bash
# Run with default settings (Medium model, Left camera, AUTO device)
python3 examples/rtmo_pose_estimation.py

# Run the small model on the NPU using the center camera
python3 examples/rtmo_pose_estimation.py --size s --device NPU --camera center

# Run on a directory of images
python3 examples/rtmo_pose_estimation.py --dir /path/to/images

# Run the stereo camera example (combines left and right camera streams side-by-side)
python3 examples/stereo_rtmo_pose_estimation.py
```

### 3D RGB-D Pose Estimation (ReRun)

We also provide an advanced example that uses the RGB-D camera streams to infer and visualize 3D human pose keypoints alongside the point cloud in ReRun.

```bash
# Run with left camera stream and visualize 3D skeletons
python3 examples/rgbd_rtmo_pose_estimation.py --left
```

### 3D RGB-D SAM 3.1 Body Segmentation (ReRun)

We provide an example that uses SAM 3.1 to segment people in the RGB-D camera streams and visualize the 2D masks and 3D point clouds in ReRun.

```bash
# Run with left camera stream and visualize SAM 3.1 segmentations
python3 examples/rgbd_sam3_body_segmentation.py --left
```

### 3D Robot Body Prediction (ReRun)

We also provide an example that tracks the robot's physical links using time-synchronized joint states, computing exact 3D Cartesian coordinates with Pinocchio and visualizing them against the RGB-D point cloud in ReRun.

```bash
# Run to visualize the robot links and coordinate frames inside the point cloud
python3 examples/robot_body_prediction.py
```

> [!NOTE]
> This script natively integrates with the `stretch4_emulated_rgbd` package to provide high-performance, temporally synchronized streams. It will automatically load and apply any optimized Extrinsics calibration present for the current robot without requiring any additional command line arguments.

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
