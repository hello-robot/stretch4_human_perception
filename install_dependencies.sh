#!/bin/bash
set -e

echo "Checking for Intel hardware for specific driver installations..."
if lscpu | grep -iq "GenuineIntel"; then
    echo "Intel CPU detected. Requesting sudo privileges to install Intel NPU/GPU drivers..."
    # 1. Update apt repositories
    sudo apt update || true

    # 2. Install Intel Compute Runtime for GPU (OpenCL & Level Zero)
    sudo apt install -y intel-opencl-icd libze1 libze-intel-gpu1 || echo "Warning: Failed to install Intel apt packages."

    # 3. Install official Intel NPU Driver via snap
    sudo snap install intel-npu-driver --beta || echo "Warning: Failed to install intel-npu-driver via snap."
    
    # 4. Configure Dynamic Linker so OpenVINO can find the snap NPU driver (Level Zero)
    if [ -d "/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu" ]; then
        echo "/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu" | sudo tee /etc/ld.so.conf.d/intel-npu.conf > /dev/null
        sudo ldconfig
    fi

    # 5. Ensure user has permissions
    sudo usermod -a -G render $USER || true

    # 6. Reload udev rules
    sudo udevadm control --reload-rules && sudo udevadm trigger || true

    echo "Intel hardware drivers installed successfully."
else
    echo "Non-Intel CPU detected. Skipping Intel-specific hardware driver installations."
fi

VENV_DIR="venv"

echo "Setting up virtual environment in $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Installing stretch4_emulated_rgbd dependency..."
if [ -d "../stretch4_emulated_rgbd" ]; then
    python3 -m pip install -e ../stretch4_emulated_rgbd
else
    echo "Warning: ../stretch4_emulated_rgbd not found. The optimized RGB-D stream will not be available unless installed manually."
fi

echo "Installing stretch4_human_pose_estimation and dependencies..."
python3 -m pip install -e .

echo "Installing SAM 3.1..."
if [ -d "$HOME/repos/sam3" ]; then
    python3 -m pip install -e "$HOME/repos/sam3"
    # Fix numpy downgrade caused by sam3 to prevent conflicts with rerun-sdk
    python3 -m pip install "numpy>=2"
else
    echo "Warning: SAM 3.1 not found at ~/repos/sam3. Please clone the repository if you intend to use SAM 3.1."
fi

echo "Downloading all RTMO models..."
python3 -m stretch4_human_pose_estimation.utils.setup_models --size all

echo "Installation complete!"
echo ""
echo "*******************************************************************************"
echo "*                                                                             *"
echo "*                             MANDATORY STEP!                                 *"
echo "*                                                                             *"
echo "*  You MUST log out and log back in (or reboot) for the new hardware          *"
echo "*  permissions to fully take effect. If you do not do this, the system        *"
echo "*  will not detect the NPU/GPU hardware and the models will fail to load!     *"
echo "*                                                                             *"
echo "*******************************************************************************"
echo "To activate the virtual environment for future use, run:"
echo "source $VENV_DIR/bin/activate"
