import os
import sys
import subprocess
import argparse

def install_system_dependencies(assume_yes=False):
    print("Checking for Intel hardware for specific driver installations...")
    try:
        lscpu_out = subprocess.check_output(["lscpu"]).decode("utf-8")
        if "GenuineIntel" in lscpu_out.lower() or "intel" in lscpu_out.lower():
            ans = "y"
            if not assume_yes:
                ans = input("Intel CPU detected. Request sudo privileges to install Intel NPU/GPU drivers? [y/N]: ").strip().lower()
            if ans != "y":
                print("Skipping Intel-specific system/driver configurations.")
                return

            print("Requesting sudo privileges for execution...")
            subprocess.call("sudo apt update || true", shell=True)
            subprocess.call("sudo apt install -y intel-opencl-icd libze1 libze-intel-gpu1 || echo 'Warning: Failed to install Intel packages'", shell=True)
            subprocess.call("sudo snap install intel-npu-driver --beta || echo 'Warning: Failed to install snap driver'", shell=True)

            lib_path = "/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu"
            if os.path.exists(lib_path):
                subprocess.call(f"echo '{lib_path}' | sudo tee /etc/ld.so.conf.d/intel-npu.conf > /dev/null", shell=True)
                subprocess.call("sudo ldconfig", shell=True)

            rc = subprocess.call("sudo usermod -a -G render $USER || true", shell=True)
            subprocess.call("sudo udevadm control --reload-rules && sudo udevadm trigger || true", shell=True)

            print("Intel hardware drivers installed successfully.")
            print("\n*******************************************************************************")
            print("*                             MANDATORY STEP!                                 *")
            print("*  You MUST log out and log back in (or reboot) for the new hardware          *")
            print("*  permissions to fully take effect. If you do not do this, the system        *")
            print("*  will not detect the NPU/GPU hardware and the models will fail to load!     *")
            print("*******************************************************************************\n")
        else:
            print("Non-Intel CPU detected. Skipping Intel-specific hardware driver installations.")
    except Exception as e:
        print(f"Failed to check or install system dependencies: {e}")

def run_setup_models_script():
    parser = argparse.ArgumentParser(description="Download RTMO Models")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l', 'all'], default='all',
                        help="Size of the RTMO model to download ('t', 's', 'm', 'l' or 'all'). Default is 'all'.")
    args = parser.parse_args()
    run_setup_models(args.size)

def run_setup_models(size="all"):
    # Call setup_models.py located right next to this file
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    setup_models_script = os.path.join(pkg_dir, "setup_models.py")

    if os.path.exists(setup_models_script):
        print(f"Executing {setup_models_script} to download models...")
        try:
            subprocess.check_call([sys.executable, setup_models_script, "--size", size], cwd=pkg_dir)
            print("Setup complete! Models are ready for use.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing setup_models.py: {e}")
            sys.exit(1)
    else:
        print(f"Error: {setup_models_script} not found in {pkg_dir}!")
        print(f"Please ensure setup_models.py exists alongside install_deps.py.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Install Stretch 4 Human Pose Estimation Dependencies and Models")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l', 'all'], default='all',
                        help="Size of the RTMO model to download ('t', 's', 'm', 'l' or 'all'). Default is 'all'.")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-confirm all installation prompts.")
    args = parser.parse_args()

    install_system_dependencies(assume_yes=args.yes)
    run_setup_models(args.size)

if __name__ == "__main__":
    main()
