#!/usr/bin/env python3
import os
import sys
import argparse
import urllib.request

MODELS = {
    "RTMO-t": {
        "files": {
            "rtmo-t.onnx": "https://huggingface.co/bukuroo/RTMO-ONNX/resolve/main/rtmo-t.onnx"
        },
        "target": "rtmo-t.onnx",
        "description": "Real-Time Multi-person one-stage Pose estimation model from OpenMMLab (Tiny)."
    },
    "RTMO-s": {
        "files": {
            "rtmo-s.onnx": "https://huggingface.co/bukuroo/RTMO-ONNX/resolve/main/rtmo-s.onnx"
        },
        "target": "rtmo-s.onnx",
        "description": "Real-Time Multi-person one-stage Pose estimation model from OpenMMLab (Small)."
    },
    "RTMO-m": {
        "files": {
            "rtmo-m.onnx": "https://huggingface.co/bukuroo/RTMO-ONNX/resolve/main/rtmo-m.onnx"
        },
        "target": "rtmo-m.onnx",
        "description": "Real-Time Multi-person one-stage Pose estimation model from OpenMMLab (Medium)."
    },
    "RTMO-l": {
        "files": {
            "rtmo-l.onnx": "https://huggingface.co/bukuroo/RTMO-ONNX/resolve/main/rtmo-l.onnx"
        },
        "target": "rtmo-l.onnx",
        "description": "Real-Time Multi-person one-stage Pose estimation model from OpenMMLab (Large)."
    },
    "SAM 3.1": {
        "format": "sam3_setup",
        "target": "sam3.1_setup_complete",
        "description": "SAM 3.1 segmentation model (Requires Hugging Face login)."
    },
    "MediaPipe-Pose": {
        "format": "direct",
        "files": {
            "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        },
        "target": "pose_landmarker_full.task",
        "description": "Google MediaPipe Pose Landmarker."
    },
    "MediaPipe-Hand": {
        "format": "direct",
        "files": {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        "target": "hand_landmarker.task",
        "description": "Google MediaPipe Hand Landmarker."
    },
    "MediaPipe-Face": {
        "format": "direct",
        "files": {
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        },
        "target": "face_landmarker.task",
        "description": "Google MediaPipe Face Landmarker."
    },
    "MediaPipe-Holistic": {
        "format": "direct",
        "files": {
            "holistic_landmarker.task": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
        },
        "target": "holistic_landmarker.task",
        "description": "Google MediaPipe Holistic Landmarker."
    }
}

class DownloadProgressBar:
    def __init__(self, prefix="Downloading"):
        self.prefix = prefix
        self.last_percent = -1

    def __call__(self, count, block_size, total_size):
        if total_size <= 0:
            return
        percent = int(count * block_size * 100 / total_size)
        if percent > 100:
            percent = 100
        if percent > self.last_percent:
            sys.stdout.write(f"\r{self.prefix}... {percent}%")
            sys.stdout.flush()
            self.last_percent = percent
        if percent == 100 and self.last_percent != 100:
            sys.stdout.write("\n")
            self.last_percent = 100

def main():
    parser = argparse.ArgumentParser(description="Download RTMO Models for Stretch 4 Human Pose Estimation")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l', 'all'], default='m',
                        help="Size of the RTMO model to download ('t', 's', 'm', 'l' or 'all'). Default is 'm'.")
    parser.add_argument("--sam3", action="store_true", help="Include setup instructions for SAM 3.1")
    parser.add_argument("--mediapipe", action="store_true", help="Download the MediaPipe Pose model")
    args = parser.parse_args()

    models_to_download = {}
    if args.size == 'all':
        for k, v in MODELS.items():
            if k not in ["SAM 3.1", "MediaPipe-Pose"]:
                models_to_download[k] = v
    else:
        model_key = f"RTMO-{args.size}"
        models_to_download[model_key] = MODELS[model_key]

    if args.sam3 or args.size == 'all':
        models_to_download["SAM 3.1"] = MODELS["SAM 3.1"]
        
    if args.mediapipe or args.size == 'all':
        models_to_download["MediaPipe-Pose"] = MODELS["MediaPipe-Pose"]
        models_to_download["MediaPipe-Hand"] = MODELS["MediaPipe-Hand"]
        models_to_download["MediaPipe-Face"] = MODELS["MediaPipe-Face"]
        models_to_download["MediaPipe-Holistic"] = MODELS["MediaPipe-Holistic"]

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Checking models in {models_dir}...\n")

    for name, info in models_to_download.items():
        if info.get("format") == "sam3_setup":
            target_path = os.path.join(models_dir, info["target"])
            if os.path.exists(target_path):
                print(f"[{name}] Setup already complete, skipping.")
                continue
            
            print(f"[{name}] To use SAM 3.1, ensure you have signed up for access at https://huggingface.co/facebook/sam3.1")
            print(f"[{name}] You must authenticate via `huggingface-cli login` on your system.")
            print(f"[{name}] We assume the official sam3 repository is cloned at ~/repos/sam3")
            print(f"[{name}] Run `pip install -e ~/repos/sam3` to install the package if not already done.")
            
            with open(target_path, 'w') as f:
                f.write("Setup complete instructions provided.")
            print(f"[{name}] Finished instruction step.\n")
            continue

        target_path = info["target"]
        
        all_files_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in info.get("files", {}).keys())
        if all_files_exist:
            print(f"[{name}] Models already exist, skipping.")
            continue
            
        for filename, file_url in info.get("files", {}).items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                print(f"[{name}] {filename} already exists, skipping.")
                continue
            
            print(f"[{name}] Downloading {filename}...")
            progress = DownloadProgressBar(prefix=f"Downloading {filename}")
            urllib.request.urlretrieve(file_url, filepath, reporthook=progress)
            print() # new line

    print("Setup complete! Models are ready for use.")

if __name__ == "__main__":
    main()
