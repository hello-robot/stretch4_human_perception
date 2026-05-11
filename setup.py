from setuptools import setup, find_packages

setup(
    name="stretch4_human_pose_estimation",
    version="0.1.0",
    description="Minimal package for RTMO human pose estimation on Stretch 4.",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'human_pose_estimation_install_dependencies=stretch4_human_pose_estimation.utils.install_deps:main',
            'human_pose_estimation_setup_models=stretch4_human_pose_estimation.utils.setup_models:main',
        ],
    },
    install_requires=[
        "numpy",
        "opencv-python",
        "openvino>=2023.0",
        "torch",
        "torchvision",
        "Pillow",
        "einops",
        "pycocotools",
        "psutil",
        "scikit-image",
        "scikit-learn",
        "decord",
        "mediapipe",
        "pin",
        "rerun-sdk>=0.31.4",
    ],
    python_requires=">=3.12",
)
