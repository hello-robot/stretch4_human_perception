import argparse
import cv2
import os
import glob
import time
from stretch4_human_pose_estimation import RTMOPipeline

def process_and_display(pipeline: RTMOPipeline, image, window_name="RTMO Pose Estimation", display_text=None, style="cvpr"):
    start_time = time.time()
    results = pipeline.predict(image)
    inference_time = time.time() - start_time
    
    vis_image = pipeline.visualize(image, results, style=style)
    
    cv2.putText(vis_image, f"Inference: {inference_time*1000:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    if display_text:
        y0, dy = 60, 25
        for i, line in enumerate(display_text):
            y = y0 + i*dy
            cv2.putText(vis_image, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    cv2.imshow(window_name, vis_image)
    return cv2.waitKey(1)

def run_camera(pipeline: RTMOPipeline, camera_name: str, display_text=None, style="cvpr"):
    import stretch4_body.subsystem.cameras as cameras
    
    print(f"Starting camera stream from '{camera_name}'...")
    
    if camera_name == "left":
        stream = cameras.stream_left_camera()
    elif camera_name == "right":
        stream = cameras.stream_right_camera()
    elif camera_name == "center":
        stream = cameras.stream_center_camera()
    elif camera_name == "gripper":
        stream = cameras.stream_gripper_camera()
    else:
        raise ValueError(f"Unknown camera: {camera_name}")

    print("Press 'q' to quit.")
    
    try:
        for frame in stream:
            if frame is None:
                continue
                
            if hasattr(frame, 'image'):
                img = frame.image
            elif hasattr(frame, 'left'):
                img = frame.left.image
            else:
                continue
                
            key = process_and_display(pipeline, img, window_name=f"Live Camera: {camera_name}", display_text=display_text, style=style)
            
            if key & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

def run_directory(pipeline: RTMOPipeline, directory: str, display_text=None, style="cvpr"):
    print(f"Processing images in directory: {directory}")
    print("Press any key to advance to the next image. Press 'q' to quit.")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_paths.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"No valid images found in {directory}.")
        return

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue
            
        start_time = time.time()
        results = pipeline.predict(img)
        inference_time = time.time() - start_time
        
        vis_image = pipeline.visualize(img, results, style=style)
        
        cv2.putText(vis_image, f"File: {os.path.basename(img_path)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Inference: {inference_time*1000:.1f}ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        if display_text:
            y0, dy = 100, 25
            for i, line in enumerate(display_text):
                y = y0 + i*dy
                cv2.putText(vis_image, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Directory Evaluation", vis_image)
        key = cv2.waitKey(0)
        
        if key & 0xFF == ord('q'):
            print("Quitting directory evaluation.")
            break

    cv2.destroyAllWindows()
    print("Finished processing directory.")

def main():
    parser = argparse.ArgumentParser(description="Run RTMO Pose Estimation")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l'], default='m',
                        help="Size of the RTMO model to run (default: m)")
    parser.add_argument("--device", type=str, choices=["AUTO", "CPU", "NPU", "GPU"], default="AUTO",
                        help="Inference device (default: AUTO)")
    parser.add_argument("--style", type=str, choices=["cvpr", "red_green"], default="cvpr",
                        help="Visualization color style (default: cvpr)")
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--camera", type=str, choices=["left", "right", "center", "gripper"], default="left",
                        help="Camera to use for live evaluation (default: left)")
    input_group.add_argument("--dir", type=str,
                        help="Directory containing images to evaluate sequentially")
                        
    args = parser.parse_args()
    
    print(f"Initializing RTMO Pipeline (Size: {args.size}, Device: {args.device})...")
    try:
        pipeline = RTMOPipeline(size=args.size, device=args.device)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    display_text = [
        f"Model: RTMO-{args.size}",
        f"Device: {args.device}"
    ]

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory '{args.dir}' does not exist.")
            return
        run_directory(pipeline, args.dir, display_text, style=args.style)
    else:
        run_camera(pipeline, args.camera, display_text, style=args.style)

if __name__ == "__main__":
    main()
