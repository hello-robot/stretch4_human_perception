import argparse
import cv2
import time
import numpy as np
from stretch4_human_pose_estimation import RTMOPipeline
import stretch4_body.subsystem.cameras as cameras

def process_and_display(pipeline: RTMOPipeline, image, window_name="Stereo RTMO Pose Estimation", display_text=None, style="cvpr"):
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

def run_stereo_camera(pipeline: RTMOPipeline, display_text=None, style="cvpr"):
    print("Starting stereo camera stream (left and right)...")
    
    stream = cameras.stream_left_right_camera()
    print("Press 'q' to quit.")
    
    try:
        for frame in stream:
            if frame is None:
                continue
                
            if hasattr(frame, 'left') and hasattr(frame, 'right'):
                if frame.left is not None and hasattr(frame.left, 'image') and \
                   frame.right is not None and hasattr(frame.right, 'image'):
                    left_img = frame.left.image
                    right_img = frame.right.image
                    
                    if left_img is None or right_img is None:
                        continue
                        
                    # Concatenate images horizontally (side by side)
                    if left_img.shape[0] != right_img.shape[0]:
                        right_img = cv2.resize(right_img, (int(right_img.shape[1] * left_img.shape[0] / right_img.shape[0]), left_img.shape[0]))
                    
                    combined_img = np.hstack((left_img, right_img))
                    
                    key = process_and_display(pipeline, combined_img, window_name="Stereo Camera RTMO", display_text=display_text, style=style)
                    
                    if key & 0xFF == ord('q'):
                        break
    finally:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run Stereo RTMO Pose Estimation")
    parser.add_argument("--size", type=str, choices=['t', 's', 'm', 'l'], default='m',
                        help="Size of the RTMO model to run (default: m)")
    parser.add_argument("--device", type=str, choices=["AUTO", "CPU", "NPU", "GPU"], default="AUTO",
                        help="Inference device (default: AUTO)")
    parser.add_argument("--style", type=str, choices=["cvpr", "red_green"], default="cvpr",
                        help="Visualization color style (default: cvpr)")
                        
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

    run_stereo_camera(pipeline, display_text, style=args.style)

if __name__ == "__main__":
    main()
