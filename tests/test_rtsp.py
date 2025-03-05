from ultralytics import YOLO
import cv2
import time
from torch.serialization import add_safe_globals
import argparse
import numpy as np

# Add YOLO to safe globals for deserialization
add_safe_globals(['nets.nn.YOLO', 'YOLO'])

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Person Detection from RTSP Stream')
    parser.add_argument('--rtsp', type=str, default="rtsp://admin:MQ@20130516@192.168.6.212",
                        help='RTSP camera URL')
    parser.add_argument('--model', type=str, default="yolo-person.pt",
                        help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
                        help='Device to run inference on (cuda:0 or cpu)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Minimum interval between detection attempts (seconds)')
    return parser.parse_args()

def main():
    """Main function to run YOLO detection on RTSP stream"""
    args = parse_args()
    
    # Print configuration
    print(f"[CONFIG] RTSP URL: {args.rtsp}")
    print(f"[CONFIG] YOLO Model: {args.model}")
    print(f"[CONFIG] Confidence threshold: {args.conf}")
    print(f"[CONFIG] Device: {args.device}")
    print(f"[CONFIG] Detection interval: {args.interval}s")
    
    # Load YOLO model
    print("[INFO] Loading YOLO model...")
    try:
        model = YOLO(args.model)
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return
    
    # Open RTSP stream
    print(f"[INFO] Connecting to RTSP stream: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open RTSP stream: {args.rtsp}")
        return
    
    print("[INFO] RTSP stream connected")
    
    # Variables for timing
    last_detection_time = 0
    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("[WARNING] Failed to get frame, retrying in 1 second...")
                time.sleep(1)
                # Try to reconnect
                cap.release()
                cap = cv2.VideoCapture(args.rtsp)
                continue
                
            # Update FPS calculation
            fps_counter += 1
            if time.time() - fps_timer >= 5:  # Calculate FPS every 5 seconds
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
                print(f"[INFO] Current FPS: {fps:.2f}")
                
            # Check if enough time has passed since last detection
            current_time = time.time()
            if current_time - last_detection_time < args.interval:
                continue
                
            # Run detection
            results = model(frame, conf=args.conf, device=args.device, verbose=False)
            
            # Extract detection results
            boxes = []
            confidences = []
            class_names = []
            
            result = results[0]
            if result.boxes:
                for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                        result.boxes.conf.cpu().numpy(),
                                        result.boxes.cls.cpu().numpy().astype(int)):
                    boxes.append(box)
                    confidences.append(conf)
                    class_names.append(result.names[cls])
            
            # Print detection results if people found
            if len(boxes) > 0:
                print("-" * 50)
                print(f"[DETECTION] Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"[DETECTION] Found {len(boxes)} people")
                
                # Print details for each detection
                for i, (box, conf, name) in enumerate(zip(boxes, confidences, class_names)):
                    x1, y1, x2, y2 = box
                    print(f"  Person {i+1}: {name} (conf: {conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                    
                last_detection_time = current_time
                
    except KeyboardInterrupt:
        print("[INFO] Detection stopped by user")
    finally:
        cap.release()
        print("[INFO] Stream closed")

if __name__ == "__main__":
    main()