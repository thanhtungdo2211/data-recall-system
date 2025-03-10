from ultralytics import YOLO
import cv2
import time
import json
import base64
from kafka import KafkaProducer
from torch.serialization import add_safe_globals
import uuid

# Add YOLO class to safe globals to allow deserialization
add_safe_globals(['nets.nn.YOLO', 'YOLO'])

CAMERA_URL = "rtsp://admin:MQ@20130516@192.168.6.212"
YOLO_MODEL_PATH = "yolo-person.pt"
CONFIDENCE_THRESHOLD = 0.25
KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
KAFKA_TOPIC = "object-detection-events"
MIN_TIME_BETWEEN_EVENTS = 5.0  # Min time between sending events (in seconds)
JPEG_QUALITY = 50  # JPEG compression quality (0-100)

class ObjectDetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, device='cuda:0', verbose=False)
        
        detections = []
        
        result = results[0]
        
        if result.boxes:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                
                detection = {
                    'class_id': int(class_id),
                    'class_name': result.names[class_id],
                    'confidence': float(confidence),
                    'box': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                    }
                }
                detections.append(detection)
    
        return detections

class DetectionProducer:
    def __init__(self, bootstrap_servers, topic):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            # Optimize batch and compression
            batch_size=16384,  # 16KB
            linger_ms=50,  # Wait up to 50ms to accumulate batch
            compression_type='gzip',  # Compress with gzip'
            max_request_size=10485760  # 1MB
        )
        self.topic = topic
        
    def send_detection(self, frame_bytes, detections, timestamp, camera_id="main"):
        # Create unique ID for event
        event_id = str(uuid.uuid4())
        
        # Only send metadata and image
        data_to_send = {
            'event_id': event_id,
            'camera_id': camera_id,
            'timestamp': timestamp,
            'detection_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            'frame': base64.b64encode(frame_bytes).decode('utf-8'),
            'detections': detections,
            'detection_count': len(detections)
        }
        
        # Use key to ensure events from the same camera go to the same partition
        future = self.producer.send(self.topic, key=camera_id.encode('utf-8'), value=data_to_send)
        
        try:
            # Don't wait for result - continue processing next frame
            # future.get() is called asynchronously in callback
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            return True
        except Exception as e:
            print(f"Error sending data: {str(e)}")
            return False
            
    def _on_send_success(self, record_metadata):
        print(f"Success: {record_metadata.topic} [partition {record_metadata.partition}]")
        
    def _on_send_error(self, exc):
        print(f"Error sending to Kafka: {str(exc)}")

def main():
    camera = cv2.VideoCapture(CAMERA_URL)
    if not camera.isOpened():
        raise Exception(f"Failed to open camera: {CAMERA_URL}")

    detector = ObjectDetector(YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD)
    producer = DetectionProducer(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)
    
    last_sent_time = 0
    
    print("Starting person detection loop...")

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera, retrying...")
            time.sleep(1)
            continue
        
        current_time = time.time()
        time_since_last_event = current_time - last_sent_time
        
        # Only detect objects if enough time has passed since last event
        if time_since_last_event < MIN_TIME_BETWEEN_EVENTS:
            time.sleep(0.01)  # Reduce CPU load
            continue
            
        # Detect objects in frame
        detections = detector.detect(frame)
        
        # Only send event when people are detected
        if len(detections) > 0:
            print(f"Detected {len(detections)} people at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
            
            # Convert frame to JPEG with controlled quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            jpg_as_bytes = buffer.tobytes()
            
            # Send event to Kafka
            if producer.send_detection(jpg_as_bytes, detections, current_time):
                last_sent_time = current_time
                
        # Reduce CPU load
        time.sleep(0.01)
        
if __name__ == "__main__":
    main()