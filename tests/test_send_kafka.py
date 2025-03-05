import time
import json
import base64
import uuid
import random
import numpy as np
import cv2
from kafka import KafkaProducer
from tqdm import tqdm

KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
KAFKA_TOPIC = "object-detection-events"

class_names = ["person", "car", "truck", "bicycle", "motorcycle", "bus"]

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
        
        # Create message in the exact format requested
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

def generate_random_frame(width=640, height=480):
    """Generate a random image frame."""
    # Create a random colored image
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Convert to JPEG bytes
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer.tobytes()

def generate_random_detection():
    """Generate random detection data."""
    class_id = random.randint(0, len(class_names) - 1)
    
    # Create random bounding box coordinates
    x1 = random.uniform(0, 500)
    y1 = random.uniform(0, 400)
    x2 = x1 + random.uniform(50, 150)
    y2 = y1 + random.uniform(50, 150)
    
    return {
        'class_id': int(class_id),
        'class_name': class_names[class_id],
        'confidence': float(random.uniform(0.25, 0.99)),
        'box': {
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2)
        }
    }

def generate_random_detections(min_count=1, max_count=5):
    """Generate a random number of detections."""
    count = random.randint(min_count, max_count)
    return [generate_random_detection() for _ in range(count)]

def generate_random_camera_id():
    """Generate a random camera ID."""
    camera_locations = ["entrance", "parking", "lobby", "hallway", "exit"]
    camera_numbers = list(range(1, 6))
    return f"{random.choice(camera_locations)}-cam{random.choice(camera_numbers)}"

def run_test(num_messages=100):
    """Send random detection messages to Kafka."""
    producer = DetectionProducer(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)
    
    print(f"Sending {num_messages} random detection messages to Kafka...")
    
    for i in tqdm(range(num_messages)):
        # Generate random data
        frame_bytes = generate_random_frame()
        detections = generate_random_detections()
        timestamp = time.time() - random.uniform(0, 3600)  # Random time within the last hour
        camera_id = generate_random_camera_id()
        
        # Send to Kafka
        success = producer.send_detection(
            frame_bytes, 
            detections, 
            timestamp, 
            camera_id
        )
        
        if not success:
            print(f"Failed to send message {i+1}")
        
        # Small delay to avoid overwhelming the broker
        time.sleep(0.1)
    
    # Allow time for callbacks to complete
    time.sleep(2)
    print("Test completed!")

if __name__ == "__main__":
    run_test(100)