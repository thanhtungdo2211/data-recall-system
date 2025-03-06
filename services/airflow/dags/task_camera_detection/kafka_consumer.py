from datetime import datetime
import time
from kafka import KafkaConsumer
import json
import base64
import logging

class DetectionConsumer:
    def __init__(self, bootstrap_servers, topic, group_id, batch_size=100, max_total_messages=None, timeout_ms=60000):
        """Initialize Kafka consumer for object detection events.
        
        Args:
            bootstrap_servers: Kafka servers list
            topic: Kafka topic to subscribe to
            group_id: Consumer group ID
            batch_size: Number of messages to consume in one batch
            max_total_messages: Maximum total messages to consume (None = unlimited)
            timeout_ms: Timeout in milliseconds to wait for messages
        """
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,  # Tự động commit offset sau khi xử lý
            auto_commit_interval_ms=5000,  # Commit mỗi 5 giây
            consumer_timeout_ms=timeout_ms,
            max_poll_records=batch_size,
            fetch_max_bytes=52428800  # 50MB
        )
        self.batch_size = batch_size
        self.max_total_messages = max_total_messages

    
    def consume(self, callback=None):
        """Consume messages from Kafka topic in batches.
        
        Args:
            callback: Optional function to call after each batch
            
        Returns:
            List of processed detection events
        """
        processed_events = []
        message_count = 0
        total_messages = 0
        batch_number = 0
        
        logging.info(f"Starting to consume messages from Kafka")
        
        while True:
            # Reset batch counter
            message_count = 0
            batch_processed = []
            batch_start = time.time()
            
            # Process a batch of messages
            for message in self.consumer:
                try:
                    detection_data = message.value
                    processed_event = self.process_detection(detection_data)
                    if processed_event:
                        batch_processed.append(processed_event)
                        message_count += 1
                        total_messages += 1
                        
                    # Stop after processing batch_size messages
                    if message_count >= self.batch_size:
                        logging.info(f"Reached batch size ({self.batch_size}), processing batch")
                        break
                        
                    # Stop if we've reached max total messages
                    if self.max_total_messages and total_messages >= self.max_total_messages:
                        logging.info(f"Reached maximum total messages ({self.max_total_messages})")
                        break
                        
                except Exception as e:
                    logging.error(f"Error processing message: {str(e)}")
                    continue
            
            # Add batch to results
            if batch_processed:
                processed_events.extend(batch_processed)
                batch_number += 1
                batch_time = time.time() - batch_start
                logging.info(f"Batch {batch_number}: Processed {len(batch_processed)} messages in {batch_time:.2f}s")
                
                # Call callback function if provided
                if callback and callable(callback):
                    callback(batch_processed)
            else:
                # No messages in this batch, we've consumed all available messages
                logging.info("No more messages to consume")
                break
                
            # Check for maximum total messages
            if self.max_total_messages and total_messages >= self.max_total_messages:
                logging.info(f"Reached maximum total messages ({total_messages}/{self.max_total_messages})")
                break
        
        self.consumer.close()
        logging.info(f"Consumed {len(processed_events)} messages in {batch_number} batches")
        return processed_events
    
    def process_detection(self, detection_data):
        """Process detection event.

        Args:
            detection_data: Dictionary with detection results
        
        Returns:
            Processed detection event
        """
        try:
            # Extrac base infformation
            event_id = detection_data.get('event_id')
            camera_id = detection_data.get('camera_id', 'unknown')
            timestamp = detection_data.get('timestamp')
            detection_count = detection_data.get('detection_count', 0)
            detections = detection_data.get('detections', [])
            
            # Get frame bytes from base64
            frame_bytes = base64.b64decode(detection_data.get('frame', ''))
            
            return {
                'event_id': event_id,
                'camera_id': camera_id,
                'timestamp': timestamp,
                'detection_count': detection_count,
                'detections': detections,
                'frame_bytes': frame_bytes
            }
        
        except Exception as e:
            logging.error(f"Error in process_detection: {str(e)}")
            return None
    
def consume_from_kafka(bootstrap_servers, topic, group_id, batch_size=100, max_runtime=3600, **context):
    """Consume messages from Kafka topic in batches.
    
    Args:
        bootstrap_servers: Kafka servers list
        topic: Kafka topic to subscribe to
        group_id: Consumer group ID
        batch_size: Size of each batch
        max_runtime: Maximum runtime in seconds (default: 1 hour)
        context: Airflow task context
    
    Returns:
        Total number of messages processed
    """
    logging.info(f"Starting Kafka consumption from {topic} on {bootstrap_servers}")
    
    # Initialize Kafka Consumer and process messages
    consumer = DetectionConsumer(bootstrap_servers, topic, group_id, batch_size=batch_size)
    processed_events = consumer.consume()
    
    # Push results to XCom for next tasks
    task_instance = context['ti']
    task_instance.xcom_push(key='processed_events', value=processed_events)
    task_instance.xcom_push(key='batch_id', value=context['execution_date'].strftime('%Y%m%d%H%M%S'))
    
    return len(processed_events)