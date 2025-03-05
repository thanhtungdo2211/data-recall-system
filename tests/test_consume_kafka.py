from datetime import datetime
from kafka import KafkaConsumer
import json
import base64
import logging

class DetectionConsumer:
    def __init__(self, bootstrap_servers, topic, group_id, max_messages=10, timeout_ms=60000):
        """Initialize Kafka consumer for object detection events.
        
        Args:
            bootstrap_servers: Kafka servers list
            topic: Kafka topic to subscribe to
            group_id: Consumer group ID
            max_messages: Maximum number of messages to consume per task run
            timeout_ms: Timeout in milliseconds to wait for messages
        """
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=timeout_ms,
            max_poll_records=max_messages,
            fetch_max_bytes=52428800  # 50MB max fetch to handle large messages with images
        )
        self.max_messages = max_messages
    
    def consume(self):
        """Consume messages from Kafka and process detection data.
        
        Returns:
            List of processed detection events
        """
        processed_events = []
        message_count = 0
        
        logging.info(f"Starting to consume messages from Kafka")
        
        for message in self.consumer:
            try:
                detection_data = message.value
                processed_event = self.process_detection(detection_data)
                if processed_event:
                    processed_events.append(processed_event)
                    message_count += 1
                    logging.info(f"Processed message {message_count} of {self.max_messages}")
                    
                # Stop after processing max_messages
                if message_count >= self.max_messages:
                    logging.info(f"Reached maximum messages ({self.max_messages}), stopping consumption")
                    break
                    
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")
                continue
        
        self.consumer.close()
        logging.info(f"Consumed {len(processed_events)} messages")
        return processed_events
    
    def process_detection(self, detection_data):
        """Process the detection data received from Kafka.
        
        Args:
            detection_data: Dictionary with detection results
            
        Returns:
            Dictionary with processed data for next tasks
        """
        try:
            # Extract base information
            event_id = detection_data.get('event_id')
            camera_id = detection_data.get('camera_id', 'unknown')
            timestamp = detection_data.get('timestamp')
            detection_time = detection_data.get('detection_time')
            detections = detection_data.get('detections', [])
            detection_count = detection_data.get('detection_count', 0)
            
            # Get frame bytes from base64
            frame_bytes = base64.b64decode(detection_data.get('frame', ''))
            
            return {
                'event_id': event_id,
                'camera_id': camera_id,
                'timestamp': timestamp,
                'detection_time': detection_time,
                'detections': detections,
                'detection_count': detection_count,
                'frame_bytes': frame_bytes
            }
        except Exception as e:
            logging.error(f"Error in process_detection: {str(e)}")
            return None

def consume_from_kafka(bootstrap_servers, topic, group_id, max_messages=10, **context):
    """Airflow task to consume detection data from Kafka.
    
    Args:
        bootstrap_servers: Kafka servers
        topic: Kafka topic
        group_id: Consumer group ID
        max_messages: Maximum number of messages to consume
        context: Airflow task context
        
    Returns:
        List of processed events for downstream tasks
    """
    logging.info(f"Starting Kafka consumption from {topic} on {bootstrap_servers}")
    
    # Get the execution date and create batch ID
    if 'execution_date' in context:
        # When running in Airflow
        execution_date = context['execution_date']
    else:
        # When running as standalone script
        execution_date = datetime.now()
        
    batch_id = execution_date.strftime('%Y%m%d%H%M%S')
    
    try:
        # Create consumer and process messages
        consumer = DetectionConsumer(bootstrap_servers, topic, group_id, max_messages)
        processed_events = consumer.consume()
        
        if not processed_events:
            logging.info("No messages were consumed from Kafka")
            return []
            
        print(len(processed_events)) 

        logging.info(f"Successfully processed {len(processed_events)} detection events")
        return processed_events
        
    except Exception as e:
        logging.error(f"Error consuming from Kafka: {str(e)}")
        raise
    
    finally:
        logging.info("Finished Kafka consumption")  
        consumer.consumer.close()
    
if __name__ == '__main__':
    # Test the Kafka consumer
    bootstrap_servers = 'localhost:9094'
    topic = 'object-detection-events'
    group_id = 'object-detection-consumer-group'
    max_messages = 1
    
    consume_from_kafka(bootstrap_servers, topic, group_id, max_messages)