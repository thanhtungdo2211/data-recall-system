import logging
import sqlalchemy
from sqlalchemy.orm import Session
from datetime import datetime
import json
from common.db_utils import DetectionEvents, Detections

def save_to_postgres(host, port, database, user, password, **context):
    """Save detection metadata to PostgreSQL.
    
    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: PostgreSQL database name
        user: PostgreSQL username
        password: PostgreSQL password
        context: Airflow task context
    """
    try:
        # Create database connection
        engine = sqlalchemy.create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        
        # Get events with MinIO URLs from previous task
        task_instance = context['ti']
        events_with_urls = task_instance.xcom_pull(
            task_ids='store_in_minio',
            key='events_with_urls'
        )
        
        if not events_with_urls:
            logging.warning("No events to save to PostgreSQL")
            return
            
        # Create session and save data
        with Session(engine) as session:
            saved_count = 0
            
            for event_data in events_with_urls:
                try:
                    # Create DetectionEvents record
                    detection_event = DetectionEvents(
                        camera_id=event_data.get('camera_id', 'unknown'),
                        timestamp=datetime.fromtimestamp(event_data.get('timestamp', 0)),
                        detection_count=event_data.get('detection_count', 0),
                        bucket_name=event_data.get('bucket_name', ''),
                        image_url=event_data.get('image_url', '')
                    )
                    session.add(detection_event)
                    session.flush()  # Generate ID
                    
                    # Create Detection records
                    for detection in event_data.get('detections', []):
                        detection_record = Detections(
                            event_id=detection_event.id,
                            class_id=detection.get('class_id', 0),
                            class_name=detection.get('class_name', 'unknown'),
                            confidence=detection.get('confidence', 0.0),
                            box_x1=detection.get('box', {}).get('x1', 0.0),
                            box_y1=detection.get('box', {}).get('y1', 0.0),
                            box_x2=detection.get('box', {}).get('x2', 0.0),
                            box_y2=detection.get('box', {}).get('y2', 0.0)
                        )
                        session.add(detection_record)
                        
                    saved_count += 1
                    
                except Exception as e:
                    logging.error(f"Error saving event to PostgreSQL: {str(e)}")
                    continue
                    
            # Commit all changes
            session.commit()
            logging.info(f"Saved {saved_count} events to PostgreSQL")
            
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {str(e)}")
        raise