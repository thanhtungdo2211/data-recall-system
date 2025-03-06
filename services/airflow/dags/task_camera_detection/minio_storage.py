from minio import Minio
import io
import logging
import uuid
from datetime import datetime

def store_in_minio(endpoint, access_key, secret_key, bucket_name, **context):
    """Store received frames in MinIO object storage.
    
    Args:
        endpoint: MinIO server endpoint
        access_key: MinIO access key
        secret_key: MinIO secret key
        bucket_name: MinIO bucket name
        context: Airflow task context
    
    Returns:
        List of events with updated MinIO info
    """
    try:
        # Initialize MinIO client
        minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        
        # Create bucket if it doesn't exist
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logging.info(f"Created bucket: {bucket_name}")
        
        # Get processed events from previous task
        task_instance = context['ti']
        processed_events = task_instance.xcom_pull(
            task_ids='consume_from_kafka',
            key='processed_events'
        )
        
        if not processed_events:
            logging.warning("No events to process")
            return []
            
        batch_id = task_instance.xcom_pull(
            task_ids='consume_from_kafka',
            key='batch_id'
        )
        
        # Store frames in MinIO and update events with URLs
        events_with_urls = []
        date_prefix = datetime.now().strftime('%Y/%m/%d')
        
        for event in processed_events:
            if 'frame_bytes' in event and event['frame_bytes']:
                # Create unique object name
                object_name = f"{date_prefix}/{batch_id}/{event['camera_id']}/{event['event_id']}.jpg"
                
                # Upload frame to MinIO
                minio_client.put_object(
                    bucket_name,
                    object_name,
                    io.BytesIO(event['frame_bytes']),
                    length=len(event['frame_bytes']),
                    content_type='image/jpeg'
                )
                
                # Add MinIO info to event
                event_with_url = event.copy()
                event_with_url.pop('frame_bytes', None)  # Remove binary data
                event_with_url['bucket_name'] = bucket_name
                event_with_url['image_url'] = object_name
                
                events_with_urls.append(event_with_url)
                logging.info(f"Stored frame in MinIO: {object_name}")
        
        # Pass the updated events to the next task
        task_instance.xcom_push(key='events_with_urls', value=events_with_urls)
        return events_with_urls
        
    except Exception as e:
        logging.error(f"Error storing in MinIO: {str(e)}")
        raise