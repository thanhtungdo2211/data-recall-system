import os
from dotenv import load_dotenv

from airflow.providers.postgres.hooks.postgres import PostgresHook


# Minio client setup
def get_minio_client(config):
    """Create a MinIO client from configuration"""
    if not config or not all(key in config for key in ['endpoint', 'access_key', 'secret_key']):
        raise ValueError("MinIO configuration missing required fields: endpoint, access_key, secret_key")
    from minio import Minio  
    
    return Minio(
        endpoint=config['endpoint'],
        access_key=config['access_key'],
        secret_key=config['secret_key'],
        secure=False
    )
    
def produce_yolo_dataset(postgres_config, minio_config, processing_config, **kwargs):
    """
    Produce YOLO format dataset from frames with human detections.
    
    1. Query DB for human detection frames
    2. Download images from MinIO
    3. Create YOLO format labels
    4. Save as YOLO dataset structure
    """
    # Set up output directories
    base_output_dir = processing_config['output_dir']
    images_dir = os.path.join(base_output_dir, 'images')
    labels_dir = os.path.join(base_output_dir, 'labels')
    target_class = processing_config['target_class']
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get PostgreSQL connection
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    # Query to get detection events with human class and their bounding boxes
    query = """
    SELECT 
        de.id, 
        de.bucket_name, 
        de.image_url, 
        de.detection_count,
        array_agg(d.class_id) as class_ids,
        array_agg(d.class_name) as class_names,
        array_agg(d.confidence) as confidences,
        array_agg(d.box_x1) as box_x1s,
        array_agg(d.box_y1) as box_y1s,
        array_agg(d.box_x2) as box_x2s,
        array_agg(d.box_y2) as box_y2s
    FROM detection_events de
    JOIN detections d ON de.id = d.event_id
    WHERE d.class_name = %s
    AND de.validated = false
    GROUP BY de.id, de.bucket_name, de.image_url, de.detection_count;
    """
    
    # Execute query and get results
    connection = pg_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(query, (target_class,))
    rows = cursor.fetchall()
    
    if not rows:
        print("No human detections found in the database")
        cursor.close()
        connection.close()
        return {"processed_images": 0, "dataset_path": base_output_dir}
    
    # Get column names
    column_names = [desc[0] for desc in cursor.description]
    
    # Initialize variables
    minio_client = get_minio_client(minio_config)
    downloaded_events = []
    processed_count = 0
    
    print(f"Found {len(rows)} detection events with human class")
    
    # Step 1: Download all images from MinIO
    for row in rows:
        event = dict(zip(column_names, row))
        try:
            # Debug info
            print(f"Processing event {event['id']} from {event['bucket_name']}/{event['image_url']}")
            
            # Set up image path
            image_filename = f"{event['id']}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # Download directly to disk
            minio_client.fget_object(
                event['bucket_name'],
                event['image_url'], 
                image_path
            )
            
            downloaded_events.append(event)
            print(f"Downloaded image for event {event['id']}")
            
        except Exception as e:
            print(f"Error downloading event {event['id']}: {str(e)}")
    
    print(f"Successfully downloaded {len(downloaded_events)} images")
    
    # Step 2: Create label files for successfully downloaded images
    for event in downloaded_events:
        try:
            label_filename = f"{event['id']}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for i in range(len(event['class_ids'])):
                    # Get bounding box coordinates
                    x1 = float(event['box_x1s'][i])
                    y1 = float(event['box_y1s'][i])
                    x2 = float(event['box_x2s'][i])
                    y2 = float(event['box_y2s'][i])
                    class_id = int(event['class_ids'][i])
                    
                    # Convert to YOLO format: class_id, x_center, y_center, width, height
                    # Note: Assuming the coordinates are already normalized (0-1)
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    
                    # Write in YOLO format
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            processed_count += 1
            print(f"Created label file for event {event['id']}")
            
        except Exception as e:
            print(f"Error creating label for event {event['id']}: {str(e)}")
    
    # Close DB connection
    cursor.close()
    connection.close()
    
    print(f"Completed processing {processed_count} images with labels")
    return {
        "processed_images": processed_count,
        "dataset_path": base_output_dir
    }