from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import pandas as pd
import io
import glob
import shutil
from PIL import Image
from minio import Minio
from airflow.providers.postgres.hooks.postgres import PostgresHook

from task_validate_data.active_learning_inference import run_active_learning

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

load_dotenv()

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'detection-frames')

# Minio client setup
def get_minio_client():
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def produce_yolo_dataset(**kwargs):
    """
    Produce YOLO format dataset from frames with human detections.
    
    1. Query DB for human detection frames
    2. Download images from MinIO
    3. Create YOLO format labels
    4. Save as YOLO dataset structure
    """
    # Set up output directories
    base_output_dir = '/central-storage/produced-dataset/human_detections'
    images_dir = os.path.join(base_output_dir, 'images')
    labels_dir = os.path.join(base_output_dir, 'labels')
    
    # Create directories if they don't exist
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
    WHERE d.class_name = 'human'
    AND de.validated = false
    GROUP BY de.id, de.bucket_name, de.image_url, de.detection_count
    LIMIT 100;
    """
    
    # Execute query and get results
    connection = pg_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("No human detections found in the database")
        cursor.close()
        connection.close()
        return {"processed_images": 0, "dataset_path": base_output_dir}
    
    # Get column names
    column_names = [desc[0] for desc in cursor.description]
    
    # Initialize variables
    minio_client = get_minio_client()
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
    
    # Mark these items as validated in the database
    # if processed_count > 0:
    #     event_ids = [event['id'] for event in downloaded_events]
    #     id_list = ','.join(str(id) for id in event_ids)
    #     update_query = f"UPDATE detection_events SET validated = true WHERE id IN ({id_list})"
    #     cursor.execute(update_query)
    #     connection.commit()
    #     print(f"Marked {len(event_ids)} events as validated in the database")
    
    # Close DB connection
    cursor.close()
    connection.close()
    
    print(f"Completed processing {processed_count} images with labels")
    return {
        "processed_images": processed_count,
        "dataset_path": base_output_dir
    }

def active_learning_task(**kwargs):
    """
    Apply active learning to filter the most informative images from the dataset.
    
    Uses the dataset produced by produce_yolo_dataset and runs active learning
    algorithms to select the most valuable samples for training.
    """
    # Get the path to the dataset from the previous task
    ti = kwargs['ti']
    prev_task_result = ti.xcom_pull(task_ids='produce_yolo_dataset')
    base_output_dir = prev_task_result["dataset_path"]
    
    # Skip if no images were processed
    if prev_task_result["processed_images"] == 0:
        print("No images were processed in the previous task. Skipping active learning.")
        return {
            "selected_images": 0,
            "selected_paths": []
        }
    
    # Set up paths
    images_dir = os.path.join(base_output_dir, 'images')
    
    # Set up active learning configuration
    config_path = '/opt/airflow/dags/task_validate_data/activate_learning/setting.yaml'
    ground_truth_path = '/central-storage/produced-dataset/human_detections/train/images'
    model_path = '/central-storage/models/yolov8n.pt'
    
    # Create output directories for active learning results
    al_output_dir = '/central-storage/produced-dataset/human_detections/active_learning_output'
    os.makedirs(al_output_dir, exist_ok=True)
    
    # Set up active learning output directories in YAML config paths
    
    print(f"Running active learning on {prev_task_result['processed_images']} images")
    
    # Run active learning algorithm
    try:
        sampled_path, unsampled_path = run_active_learning(
            config_path=config_path,
            dataset_images_path=images_dir,
            ground_truth_images_path=ground_truth_path,
            model_path=model_path
        )
        
        if not sampled_path:
            print("Active learning did not select any images")
            return {
                "selected_images": 0,
                "selected_paths": []
            }
        
        # Read the selected image paths
        with open(sampled_path, 'r') as f:
            selected_images = f.read().strip().split('\n')
        
        print(f"Active learning selected {len(selected_images)} images")
        
        # Copy selected images to a filtered directory
        filtered_dir = os.path.join(al_output_dir, 'filtered_images')
        os.makedirs(filtered_dir, exist_ok=True)
        
        # Copy the selected images to the filtered directory
        for img_path in selected_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(filtered_dir, img_name)
            shutil.copy(img_path, dest_path)
            
            # Also copy corresponding label file if exists
            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label = os.path.join(base_output_dir, 'labels', label_name)
            if os.path.exists(src_label):
                dest_label = os.path.join(filtered_dir, label_name)
                shutil.copy(src_label, dest_label)
        
        # Return the results
        return {
            "selected_images": len(selected_images),
            "selected_paths": selected_images,
            "filtered_dir": filtered_dir
        }

    except Exception as e:
        print(f"Error in active learning: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "selected_images": 0,
            "selected_paths": [],
            "error": str(e)
        }

# Define the DAG
dag = DAG(
    'active_learning_filter',
    default_args=default_args,
    description='A DAG to run active learning on images from MinIO',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Define the produce data task
produce_data_task = PythonOperator(
    task_id='produce_yolo_dataset',
    python_callable=produce_yolo_dataset,
    provide_context=True,
    dag=dag,
)

# Define the active learning task
active_learning_filter_task = PythonOperator(
    task_id='active_learning_filter',
    python_callable=active_learning_task,
    provide_context=True,
    dag=dag,
)

# Define the task dependencies
(
    produce_data_task 
    >> active_learning_filter_task
)