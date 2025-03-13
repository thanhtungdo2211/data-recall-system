import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator

from task_validate_data.active_learning_inference import active_learning_task
from task_validate_data.data_produce import produce_yolo_dataset

# Load environment variables
load_dotenv()

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Configuration
# MinIO configuration
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'minio:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
    'bucket_name': os.getenv('MINIO_BUCKET', 'detection-frames')
}

# PostgreSQL configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'postgres'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

# Active Learning configuration
ACTIVE_LEARNING_CONFIG = {
    'config_path': '/central-storage/produced-dataset/human_detections/active_learning_data/setting.yaml',
    'ground_truth_path': '/central-storage/produced-dataset/human_detections/active_learning_data/ground_truth/images',
    'model_path': '/central-storage/models/yolov8n.pt'
}

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    'output_dir': '/central-storage/produced-dataset/human_detections/dataset',
    'target_class': 'human'
}

# Define the DAG
with DAG(
    'validate_data_from_central_storage_pipeline',
    default_args=default_args,
    description='A DAG to run validate data on images from MinIO by active learning and auto label',
    schedule_interval=timedelta(days=1),
    start_date=datetime.now(),
    catchup=False,
    tags=['ai', 'computer-vision', 'active-learning', 'auto-label'],
) as dag:

    # Define the produce data task
    produce_data_task = PythonOperator(
        task_id='produce_yolo_dataset',
        python_callable=produce_yolo_dataset,
        op_kwargs={
            'postgres_config': POSTGRES_CONFIG,
            'minio_config': MINIO_CONFIG,
            'processing_config': DATA_PROCESSING_CONFIG
        },
    )

    # Define the active learning task
    active_learning_filter_task = PythonOperator(
        task_id='active_learning_filter',
        python_callable=active_learning_task,
        op_kwargs={
            'active_learning_config': ACTIVE_LEARNING_CONFIG
        },
    )

    # Define the task dependencies
(   
    produce_data_task 
    >> active_learning_filter_task
)