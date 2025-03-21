import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator #type: ignore

from task_crawl_data.crawl_data import crawl_data
from task_crawl_data.active_learning_inference import active_learning_task
from task_crawl_data.auto_label_inference import auto_label

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
    'target_class': 'human',
    'keywords': ['human', 'person standing', 'people walking'],
    'num_images': 200  # Total images to attempt to download
}

AUTO_LABEL_CONFIG = {
    "example_path": '/central-storage/produced-dataset/human_detections/autolabel_data/examples/examples.jpg',
    # "api_key": os.getenv('GEMINI_API_KEY'),
    "api_key": "AIzaSyDM7E_-Iwz0t38_-w41B2XRAh_S80YoefY",
    "delay": 5,
    "labels_folder": '/central-storage/produced-dataset/human_detections/autolabel_data/processed_labels',
    
    # Model paths for AutoLabelSam2
    "model_paths": {
        "mlp": "/central-storage/produced-dataset/human_detections/autolabel_data/MLP_small_box_w1_fewshot.tar",
        "point_decoder": "/central-storage/produced-dataset/human_detections/autolabel_data/point_decoder_vith.pth",
        "sam2": "/central-storage/produced-dataset/human_detections/autolabel_data/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
}

# Define the DAG
with DAG(
    'crawl_dataset_to_central_storage_pipeline',
    default_args=default_args,
    description='A DAG to run validate data on images from MinIO by active learning and auto label',
    schedule_interval=timedelta(days=1),
    start_date=datetime.now(),
    catchup=False,
    tags=['ai', 'computer-vision', 'active-learning', 'auto-label'],
) as dag:

    # Define the produce data task
    crawl_data_task = PythonOperator(
        task_id='crawl_images',
        python_callable=crawl_data,
        op_kwargs={
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
    
    # Auto label task    
    auto_label_task = PythonOperator(
        task_id='auto_label',
        python_callable=auto_label,
        op_kwargs={
            'auto_label_config': AUTO_LABEL_CONFIG
        },
    )
(   
    crawl_data_task 
    >> active_learning_filter_task
    >> auto_label_task
)