import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator

from task_validate_data.active_learning_inference import active_learning_task
from task_validate_data.data_produce import produce_yolo_dataset
# from task_validate_data.report_deepchecks import validate_data
from task_validate_data.auto_label_inference import auto_label

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
    
    # Auto label task    
    auto_label_task = PythonOperator(
        task_id='auto_label',
        python_callable=auto_label,
        op_kwargs={
            'auto_label_config': AUTO_LABEL_CONFIG
        },
    )
    
#    # Deepchecks task and autolabel task will execute in parallel
#     deep_checks_task = PythonOperator(
#         task_id='deep_checks',
#         python_callable=validate_data,
#         op_kwargs={
#             'ds_repo_path': '/central-storage/produced-dataset/human_detections/dataset',
#             'save_path': '/central-storage/produced-dataset/human_detections/dataset/ds_val.html',
#             'img_ext': 'jpeg'
#         },
#     )
    
#   # Define the task dependencies
# (   
#     produce_data_task 
#     >> active_learning_filter_task
#     >> [deep_checks_task, auto_label_task]
# )
(   
    produce_data_task 
    >> active_learning_filter_task
    >> auto_label_task
)