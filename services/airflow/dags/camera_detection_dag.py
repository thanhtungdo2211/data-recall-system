import os
from datetime import datetime
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from task_camera_detection.kafka_consumer import consume_from_kafka
from task_camera_detection.minio_storage import store_in_minio
from task_camera_detection.postgres_writer import save_to_postgres

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Load environment variables from .env file (in development)
load_dotenv()

# Configuration
CAMERA_URL = os.getenv('CAMERA_URL', '0')  # Use 0 for webcam, or RTSP URL for network camera
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'best.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS','kafka:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'object-detection-events')
KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'object-detection-consumer-group')

# MinIO configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY','minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'detection-frames')

# PostgreSQL configuration
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

with DAG(
    'object_detection_pipeline',
    default_args=default_args,
    description='Object detection pipeline with YOLOv8, Kafka, MinIO, and PostgreSQL',
    # schedule_interval="0 17 * * *",  # 17:00 UTC = 00:00 GMT+7
    start_date=datetime.now(),  # Bắt đầu từ ngày cụ thể ..
    catchup=False,
    tags=['ai', 'computer-vision', 'yolo'],
) as dag:
    # Task 1: Consume detection events from Kafka
    consume_from_kafka_task = PythonOperator(
        task_id='consume_from_kafka',
        python_callable=consume_from_kafka,
        op_kwargs={
            'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
            'topic': KAFKA_TOPIC,
            'group_id': KAFKA_GROUP_ID
        },
    )
    
    # Task 2: Store frames in MinIO
    store_in_minio_task = PythonOperator(
        task_id='store_in_minio',
        python_callable=store_in_minio,
        op_kwargs={
            'endpoint': MINIO_ENDPOINT,
            'access_key': MINIO_ACCESS_KEY,
            'secret_key': MINIO_SECRET_KEY,
            'bucket_name': MINIO_BUCKET
        },
    )
    
    # Task 3: Save metadata to PostgreSQL
    save_to_postgres_task = PythonOperator(
        task_id='save_to_postgres',
        python_callable=save_to_postgres,
        op_kwargs={
            'host': POSTGRES_HOST,
            'port': POSTGRES_PORT,
            'database': POSTGRES_DB,
            'user': POSTGRES_USER,
            'password': POSTGRES_PASSWORD
        },
    )
    
    # Define task dependencies
    (
        consume_from_kafka_task 
        >> store_in_minio_task 
        >> save_to_postgres_task
    )