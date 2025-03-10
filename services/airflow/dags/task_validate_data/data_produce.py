# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.postgres.hooks.postgres import PostgresHook
# from datetime import datetime, timedelta
# import io
# import os
# import pandas as pd
# from minio import Minio
# import tempfile

# # Minio client setup
# minio_client = Minio(
#     endpoint=os.environ.get('MINIO_ENDPOINT', 'minio:9000'),
#     access_key=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
#     secret_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
#     secure=False
# )

# # Extract image URLs from PostgreSQL where class_name is 'human'
# def extract_image_urls(**kwargs):
#     pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
#     # Query that joins detection_events with detections to find images with 'human' class
#     query = """
#     SELECT DISTINCT de.id, de.bucket_name, de.image_url
#     FROM detection_events de
#     JOIN detections d ON de.id = d.event_id
#     WHERE d.class_name = 'human'
#     AND de.validated = false
#     LIMIT 100;
#     """
    
#     df = pg_hook.get_pandas_df(query)
    
#     # Pass dataframe to the next task
#     kwargs['ti'].xcom_push(key='image_urls_data', value=df.to_dict(orient='records'))
    
#     return f"Extracted {len(df)} image URLs with human detections"

# # Download images from MinIO to central_storage
# def download_images_from_minio(**kwargs):
#     ti = kwargs['ti']
#     data = ti.xcom_pull(key='image_urls_data', task_ids='extract_image_urls')
    
#     # Create central_storage directory if it doesn't exist
#     central_storage_dir = os.environ.get('CENTRAL_STORAGE_PATH', '/opt/airflow/central_storage/human_images')
#     os.makedirs(central_storage_dir, exist_ok=True)
    
#     downloaded_count = 0
#     failed_count = 0
    
#     for item in data:
#         bucket_name = item['bucket_name']
#         image_url = item['image_url']
        
#         # Extract object name from the URL
#         object_name = image_url.split('/')[-1]
        
#         # Local file path to save the image
#         local_path = os.path.join(central_storage_dir, f"{item['id']}_{object_name}")
        
#         # Download the object
#         try:
#             minio_client.fget_object(bucket_name, object_name, local_path)
#             downloaded_count += 1
#         except Exception as e:
#             print(f"Error downloading {image_url} from bucket {bucket_name}: {e}")
#             failed_count += 1
    
#     return f"Downloaded {downloaded_count} images with human detections. Failed: {failed_count}"

# def produce_data(**kwargs):
#     extract_image_urls(**kwargs)
#     download_images_from_minio(**kwargs)
#     return "Data production completed"