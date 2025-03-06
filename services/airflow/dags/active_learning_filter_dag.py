from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import io
import os
import pandas as pd
from minio import Minio
import tempfile

# Minio client setup
def get_minio_client():
    minio_client = Minio(
        endpoint=os.environ.get('MINIO_ENDPOINT', 'minio:9000'),
        access_key=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
        secret_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
        secure=False
    )
    return minio_client

# Extract image URLs from PostgreSQL
def extract_image_urls(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    # Modify this query based on your database schema
    query = """
    SELECT id, image_url, metadata 
    FROM images 
    WHERE processed = false
    LIMIT 100;
    """
    
    df = pg_hook.get_pandas_df(query)
    
    # Pass dataframe to the next task
    kwargs['ti'].xcom_push(key='image_urls_data', value=df.to_dict(orient='records'))
    
    return "Extracted URLs successfully"

# Download images from MinIO
def download_images_from_minio(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='image_urls_data', task_ids='extract_image_urls')
    
    minio_client = get_minio_client()
    downloaded_data = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for item in data:
            image_url = item['image_url']
            # Parse bucket and object name from URL
            # This is a simple example - adjust based on your URL format
            parts = image_url.replace("minio://", "").split('/', 1)
            if len(parts) == 2:
                bucket_name, object_name = parts
                
                # Local file path to save the image
                local_path = os.path.join(temp_dir, os.path.basename(object_name))
                
                # Download the object
                try:
                    minio_client.fget_object(bucket_name, object_name, local_path)
                    item['local_path'] = local_path
                    downloaded_data.append(item)
                except Exception as e:
                    print(f"Error downloading {image_url}: {e}")
    
    kwargs['ti'].xcom_push(key='downloaded_data', value=downloaded_data)
    
    return "Downloaded images successfully"

# Run active learning module
def run_active_learning(**kwargs):
    ti = kwargs['ti']
    downloaded_data = ti.xcom_pull(key='downloaded_data', task_ids='download_images_from_minio')
    
    # TODO: Implement your active learning logic here
    # For example:
    # from your_module import active_learning
    # results = active_learning(downloaded_data)
    
    # Placeholder for results
    results = [{"id": item["id"], "processed": True, "result": "placeholder"} for item in downloaded_data]
    
    kwargs['ti'].xcom_push(key='processed_results', value=results)
    
    return "active learning completed"

# Update database with results
def update_database(**kwargs):
    ti = kwargs['ti']
    results = ti.xcom_pull(key='processed_results', task_ids='run_active_learning')
    
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    for result in results:
        # Update the database to mark images as processed
        # Modify this query based on your database schema
        query = f"""
        UPDATE images 
        SET processed = true, result = '{result["result"]}' 
        WHERE id = {result["id"]};
        """
        pg_hook.run(query)
    
    return "Database updated successfully"

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
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

# Define the tasks
extract_urls_task = PythonOperator(
    task_id='extract_image_urls',
    python_callable=extract_image_urls,
    provide_context=True,
    dag=dag,
)

download_images_task = PythonOperator(
    task_id='download_images_from_minio',
    python_callable=download_images_from_minio,
    provide_context=True,
    dag=dag,
)

active_learning_task = PythonOperator(
    task_id='run_active_learning',
    python_callable=run_active_learning,
    provide_context=True,
    dag=dag,
)

update_db_task = PythonOperator(
    task_id='update_database',
    python_callable=update_database,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
extract_urls_task >> download_images_task >> active_learning_task >> update_db_task