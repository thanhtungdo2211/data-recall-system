import os
import sys
sys.path.insert(1, '.')
sys.path.insert(1, 'services/airflow/dags/common')

import io
import uuid
import glob
import logging
import sqlalchemy
from sqlalchemy.orm import Session
from minio import Minio
from datetime import datetime
from PIL import Image

from db_utils import DetectionEvents, Detections, Base

def upload_yolo_dataset(
    yolo_dataset_path,
    minio_endpoint,
    minio_access_key,
    minio_secret_key,
    minio_bucket_name,
    db_host,
    db_port,
    db_name,
    db_user,
    db_password,
):
    """
    Upload YOLO format dataset to MinIO and store metadata in PostgreSQL.
    
    Args:
        yolo_dataset_path: Path to YOLO dataset folder containing 'images' and 'labels' subfolders
        minio_endpoint: MinIO server endpoint
        minio_access_key: MinIO access key
        minio_secret_key: MinIO secret key
        minio_bucket_name: MinIO bucket name
        db_host: PostgreSQL host
        db_port: PostgreSQL port
        db_name: PostgreSQL database name
        db_user: PostgreSQL username
        db_password: PostgreSQL password
    """
    logging.info(f"Processing YOLO dataset from: {yolo_dataset_path}")
    
    # Check if path exists and has required structure
    images_path = os.path.join(yolo_dataset_path, "images")
    labels_path = os.path.join(yolo_dataset_path, "labels")
    
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise ValueError(f"Dataset at {yolo_dataset_path} does not have required 'images' and 'labels' folders")
    
    # Initialize MinIO client
    minio_client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    
    # Create bucket if it doesn't exist
    if not minio_client.bucket_exists(minio_bucket_name):
        minio_client.make_bucket(minio_bucket_name)
        logging.info(f"Created bucket: {minio_bucket_name}")
    
    # Initialize database connection
    engine = sqlalchemy.create_engine(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    # Generate a batch ID for this upload
    batch_id = str(uuid.uuid4())
    date_prefix = datetime.now().strftime('%Y/%m/%d')
    upload_timestamp = datetime.now()
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                  glob.glob(os.path.join(images_path, "*.jpeg")) + \
                  glob.glob(os.path.join(images_path, "*.png"))
    
    # YOLO class names mapping (example - adjust according to your dataset)
    class_mapping = {
        0: "human",
        1: "human"
        # 2: "car",
        # # Add more classes as needed
    }
    
    # Process each image and its label
    uploaded_count = 0
    with Session(engine) as session:
        for image_file in image_files:
            try:
                # Generate simulated camera_id and event_id
                camera_id = f"camera_{uuid.uuid4().hex[:8]}"
                event_id = str(uuid.uuid4())
                
                # Get base filename without extension
                base_filename = os.path.splitext(os.path.basename(image_file))[0]
                
                # Check if corresponding label file exists
                label_file = os.path.join(labels_path, f"{base_filename}.txt")
                if not os.path.exists(label_file):
                    logging.warning(f"No label file found for {image_file}. Skipping.")
                    continue
                
                # Read image and get dimensions
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
                    # Convert to bytes for uploading
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
                    img_bytes = img_byte_arr.getvalue()
                
                # Upload to MinIO
                object_name = f"{date_prefix}/{batch_id}/{camera_id}/{event_id}.jpg"
                minio_client.put_object(
                    minio_bucket_name,
                    object_name,
                    io.BytesIO(img_bytes),
                    length=len(img_bytes),
                    content_type='image/jpeg'
                )
                
                # Parse label file and count detections
                detections = []
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert normalized coordinates to absolute coordinates
                            x1 = (x_center - width/2) 
                            y1 = (y_center - height/2)
                            x2 = (x_center + width/2)
                            y2 = (y_center + height/2)
                            
                            detections.append({
                                "class_id": class_id,
                                "class_name": class_mapping.get(class_id, f"class_{class_id}"),
                                "confidence": 1.0,  # YOLO labels don't include confidence, default to 1.0
                                "box": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2
                                }
                            })
                
                # Create database entry for the event
                detection_event = DetectionEvents(
                    camera_id=camera_id,
                    timestamp=upload_timestamp,
                    detection_count=len(detections),
                    bucket_name=minio_bucket_name,
                    image_url=object_name
                )
                
                session.add(detection_event)
                session.flush()  # Generate ID
                
                # Create database entries for detections
                for detection in detections:
                    detection_record = Detections(
                        event_id=detection_event.id,
                        class_id=detection["class_id"],
                        class_name=detection["class_name"],
                        confidence=detection["confidence"],
                        box_x1=detection["box"]["x1"],
                        box_y1=detection["box"]["y1"],
                        box_x2=detection["box"]["x2"],
                        box_y2=detection["box"]["y2"]
                    )
                    session.add(detection_record)
                
                uploaded_count += 1
                logging.info(f"Processed image {uploaded_count}/{len(image_files)}: {image_file}")
                
            except Exception as e:
                logging.error(f"Error processing {image_file}: {str(e)}")
                continue
                
        # Commit all changes
        session.commit()
    
    logging.info(f"Successfully uploaded {uploaded_count} images with labels to MinIO and PostgreSQL")
    return {
        "processed_images": uploaded_count,
        "batch_id": batch_id,
        "timestamp": upload_timestamp.isoformat()
    }

if __name__ == "__main__":
    # Example usage
    result = upload_yolo_dataset(
        yolo_dataset_path="/mnt/d/Personal/Programing/PersonalProjects/data-recall-system/services/central-storage/dataset/human/test",
        minio_endpoint="localhost:9000" ,
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket_name="detection-frames",
        db_host="localhost",
        db_port="5432",
        db_name="postgres",
        db_user="postgres",
        db_password="postgres"
    )
    print(f"Upload results: {result}")