from minio import Minio
import os
import io
from PIL import Image

def download_image_from_minio(image_url, bucket_name="detection-frames", 
                             minio_endpoint="localhost:9000", 
                             access_key="minioadmin", 
                             secret_key="minioadmin",
                             secure=False):
    """
    Download an image from Minio based on the image URL.
    
    Args:
        image_url (str): The path to the image in Minio
        bucket_name (str): The Minio bucket name
        minio_endpoint (str): Minio server endpoint
        access_key (str): Minio access key
        secret_key (str): Minio secret key
        secure (bool): Whether to use HTTPS
        
    Returns:
        PIL.Image.Image: The downloaded image as a PIL Image object
    """
    # Create Minio client
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    
    # Get object data
    response = client.get_object(bucket_name, image_url)
    
    # Read image data
    image_data = response.read()
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    return image

def save_image_from_minio(image_url, output_path, bucket_name="detection-frames",
                         minio_endpoint="localhost:9000", 
                         access_key="minioadmin", 
                         secret_key="minioadmin",
                         secure=False):
    """
    Download and save an image from Minio to local filesystem.
    
    Args:
        image_url (str): The path to the image in Minio
        output_path (str): Local path to save the image
        bucket_name (str): The Minio bucket name
        minio_endpoint (str): Minio server endpoint
        access_key (str): Minio access key
        secret_key (str): Minio secret key
        secure (bool): Whether to use HTTPS
        
    Returns:
        str: Path to the saved image
    """
    # Create Minio client
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download file directly to the path
    client.fget_object(bucket_name, image_url, output_path)
    
    return output_path