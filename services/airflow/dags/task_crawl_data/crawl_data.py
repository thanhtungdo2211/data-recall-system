import os
import requests
import time
import random
import logging
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from ai_services.crawl_data.searchByText.search_by_key import search_by_text

def crawl_data(
    processing_config: Dict[str, Any],
    **kwargs) -> Dict[str, Any]:
    """
    Crawl image data based on keywords, download and save to central storage.
    
    Args:
        postgres_config: PostgreSQL configuration
        minio_config: MinIO configuration
        processing_config: Contains output directory and target class
        **kwargs: Additional Airflow context
    
    Returns:
        Dict with stats and paths
    """
    # Extract configuration
    output_dir = processing_config.get('output_dir')
    target_class = processing_config.get('target_class', 'human')
    num_images = processing_config.get('num_images', 200)
    keywords = processing_config.get('keywords', [target_class])
    
    # Ensure output directories exist
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    total_downloaded = 0
    failed_downloads = 0
    image_paths = []
    
    # Process each keyword
    for keyword in keywords:
        logging.info(f"Searching for images with keyword: {keyword}")
        
        try:
            # Get image URLs using the search_by_text function
            image_urls = search_by_text(keyword, num_images)
            logging.info(f"Found {len(image_urls)} URLs for keyword '{keyword}'")
            
            # Download images
            for i, url in enumerate(image_urls):
                try:
                    # Generate a filename with index and keyword
                    filename = f"{keyword}_{total_downloaded + i + 1}.jpg"
                    image_path = os.path.join(images_dir, filename)
                    
                    # Skip if file already exists
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        continue
                    
                    # Download with timeout
                    response = requests.get(url, stream=True, timeout=10)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            response.raw.decode_content = True
                            shutil.copyfileobj(response.raw, f)
                        
                        image_paths.append(image_path)
                        total_downloaded += 1
                        
                        # Log progress periodically
                        if total_downloaded % 10 == 0:
                            logging.info(f"Downloaded {total_downloaded} images so far...")
                    else:
                        failed_downloads += 1
                        
                except Exception as e:
                    logging.error(f"Error downloading image from {url}: {str(e)}")
                    failed_downloads += 1
                    
                # Add random delay between downloads (0.5-2 seconds)
                time.sleep(random.uniform(0.5, 2))
                
        except Exception as e:
            logging.error(f"Error searching for '{keyword}': {str(e)}")
            
    logging.info(f"Crawling complete. Downloaded {total_downloaded} images, {failed_downloads} failed.")
    
    # Return results for next task
    return {
        "images_dir": images_dir,
        "total_downloaded": total_downloaded,
        "image_paths": image_paths,
        "target_class": target_class
    }