import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

def auto_label(auto_label_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Airflow task to automatically label images by comparing them to an example.
    
    Args:
        auto_label_config: Dictionary containing configurations:
            - example_path: Path to the example image
            - api_key: API key for image comparison service
            - delay: Time to wait between processing images (seconds)
            - model_paths: Dict with paths to model checkpoints
        **kwargs: Airflow context variables
        
    Returns:
        dict: Results including count of processed images and labels
    """
    # Import task-specific dependencies inside function to avoid scheduler issues
    sys.path.append('/opt/airflow/dags/ai_services/auto_label')
    try:
        from ai_services.auto_label.detect import AutoLabelSam2
        from ai_services.auto_label.compare import CompareImage
    except ImportError as e:
        logging.error(f"Failed to import required modules: {e}")
        return {"error": str(e), "labeled_count": 0}
    
    # Extract configuration values with defaults
    example_path = auto_label_config.get('example_path')
    labels_folder = auto_label_config.get('labels_folder')
    api_key = auto_label_config.get('api_key', os.environ.get("GEMINI_API_KEY"))
    delay = auto_label_config.get('delay', 2)
    model_paths = auto_label_config.get('model_paths', {})
    
    # Log configuration
    logging.info(f"Auto-label configuration: example_path={example_path}, labels_folder={labels_folder}")
    
    # Validate required parameters
    if not example_path:
        error_msg = "Missing required parameter: example_path"
        logging.error(error_msg)
        return {"error": error_msg, "labeled_count": 0}
    
    # Get task instance for XCom
    ti = kwargs.get('ti')
    
    # Pull filtered images from active learning task if available
    if ti:
        active_learning_result = ti.xcom_pull(task_ids='active_learning_filter')
        if active_learning_result and "sample_paths" in active_learning_result:
            filtered_image_paths = active_learning_result["sample_paths"]
            logging.info(f"Retrieved {len(filtered_image_paths)} filtered images from active learning task")
    
    # Ensure output directory exists
    os.makedirs(labels_folder, exist_ok=True)
    
    try:
        # Initialize models - with error handling
        detector = AutoLabelSam2(model_paths=model_paths)
        comparator = CompareImage(api_key)
        
        # Get example bounding box
        example_bbox = get_example_bbox(example_path)
        
        labeled_count = 0
        
        # Process each image
        for img_path in filtered_image_paths:
            logging.info(f"Processing {os.path.basename(img_path)}...")
            
            try:
                # Extract and filter bounding boxes
                bboxes = detector(example_path, example_bbox, img_path)
                filtered_bboxes = comparator(example_path, img_path, bboxes)
                
                # Create label file if matches found
                if len(filtered_bboxes) > 0:
                    filename = Path(img_path).stem
                    create_labels_file(filtered_bboxes, labels_folder, filename, img_path)
                    labeled_count += 1
                else:
                    logging.info(f"No matches found for {os.path.basename(img_path)}")
                
            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")
                continue
                
            # Brief pause to avoid resource contention
            time.sleep(delay)
            
        return {
            "labeled_count": labeled_count,
            "total_processed": len(filtered_image_paths),
            "labels_folder": labels_folder
        }
        
    except Exception as e:
        logging.error(f"Auto-labeling failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "labeled_count": 0}
    
def create_labels_file(bboxes, output_dir: str, filename: str, img_path: str = None) -> None:
    """Creates a YOLO format labels file from bounding boxes.
    
    Args:
        bboxes: Bounding boxes in format [x, y, width, height]
        output_dir: Directory to save labels
        filename: Name of the label file (without extension)
        img_path: Path to the image for normalization (if bboxes are not normalized)
    """
    import cv2
    labels_path = os.path.join(output_dir, f"{filename}.txt")
    
    # If image path is provided, normalize the bounding boxes
    if img_path:
        img = cv2.imread(img_path)
        if img is not None:
            img_height, img_width = img.shape[:2]
            
            with open(labels_path, 'w') as f:
                for bbox in bboxes:
                    # Extract values
                    x, y, width, height = bbox.tolist()
                    
                    # Convert to center coordinates
                    x_center = (x + width/2) / img_width
                    y_center = (y + height/2) / img_height
                    
                    # Normalize width and height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    # Format as "class_id x_center y_center width height"
                    line = f"0 {x_center} {y_center} {norm_width} {norm_height}\n"
                    f.write(line)
            
            print(f"Successfully created normalized label: {labels_path}")
    else:
        # Assume bboxes are already in YOLO format
        with open(labels_path, 'w') as f:
            for bbox in bboxes:
                # Format as "class_id x_center y_center width height"
                line = f"0 {' '.join(map(str, bbox.tolist()))}\n"
                f.write(line)
        
        print(f"Successfully created label: {labels_path}")


def get_example_bbox(image_path: str, target_size: int = 1024):
    """Extract a bounding box covering the entire example image, scaled appropriately."""
    import cv2
    import torch
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Calculate dimensions while preserving aspect ratio
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return torch.Tensor([[1., 1., new_w, new_h]])