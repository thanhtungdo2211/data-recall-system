import os
import logging

from typing import Dict, Any, Optional

# Helper function for collating batches in DeepChecks format
def deepchecks_collate(batch):
    """Process batch to deepchecks format."""
    imgs, labels = zip(*batch)
    return {'images': list(imgs), 'labels': list(labels)}
        

def export_deepchecks_report(
    train_dataset_path: str,
    test_dataset_path: Optional[str] = None,
    output_report_path: str = '/central-storage/reports/data_validation_report.html',
    img_extension: str = 'jpg',
    batch_size: int = 32,
    **kwargs
) -> Dict[str, Any]:
    """
    Run DeepChecks validation on image datasets and generate HTML reports.
    
    Args:
        train_dataset_path: Path to training dataset
        test_dataset_path: Path to test dataset (if None, will use train_dataset_path)
        output_report_path: Path to save the HTML report
        img_extension: Image file extension (default: 'jpg')
        batch_size: Batch size for data loading (default: 32)
        **kwargs: Additional Airflow context
    
    Returns:
        dict: Dictionary with report paths and validation details
    """
    import torch
    from deepchecks.vision import VisionData
    from deepchecks.vision.vision_data.simple_classification_data import SimpleClassificationDataset
    from deepchecks.vision.checks import ImagePropertyDrift
    from deepchecks.vision.suites import train_test_validation
    from torch.utils.data import DataLoader
        
    try:
        logging.info(f"Starting data validation - Train: {train_dataset_path}, Test: {test_dataset_path or train_dataset_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        
        # Default test path to train path if not provided
        if not test_dataset_path:
            test_dataset_path = train_dataset_path
            logging.info(f"No test dataset provided, using training dataset for both")
        
        # Configuration for data loading
        num_workers = 0  # Use 0 to avoid issues in container environments
        shuffle = True
        pin_memory = True
        
        # Load training data
        logging.info(f"Loading training dataset from {train_dataset_path}")
        train_dataset = SimpleClassificationDataset(
            root=train_dataset_path, 
            image_extension=img_extension
        )
        
        # Create DataLoader and VisionData for training set
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=deepchecks_collate, 
            pin_memory=pin_memory, 
            generator=torch.Generator()
        )
        
        train_visiondata = VisionData(
            batch_loader=train_dataloader, 
            label_map=train_dataset.reverse_classes_map,
            task_type='classification'
        )
        
        # Load test data
        logging.info(f"Loading test dataset from {test_dataset_path}")
        test_dataset = SimpleClassificationDataset(
            root=test_dataset_path, 
            image_extension=img_extension
        )
        
        # Create DataLoader and VisionData for test set
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=deepchecks_collate, 
            pin_memory=pin_memory, 
            generator=torch.Generator()
        )
        
        test_visiondata = VisionData(
            batch_loader=test_dataloader, 
            label_map=test_dataset.reverse_classes_map,
            task_type='classification'
        )
        
        # Run individual checks
        logging.info("Running ImagePropertyDrift check")
        drift_check = ImagePropertyDrift()
        drift_result = drift_check.run(
            train_dataset=train_visiondata, 
            test_dataset=test_visiondata
        )
        
        drift_report_path = output_report_path.replace('.html', '_drift.html')
        drift_result.save_as_html(drift_report_path)
        logging.info(f"Saved drift report to {drift_report_path}")
        
        # Run full validation suite
        logging.info("Running complete validation suite")
        suite = train_test_validation()
        suite_result = suite.run(train_visiondata, test_visiondata)
        suite_result.save_as_html(output_report_path)
        logging.info(f"Saved full validation report to {output_report_path}")
        
        return {
            "drift_report_path": drift_report_path,
            "full_report_path": output_report_path,
            "train_class_count": len(train_dataset.reverse_classes_map),
            "test_class_count": len(test_dataset.reverse_classes_map),
            "train_image_count": len(train_dataset),
            "test_image_count": len(test_dataset),
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Data validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "error_message": str(e),
            "train_dataset_path": train_dataset_path,
            "test_dataset_path": test_dataset_path
        }