def validate_data(train_path: str, test_path: str = None, save_path: str = 'ds_val.html', img_ext: str = 'jpg'):
    """
    Run DeepChecks validation on image datasets from separate directories.
    
    Args:
        train_path: Path to directory containing training images
        test_path: Path to directory containing test images (if None, will use train_path)
        save_path: Path to save the validation HTML report
        img_ext: Image extension (e.g., 'jpeg', 'jpg', 'png')
    """
    from deepchecks.vision import classification_dataset_from_directory
    from deepchecks.vision.suites import train_test_validation
    
    # Use the same directory for both if test_path is not provided
    test_path = test_path or train_path
    
    print(f"Loading training data from: {train_path}")
    train_ds = classification_dataset_from_directory(
        root=train_path, 
        object_type='VisionData',
        image_extension=img_ext
    )
    
    print(f"Loading test data from: {test_path}")
    test_ds = classification_dataset_from_directory(
        root=test_path, 
        object_type='VisionData',
        image_extension=img_ext
    )
    
    suite = train_test_validation()
    print("Running data validation test suite")
    result = suite.run(train_ds, test_ds)
    result.save_as_html(save_path)
    print(f'Finished data validation and saved report to {save_path}')
    
    return {
        "report_path": save_path,
        "train_images_count": len(train_ds),
        "test_images_count": len(test_ds)
    }

# Example usage
validate_data(train_path='/central-storage/dataset/human/train', test_path='/central-storageproduced-dataset/human_detections/dataset', save_path='/test/ds_val.html', img_ext='jpg')