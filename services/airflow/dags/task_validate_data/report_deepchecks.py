import os

def validate_data(ds_repo_path: str, save_path: str = 'ds_val.html', img_ext: str = 'jpeg'):
    from deepchecks.vision import classification_dataset_from_directory
    from deepchecks.vision.suites import train_test_validation
    
    train_ds, test_ds = classification_dataset_from_directory(
        root=os.path.join(ds_repo_path, 'images'), object_type='VisionData',
        image_extension=img_ext
    )
    suite = train_test_validation()
    print("Running data validation test sute")
    result = suite.run(train_ds, test_ds)
    result.save_as_html(save_path)
    print(f'Finish data validation and save report to {save_path}')
    print("This file will also be saved along with the MLflow's training task in the later step")