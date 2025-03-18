import glob
import yaml
import sys
import os

def run_active_learning(config_path, dataset_images_path, ground_truth_images_path, model_path):
    """
    Run active learning pipeline to select the most informative samples.
    
    Args:
        config_path (str): Path to the setting.yaml configuration file
        dataset_images_path (str): Path to the test images to select from
        ground_truth_images_path (str): Path to the training images for quality assessment
        model_path (str): Path to the YOLOv8 model weights
        
    Returns:
        tuple: (sampled_path, unsampled_path) - Paths to files containing sampled and unsampled images
    """
    sys.path.append('/opt/airflow/dags/task_validate_data/active_learning')
    from ai_services.active_learning.plugins.quality import Quality
    from ai_services.active_learning.plugins.uncertainty import Uncertainty
    from ai_services.active_learning.plugins.difficulty import CalibratedSampler 
    from ai_services.active_learning.plugins.extract import FeatsExtraction
    from ai_services.active_learning.plugins.disversity import DiversitySampler
    from ai_services.active_learning.ultralytics import YOLO

    model = YOLO(model_path).cpu()
    ls = glob.glob(f"{dataset_images_path}/*.jpg")

    with open(config_path, 'r') as stream:
        json_cfg = yaml.safe_load(stream)
    
    quality = Quality(model, base_momentum=json_cfg['base_momentum'], img_path=ground_truth_images_path)
    quality(nc = json_cfg['nc'], names =  json_cfg['names'], out=json_cfg['quality'])

    # Run plugins
    uncertainty = Uncertainty(model)
    uncertainty(ls, out=json_cfg['uncertainty'])
    
    calibrated_sampler = CalibratedSampler(
        n_images=int(json_cfg['budget_uncertainty']*len(ls)),
        score_thr=json_cfg['score_thr'],
        floor_score_thr=json_cfg['floor_score_thr'],
        category_valid=[0]
    )
    idxs_valid, idxs_non_valid = calibrated_sampler(
        json_cfg['quality'], 
        json_cfg['uncertainty'], 
        max_entropy_filter=True
    )

    feats_extraction = FeatsExtraction(
        model, 
        json_cfg['uncertainty_out'], 
        'ultralytics/cfg/models/v8/yolov8.yaml'
    )
    K = int(json_cfg['budget_diversity'] * int(json_cfg['budget_uncertainty'] * len(ls)))
 
    if K == 0 or len(idxs_valid) == 0:
        print('No need to sample')
        print(idxs_valid)
        return None, None
        
    feats_extraction(idxs_valid, ls, out=json_cfg['feat'])
    diversity_sampler = DiversitySampler(K=K)
    sampled_path, unsampled_path = diversity_sampler(
        feature_file=json_cfg['feat'], 
        uncertainty_file=json_cfg['uncertainty_out']
    )
    
    return sampled_path, unsampled_path

def active_learning_task(active_learning_config, **kwargs):
    """
    Apply active learning to filter the most informative images for autolabeling.
    
    Uses the dataset produced by produce_yolo_dataset and runs active learning
    algorithms to select the most valuable samples for training.
    """
    # Get the path to the dataset from the previous task
    ti = kwargs['ti']
    prev_task_result = ti.xcom_pull(task_ids='produce_yolo_dataset')
    base_output_dir = prev_task_result["dataset_path"]
    
    # Skip if no images were processed
    if prev_task_result["processed_images"] == 0:
        print("No images were processed in the previous task. Skipping active learning.")
        return {"sample_count": 0, "sample_paths": []}
    
    # Set up paths
    images_dir = os.path.join(base_output_dir, 'images')
    
    # Use configuration from parameters
    config_path = active_learning_config['config_path']
    ground_truth_path = active_learning_config['ground_truth_path']
    model_path = active_learning_config['model_path']
    
    print(f"Running active learning on {prev_task_result['processed_images']} images")
    
    # Run active learning algorithm
    try:
        sampled_paths, unsampled_paths = run_active_learning(
            config_path=config_path,
            dataset_images_path=images_dir,
            ground_truth_images_path=ground_truth_path,
            model_path=model_path
        )
        
        if not sampled_paths:
            print("Active learning did not select any images")
            return {"sample_count": 0, "sample_paths": []}
    
        
        print(f"Active learning selected {len(sampled_paths)} images for autolabeling")
        
        print(f"Sampled images path: {sampled_paths}") # For debugging
        
        # Return only the necessary information for autolabeling
        return {
            "sample_count": len(sampled_paths),
            "sample_paths": sampled_paths,  
        }

    except Exception as e:
        print(f"Error in active learning: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "sample_count": 0, 
            "sample_paths": [],
            "error": str(e)
        }
    