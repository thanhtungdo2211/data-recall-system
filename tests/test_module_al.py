import glob
import yaml
import sys
import os

# dags_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(dags_folder)
sys.path.append('/opt/airflow/dags/task_validate_data/active_learning')

from task_validate_data.active_learning.plugins.quality import Quality
from task_validate_data.active_learning.plugins.uncertainty import Uncertainty
from task_validate_data.active_learning.plugins.difficulty import Calibrated_Sampler
from task_validate_data.active_learning.plugins.extract import FeatsExtraction
from task_validate_data.active_learning.plugins.disversity import DiversitySampler
from task_validate_data.active_learning.ultralytics import YOLO

def run_active_learning(config_path, dataset_images_path, ground_truth_images_path, model_path='yolov8n.pt'):
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
    model = YOLO(model_path).cpu()
    ls = glob.glob(f"{dataset_images_path}/*.jpg")
    
    with open(config_path, 'r') as stream:
        json_cfg = yaml.safe_load(stream)
    
    quality = Quality(model, base_momentum=json_cfg['base_momentum'], img_path=ground_truth_images_path)
    quality(trainset=json_cfg['trainset'], out=json_cfg['quality'])

    # Run plugins
    uncertainty = Uncertainty(model)
    uncertainty(ls, out=json_cfg['uncertainty'])
    
    calibrated_sampler = Calibrated_Sampler(
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
