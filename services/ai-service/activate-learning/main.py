from plugins.quality import Quality
from plugins.uncertainty import Uncertainty
from plugins.difficulty import Calibrated_Sampler
from plugins.extract import FeatsExtraction
from plugins.disversity import DiversitySampler
from ultralytics import YOLO
import glob
import os
import yaml

if __name__ == '__main__':
    model = YOLO('yolov8n.pt').cpu()
    ls = glob.glob('/mnt/d/Personal/Programing/PersonalProjects/data-recall-system/services/central-storage/dataset/human/test/images/*.jpg') # Data 1000 to 100
    with open("setting.yaml", 'r') as stream:
        json_cfg  = yaml.safe_load(stream)
    
    quality = Quality(model, base_momentum=json_cfg['base_momentum'], img_path='/mnt/d/Personal/Programing/PersonalProjects/data-recall-system/services/central-storage/dataset/human/train/images') # Data ground truth
    quality(trainset=json_cfg['trainset'] , out=json_cfg['quality'])

    # # # # plugins
    uncertainty = Uncertainty(model)
    uncertainty(ls, out=json_cfg['uncertainty'])
    
    calibrated_sampler = Calibrated_Sampler(n_images= int(json_cfg['budget_uncertainty']*len(ls)), score_thr=json_cfg['score_thr'], floor_score_thr=json_cfg['floor_score_thr'], category_valid=[0])
    idxs_valid, idxs_non_valid = calibrated_sampler(json_cfg['quality'], json_cfg['uncertainty'], max_entropy_filter=True)

    feats_extraction = FeatsExtraction(model, json_cfg['uncertainty_out'], 'ultralytics/cfg/models/v8/yolov8.yaml')
    K=int(json_cfg['budget_diversity']*int(json_cfg['budget_uncertainty']*len(ls)))
 
    if K == 0 or len(idxs_valid) == 0:
        print('No need to sample')
        print(idxs_valid)
        exit()
    feats_extraction(idxs_valid, ls, out=json_cfg['feat'])
    diversity_sampler = DiversitySampler(K=K)
    sampled_path, unsampled_path = diversity_sampler(feature_file=json_cfg['feat'], uncertainty_file=json_cfg['uncertainty_out'])
    print(sampled_path, unsampled_path)
    