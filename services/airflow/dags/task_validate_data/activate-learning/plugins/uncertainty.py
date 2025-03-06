import os
import cv2
import json
import torch 
from collections import OrderedDict

class Uncertainty():

    def __init__(self, model):
        self.model = model
    

    def create_json_inference(self, batch_bboxes, batch_cls_scores , batch_labels, batch_cls_uncertainty, info_meta, json_file):

        json_content = []
        if os.path.exists(json_file):    
            with open(json_file, 'r') as f:
                json_content = json.load(f)
        

        for idx, bboxes in enumerate(batch_bboxes):
            for ids, each_bbox in enumerate(bboxes):
                item = {
                    'bbox' : each_bbox.tolist(),
                    'category_id' : int(batch_labels[idx][ids]),
                    'cls_uncertainty': batch_cls_uncertainty[idx][ids].tolist(),
                    'file_name': info_meta['name'][idx],
                    'image_id': info_meta['id'][idx],
                    'width': info_meta['width'][idx],
                    'height': info_meta['height'][idx],
                    'score': float(batch_cls_scores[idx][ids])
                }
                json_content.append(item)
        with open(json_file, 'w') as f:
            json.dump(json_content, f)
                
        
    def dataloader_inference(self, file_paths, idx):
        batch_instances = OrderedDict()
        batch_instances['meta'] = OrderedDict()
        batch_instances['meta']['id'] = []
        batch_instances['meta']['name'] = []
        batch_instances['meta']['width'] = []
        batch_instances['meta']['height'] = []
        img = cv2.imread(file_paths)
        batch_instances['meta']['id'].append(idx)
        batch_instances['meta']['name'].append(file_paths)
        batch_instances['meta']['width'].append(img.shape[1])
        batch_instances['meta']['height'].append(img.shape[0])
        return batch_instances

    def _get_bboxes_batch(self, batch_instances, json_file):
        
        info_meta = batch_instances['meta']
        results = self.model(info_meta['name'], verbose = False)
        
        batch_bboxes = torch.as_tensor([(result.boxes.xyxy.cpu().numpy()) for result in results])
        batch_cls_scores = torch.as_tensor([(result.boxes.conf.cpu().numpy()) for result in results])
        batch_labels = torch.as_tensor([(result.boxes.cls.cpu().numpy()) for result in results])
        batch_cls_uncertainties = -1 * (batch_cls_scores * torch.log(batch_cls_scores+1e-10) + (1-batch_cls_scores) * torch.log((1-batch_cls_scores) + 1e-10))
        batch_box_uncertainties = torch.zeros_like(batch_cls_uncertainties)
        self.create_json_inference(batch_bboxes, batch_cls_scores , batch_labels, batch_cls_uncertainties, info_meta, json_file)


    def __call__(self, file_paths, out):
        if os.path.exists(out):
            os.remove(out)
        for idx, file_path in enumerate(file_paths):
            batch_instances = self.dataloader_inference(file_path, idx)
            self._get_bboxes_batch(batch_instances, out)
        return out


