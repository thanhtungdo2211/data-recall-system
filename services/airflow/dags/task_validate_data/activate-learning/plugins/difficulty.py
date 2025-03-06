import json
import numpy as np
from collections import OrderedDict

class Calibrated_Sampler():
    def __init__(self, n_images, score_thr, floor_score_thr, category_valid):
        self.n_images = n_images
        self.score_thr = score_thr
        self.floor_score_thr = floor_score_thr
        self.category_valid = category_valid

    def get_class_qualities(self, quality_file_path):
        # Đọc nội dung file JSON
        with open(quality_file_path, 'r') as f:
            data = json.load(f)
        # Lấy danh sách class_quality từ file JSON
        class_qualities = data["class_quality"]

        return np.array(class_qualities)

    def _get_classwise_weight(self, quality_file_path):
        class_weight_alpha= 0.3
        class_weight_ub = 0.2
        class_qualities = self.get_class_qualities(quality_file_path)
        reverse_q = 1 - class_qualities
        b = np.exp(1. / class_weight_alpha) - 1
        _weights = 1 + class_weight_alpha * np.log(b * reverse_q + 1) * class_weight_ub

        class_weights = dict()
        for i in range(len(_weights)):
                class_weights[i] = _weights[i]
        # print( class_weights)
        return class_weights

    def is_box_valid(self, box, img_size):
            eps = 1e-10
            size_thr=16
            ratio_thr=5
            # clip box and filter out outliers
            img_w, img_h = img_size
            x1, y1, w, h = box
            if (x1 > img_w) or (y1 > img_h):
                return False
            x2 = min(img_w, x1+w)
            y2 = min(img_h, y1+h)
            w = x2 - x1
            h = y2 - y1
            return (np.sqrt(w*h) > size_thr) and (w/(h+eps) < ratio_thr) and (h/(w+eps) < ratio_thr)

    def al_acquisition(self, quality_file_path, uncertainty_path, max_entropy_filter=False):

        class_weights = self._get_classwise_weight(quality_file_path)
        # class_weights['0']

        with open(uncertainty_path) as f:
            results = json.load(f)

        category_uncertainty = OrderedDict()
        category_count = OrderedDict()
        image_uncertainties = OrderedDict()
        image_uncertainties_max = OrderedDict()
        for res in results:
            img_id = res['image_id']
            image_uncertainties[img_id] = [0.]
            image_uncertainties_max[img_id] = 0.
            img_size = (res['width'], res['height'])
            if not self.is_box_valid(res["bbox"], img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            if res['score'] > self.floor_score_thr:
                continue
            
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']
            if label not in category_uncertainty.keys():
                category_uncertainty[label] = 0.
                category_count[label] = 0.
            category_uncertainty[label] += uncertainty
            category_count[label] += 1

        category_avg_uncertainty = OrderedDict()
        for k in category_uncertainty.keys():
            category_avg_uncertainty[k] = category_uncertainty[k] / (category_count[k] + 1e-5)

        for res in results:
            img_id = res['image_id']
            img_size = (res['width'], res['height'])
            if not self.is_box_valid(res["bbox"], img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            if res['score'] > self.floor_score_thr:
                continue
            if res['category_id'] not in self.category_valid:
                continue
            # print(res['category_id'])
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']
            image_uncertainties[img_id].append(uncertainty * class_weights[label])

        for img_id in image_uncertainties.keys():
            _img_uncertainties = np.array(image_uncertainties[img_id])
            image_uncertainties_max[img_id] = _img_uncertainties.max()
            image_uncertainties[img_id] = _img_uncertainties.sum()
        

        img_ids = []
        merged_img_uncertainties = []
        check_uncertainty = 0
        for k, v in image_uncertainties.items():
            img_ids.append(k)
            merged_img_uncertainties.append(v)
            check_uncertainty += v
        if check_uncertainty == 0:
            assert 'All images have good quality'
        img_ids = np.array(img_ids)

        merged_img_uncertainties = np.array(merged_img_uncertainties)
        
        inds_sort = np.argsort(-1. * merged_img_uncertainties)
        
        sampled_inds = inds_sort[:self.n_images]
        new_sampled_inds = []

        for inds in sampled_inds:
            if merged_img_uncertainties[inds] != 0:
                new_sampled_inds.append(inds)

        unsampled_img_ids = inds_sort[self.n_images:]
        
        sampled_img_ids = img_ids[new_sampled_inds].tolist()
        unsampled_img_ids = img_ids[unsampled_img_ids].tolist()
        
        if max_entropy_filter:
            merged_img_uncertainties_max = []
            for k, v in image_uncertainties_max.items():
                merged_img_uncertainties_max.append(v)
            merged_img_uncertainties_max = np.array(merged_img_uncertainties_max)
            inds_sort_max = np.argsort(-1. * merged_img_uncertainties_max)
            sampled_inds_max = inds_sort_max[:self.n_images]
            new_sampled_inds_max = []
        
            for inds in sampled_inds:
                if merged_img_uncertainties[inds] != 0:
                    new_sampled_inds_max.append(inds)

            unsampled_img_max = inds_sort_max[self.n_images:]
            sampled_img_ids_max = img_ids[new_sampled_inds_max].tolist()
            unsampled_img_max = img_ids[unsampled_img_max].tolist()
            sampled_img_ids = list(set(sampled_img_ids + sampled_img_ids_max))
            unsampled_img_ids = list(set(unsampled_img_ids + unsampled_img_max))
        return sampled_img_ids, unsampled_img_ids

    def __call__(self, inp1, inp2, max_entropy_filter=False):
        sampled_img_ids, unsampled_img_ids = self.al_acquisition(inp1, inp2, max_entropy_filter)
        return sampled_img_ids, unsampled_img_ids
    