import cv2
import torch
import numpy as np
from torchvision.ops import nms
from ultralytics.utils.tal import make_anchors
from ops import get_img_score_distance_matrix_slow
from ultralytics.nn.tasks import yaml_model_load, parse_model
from copy import deepcopy


class FeatsExtraction():
    def __init__(self, model, uncertainty_file, model_yaml):
        self.max_bbox = 200
        self.model  = model.model.cpu()
        self.uncertainty_file = uncertainty_file
        m=self.model.model[-1]
        self.nc= len(model.names)
        self.no= self.nc + m.reg_max * 4
        self.reg_max= m.reg_max
        self.stride=m.stride
        self.feat_dim = self.model.model[22].cv3[-1][-1].in_channels
        self.model_yaml = model_yaml
        
    def get_feature(self, x):
        yaml = yaml_model_load(self.model_yaml)
        save = parse_model(deepcopy(yaml), ch=self.feat_dim , verbose=False)[1]
        y = []
        for count ,m in enumerate(self.model.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if count == 22:
                tensor1_flat = m.cv3[0][:2](x[0]).flatten(2)  # -> [1, 128, 6400]
                tensor2_flat = m.cv3[1][:2](x[1]).flatten(2)  # -> [1, 128, 1600]
                tensor3_flat = m.cv3[2][:2](x[2]).flatten(2) 
                result = torch.cat([tensor1_flat, tensor2_flat, tensor3_flat], dim=2)  # -> [1, 128, 8400]
                return result
            x = m(x)  # run
            y.append(x if m.i in save else None)  # save output save output


    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
    
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def bbox_decode(self, anchor_points, pred_dist):
                proj=torch.arange(16, dtype=torch.float).cpu()
                """Decode predicted object bounding box coordinates from anchor points and distribution."""
                b, a, c = pred_dist.shape  # batch, anchors, channels
                pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    
                # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
                # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
                return self.dist2bbox(pred_dist, anchor_points, xywh=False)

    def pre(self, img0):
        img0, w, h, width, height = self.letterbox(img0)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img0 = img0 / 255.0
        img0 = img0.transpose(2, 0, 1)
        return img0, w, h, width, height

    def letterbox(self, img, new_shape = (640, 640), color = (114, 114, 114), 
                auto = False, scale_fill = False, scaleup = False, stride = 32):
        
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        height, width = img.shape[:2]
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, dw, dh, width, height

    def predict_img(self, image_path, diversity=False):
        batch_images = []
        for img_path in [image_path]:
            img=cv2.imread(img_path)
            x, w, h, width, height = self.pre(img)
            x = torch.from_numpy(x).to('cpu').float()  # Chuyển đổi từ numpy thành tensor
            batch_images.append(x)
        batch_images = torch.stack(batch_images)
        if diversity == True:
            with torch.no_grad():
                preds = self.get_feature(batch_images)  
        else:
            with torch.no_grad():
                preds = self.model.predict(batch_images)
        return preds

    def extract_feature(self, image_path):
        pred_feats = self.predict_img(image_path, diversity = True).transpose(1, 2)
        preds = self.predict_img(image_path, diversity = False)
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1
            )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()
    
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # imgsz = torch.tensor(feats[0].shape[2:],  dtype=dtype) * stride[0].to('cpu') # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # anc_points=anchor_points * stride_tensor
        # Targets
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)*stride_tensor 

        mlvl_labels = []
        mlvl_scores = []
        for idx, pred_bbox in enumerate(pred_bboxes):
            max_values, label = torch.max(pred_scores[idx], dim=1)
            mlvl_labels.append(label.cpu())
            mlvl_scores.append(max_values.cpu())
            keep_ids = nms(pred_bbox, max_values, 0.3)

        mlvl_labels = torch.from_numpy(np.array(mlvl_labels[0], dtype=np.float32)).cpu()
        mlvl_scores = torch.from_numpy(np.array(mlvl_scores[0], dtype=np.float32)).cpu()

        det_labels = mlvl_labels[keep_ids][:self.max_bbox]
        det_scores = mlvl_scores[keep_ids][:self.max_bbox]
        det_bboxes = pred_bbox[keep_ids][:self.max_bbox]
        det_feats = pred_feats[0][keep_ids][:self.max_bbox]


        return det_labels, det_bboxes, det_feats, det_scores, anchor_points, pred_feats


    def get_all_meta_data(self,list_img_path):

        queue_det_feats = torch.zeros((len(list_img_path), 200, self.feat_dim))
        queue_det_labels = torch.zeros((len(list_img_path), 200))
        queue_det_scores = torch.zeros((len(list_img_path), 200))
        queue_det_idx = torch.zeros((len(list_img_path), 1))

        for idx, img_path in enumerate(list_img_path):
            det_labels, det_bboxes, det_feats, det_scores,_,_ = self.extract_feature(img_path) 
            queue_det_idx[idx] = int(idx)
            queue_det_feats[idx] = det_feats
            queue_det_labels[idx] = det_labels
            queue_det_scores[idx] = det_scores

        return queue_det_idx, queue_det_feats, queue_det_labels, queue_det_scores

    def compute_al(self, valid_inds, list_path, out):
        
        list_img_uncertainty_path = []
        for i in valid_inds:
            list_img_uncertainty_path.append(list_path[i])

        queue_det_idx, queue_det_feats, queue_det_labels, queue_det_scores = self.get_all_meta_data(list_img_uncertainty_path)
        img_dis_mat = get_img_score_distance_matrix_slow(
                queue_det_labels, queue_det_scores, queue_det_feats, score_thr=0.05)
        img_dis_mat = img_dis_mat.detach().cpu().numpy()
        img_ids = queue_det_idx.detach().cpu().numpy()
        with open(out, 'wb') as fwb:
                np.save(fwb, img_dis_mat)
                np.save(fwb, img_ids)

        with open(self.uncertainty_file, 'w') as f:
                for line in list_img_uncertainty_path:
                    f.write(f"{line}\n")
        return

    def __call__(self, valid_inds, list_path, out):
        self.compute_al(valid_inds, list_path, out)