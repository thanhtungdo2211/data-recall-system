import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

import albumentations as A

import torchvision.ops as vision_ops
plt.rcParams["figure.dpi"] = 300
torch.autograd.set_grad_enabled(False)
from models import PointDecoder
from models import ROIHeadMLP as ROIHead
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from ops.foundation_models.segment_anything import build_sam_vit_h
from ops.dump_clip_features import dump_clip_image_features


class AutoLabelSam2:
    def __init__(self, model_paths=None, device="cpu"):
        """
        Initialize with model paths and device setting but don't load models yet.
        
        Args:
            model_paths: Dictionary with paths to models. If None, uses default paths.
            device: Device to use for computation ('cpu' or 'cuda').
        """
        self.sam = None
        self.sam2 = None
        self.predictor = None
        self.cls_head = None
        self.point_decoder = None
        self.device = device
        self.models_loaded = False
        
        # Store model paths for later loading
        if model_paths is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoints_dir = os.path.join(current_dir, "checkpoints")
            self.model_paths = {
                "mlp": os.path.join(checkpoints_dir, "MLP_small_box_w1_fewshot.tar"),
                "point_decoder": os.path.join(checkpoints_dir, "point_decoder_vith.pth"),
                "sam2": os.path.join(checkpoints_dir, "sam2.1_hiera_large.pt"),
                "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
            }
        else:
            self.model_paths = model_paths
      
    def __call__(self, img_source_path, example_boxes, img_target_path):
        """
        Process images to detect and extract bounding boxes.
        
        Args:
            img_source_path: Path to source image
            example_boxes: Example bounding boxes
            img_target_path: Path to target image
        """
        # Load models if not already loaded
        if not self.models_loaded:
            self.load_model()
            
        width, height = Image.open(img_target_path).size
        scale = max(width, height) / 1024
        source_img = self.read_image(img_source_path)
        target_img = self.read_image(img_target_path)
        target_features = self.extract_features(target_img)
        example_features = dump_clip_image_features(source_img, example_boxes)
        pred_heatmaps, pred_points, pred_points_score = self.extract_heatmaps(target_features)
        pred_boxes = self.extract_boxes(target_img, pred_points, target_features, example_features, scale)
        return pred_boxes
      
    def load_model(self):
        """
        Load all required models using the stored model paths.
        """
        # Load models using self.model_paths
        self.sam2 = build_sam2(self.model_paths["config"], self.model_paths["sam2"], device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2)
        self.sam = build_sam_vit_h().eval().to(self.device)
        self.cls_head = ROIHead().eval().to(self.device)
        self.cls_head.load_state_dict(torch.load(self.model_paths["mlp"], map_location=self.device, weights_only=False)['cls_head'])
        self.point_decoder = PointDecoder(self.sam).eval().to(self.device)
        state_dict = torch.load(self.model_paths["point_decoder"], map_location=self.device)
        self.point_decoder.load_state_dict(state_dict)
        
        # Mark models as loaded
        self.models_loaded = True
        print("Models loaded successfully")
      
    def read_image(self, path):
        img = Image.open(path).convert("RGB")
        transform = A.Compose([
            A.LongestMaxSize(1024),
            A.PadIfNeeded(1024, 1024,
                            border_mode=0,
                            position='top_left')
        ])
        img = Image.fromarray(transform(image=np.array(img))['image'])
        return img  
      
    def calculate_bbox(self, matrix):
        mat = np.array(matrix)
        # Fine all points
        rows, cols = np.where(mat == 1)
        if len(rows) == 0:  # If no points found
            return None
        # Tính toán các giá trị cạnh
        top = [cols[0], rows[0] - 1]
        bottom = [cols[-1], rows[-1] + 1]
        left = [cols.min() - 1, rows[np.argmin(cols)]]
        right = [cols.max() + 1, rows[np.argmax(cols)]]
        bbox = [left[0], top[1], right[0], bottom[1]]
        return bbox
      
    def segment_sam2(self, image, pred_points):
        image = image.convert("RGB")
        list_bboxes = []
        width, height = image.size
        self.predictor.set_image(image)
        kqua = np.zeros((width, height), dtype=int)
        input_point = pred_points.detach().cpu().numpy()
        input_label = np.array([1])
        for input in input_point:
            kqua1 = np.zeros((1024, 1024), dtype=int)
            masks, scores, logits = self.predictor.predict(
                                    point_coords=[input],
                                    point_labels=input_label,
                                    multimask_output=True,
                                    )
            kqua1 = kqua1 | masks[0].astype(int)
            if(np.sum(masks[1].astype(int)) <  np.sum(masks[0].astype(int)) * 20):
                kqua1 = kqua1 | masks[1].astype(int)
            if(np.sum(masks[2].astype(int)) <  np.sum(masks[0].astype(int)) * 20):
                kqua1 = kqua1 | masks[2].astype(int)
            input_box = self.calculate_bbox(kqua1)
            list_bboxes.append(input_box)
        return list_bboxes
        
    def extract_features(self, image):
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            new_image = transform(image).unsqueeze(0).to(self.device)
            features = self.sam.image_encoder(new_image)
        return features
      
    def extract_heatmaps(self, features):
        with torch.no_grad():
            self.point_decoder.max_points = 1000
            self.point_decoder.point_threshold = 0.3
            self.point_decoder.nms_kernel_size = 3
            outputs_heatmaps = self.point_decoder(features)
            pred_heatmaps = outputs_heatmaps['pred_heatmaps'].cpu().squeeze().clamp(0, 1)
        pred_points = outputs_heatmaps['pred_points'].squeeze().reshape(-1, 2)
        pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()
        return pred_heatmaps, pred_points, pred_points_score

    def extract_boxes(self, image, pred_points, features, example_features, scale):
        import time
        start = time.time()
        _ = self.cls_head.eval()
        with torch.no_grad():
            all_pred_boxes = []
            all_pred_ious = []
            cls_outs = []
            for indices in torch.arange(len(pred_points)).split(128):
                with torch.no_grad():
                    outputs_points = self.sam.forward_sam_with_embeddings(features, points=pred_points[indices])
                    pred_boxes = outputs_points['pred_boxes']
                    pred_logits = outputs_points['pred_ious']
                    
                    # Change here: using the device variable instead of hardcoded "cuda"
                    anchor_boxes = torch.tensor(self.segment_sam2(image, pred_points[indices])).to(self.device)
                    anchor_boxes = anchor_boxes.clamp(0., 1024.)
                    
                    outputs_boxes = self.sam.forward_sam_with_embeddings(features, points=pred_points[indices], boxes=anchor_boxes)
                    pred_logits = torch.cat([pred_logits, outputs_boxes['pred_ious'][:, 1].unsqueeze(1)], dim=1)
                    pred_boxes = torch.cat([pred_boxes, outputs_boxes['pred_boxes'][:, 1].unsqueeze(1)], dim=1)

                    all_pred_boxes.append(pred_boxes)
                    all_pred_ious.append(pred_logits)
                    cls_outs_ = self.cls_head(features, [pred_boxes, ], [example_features, ] * len(indices))
                    cls_outs_ = cls_outs_.sigmoid().view(-1, len(example_features), 5).mean(1)
                    pred_logits = cls_outs_ * pred_logits
                cls_outs.append(pred_logits)
            pred_boxes = torch.cat(all_pred_boxes)
            pred_ious = torch.cat(all_pred_ious)
            cls_outs = torch.cat(cls_outs)
            pred_boxes = pred_boxes[torch.arange(len(pred_boxes)), torch.argmax(cls_outs, dim=1)]
            scores = cls_outs.max(1).values

            score_threshold = 0.3  # Ngưỡng tối thiểu để giữ bbox
            # Lọc các bbox có điểm số cao hơn ngưỡng
            valid_mask = scores > score_threshold  # Tạo mask để giữ lại bbox có score cao
            pred_boxes = pred_boxes[valid_mask]
            scores = scores[valid_mask]

            indices = vision_ops.nms(pred_boxes, scores, 0.5)
            pred_boxes = pred_boxes[indices]
            scores = scores[indices]
            
        pred_boxes[:, [0, 2]] *= scale  # Scale tọa độ x (x1, x2)
        pred_boxes[:, [1, 3]] *= scale  # Scale tọa độ y (y1, y2)
        bboxes = pred_boxes.type(torch.int32)
        print(f"Time to get bboxes : {time.time() - start}")
        
        # Already on CPU, so no need for .to("cpu")
        return bboxes