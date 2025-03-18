import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu

import time
from PIL import Image
import os

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class CompareImageInternVL:
    def __init__(self):
        self.init_model()
        
    def __call__(self, source_image_path, target_image_path, bboxes):
        result = {
        'bboxes': [],
        'labels': []
        }
        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(current_dir, "images")
        crop_image_path = os.path.join(image_dir, "crop_image.jpg")
        cropped_images = self.crop_bboxes_pil(target_image, bboxes)
        for i in range(0,len(cropped_images)):
            cropped_images[i].save(crop_image_path)
            res = self.compare_object(source_image_path, crop_image_path)
            if "true" in res :
                result["bboxes"].append(bboxes[i])
                result["labels"].append("")
        return result["bboxes"]
    
    def init_model(self):
        # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
        path = 'OpenGVLab/InternVL2_5-2B'
        self.model_compare = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        
    def compare_object(self, image1_path, image2_path):
        # @title Văn bản tiêu đề mặc định
        prompt = """
                Identify the main object in Image 1 and Image 2, ensuring it is a single object occupying the majority of the image. If no main object is present, immediately return false without proceeding further. Extract generalized features of both objects (such as shape, color, etc., excluding size) and compare them. If they are similar, return true; otherwise, return false. Ensure the output is a single word: true or false.

                *Example 1 (Object too small):
                Image 1: A white background with a small red dot in the corner.
                Image 2: A white background with a large red dot in the corner.
                Result: false*

                Example 2:
                Image 1: A people on a white background.
                Image 2: A people on a road.
                Result: true

                Example 3:
                Image 1: A blue car on a street.
                Image 2: A red bicycle in a park.
                Result: false

                Example 4:
                Image 1: An empty background with no visible object.
                Image 2: A green tree in a garden.
                Result: false
                """
        # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
        pixel_values1 = load_image(image1_path, max_num=12).to(torch.bfloat16)
        pixel_values2 = load_image(image2_path, max_num=12).to(torch.bfloat16)
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        question = 'Image-1: <image>\nImage-2: <image>\n' + prompt
        response, history = self.model_compare.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        return response
    
    def crop_bboxes_pil(self, image, bboxes):
        """
        Crop bounding boxes from an image and return a list of cropped images as PIL JpegImageFile objects.

        Parameters:
        - image (PIL.Image.Image): Input image from which the bounding boxes will be cropped.
        - bboxes (list of tuples): List of bounding boxes, where each bbox is a tuple
        (x_min, y_min, x_max, y_max).

        Returns:
        - cropped_images (list of PIL.JpegImagePlugin.JpegImageFile): List of cropped images corresponding to the bounding boxes.
        """
        cropped_images = []
        # print(bboxes)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            # Ensure bbox coordinates are within image boundaries and x_max > x_min, y_max > y_min
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(image.width, int(x_max)) # Changed image.height to image.width
            y_max = min(image.height, int(y_max)) # Changed image.width to image.height
            if x_max > x_min and y_max > y_min:
                # Crop the image
                cropped = image.crop((x_min, y_min, x_max, y_max))
                cropped_images.append(cropped)
            else:
                print(f"Invalid bbox: {bbox}, skipping.")  # Log invalid bboxes

        return cropped_images
