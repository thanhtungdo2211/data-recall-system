import os
from pathlib import Path

from deepchecks.vision.vision_data import VisionData

def deepchecks_collate(batch):
    """Process batch to deepchecks format."""
    imgs, labels = zip(*batch)
    return {'images': list(imgs), 'labels': list(labels)}

def generate_reports(**kwargs):
    from deepchecks.vision.vision_data.simple_classification_data import SimpleClassificationDataset 
    from deepchecks.vision.checks import ImagePropertyDrift
    from deepchecks.vision.suites import train_test_validation
    from deepchecks.vision import classification_dataset_from_directory
    from torch.utils.data import DataLoader
    from typing_extensions import Literal
    import torch

    check = ImagePropertyDrift()

    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = True

    train_dataset = SimpleClassificationDataset(root = "/mnt/d/Personal/Programing/PersonalProjects/data-recall-system/tests/backup/trains/", image_extension='jpg')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=deepchecks_collate, pin_memory=pin_memory, generator=torch.Generator())
    train_visiondata = VisionData(batch_loader=train_dataloader, label_map=train_dataset.reverse_classes_map,
                                        task_type='classification')

    test_dataset = SimpleClassificationDataset(root = "/mnt/d/Personal/Programing/PersonalProjects/data-recall-system/tests/backup/trains/", image_extension='jpg')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=deepchecks_collate, pin_memory=pin_memory, generator=torch.Generator())
    test_visiondata = VisionData(batch_loader=test_dataloader, label_map=test_dataset.reverse_classes_map,
                                        task_type='classification')

    result = check.run(train_dataset=train_visiondata, test_dataset=test_visiondata)
    print(result)
