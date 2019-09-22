import os
from PIL import Image

import albumentations as alb
import numpy as np
import torch
from torch.utils.data import Dataset


# TODO: add transforms


class TrainTransform:
    def __init__(self, crop_size=(256, 256)):
        self.transform = alb.Compose([alb.Rotate(limit=90, p=0.5),
                                      alb.RandomSizedCrop(min_max_height=(int(0.8 * crop_size[0]), int(1.2 * crop_size[0])),
                                                          height=crop_size[0], width=crop_size[1]),
                                      alb.RandomRotate90(p=0.5),
                                      alb.VerticalFlip(p=0.5),
                                      alb.HorizontalFlip(p=0.5),
                                      alb.Cutout(num_holes=8, max_h_size=8, max_w_size=8)])

    def __call__(self, image):
        return self.transform(image=image)['image']


class CellsDataset(Dataset):
    @staticmethod
    def _load_image(image_dir, image_id):
        channel_image_paths = [os.path.join(image_dir, f'{image_id}_w{c}.png') for c in range(1, 7)]
        channel_images = [np.array(Image.open(path)) for path in channel_image_paths]
        image = np.stack(channel_images, axis=-1)
        
        return image

    def __init__(self, image_dir, image_ids, labels, transform=None):
        assert len(image_ids) == len(labels)

        self.image_dir = image_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)
      
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = CellsDataset._load_image(self.image_dir, image_id)

        mean = np.mean(image.astype('float32'), axis=(0, 1))
        std = np.std(image.astype('float32'), axis=(0, 1))

        if self.transform is not None:
            image = self.transform(image)

        image = (image.astype('float32') - mean) / (std * 255)
        label = self.labels[idx]

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        label = torch.tensor(label).long()
        
        return image, label
