import argparse
import json
import random
import torch
import numpy as np
import pandas as pd

from collections import Counter
from attrdict import AttrDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CellsDataset
from model import Encoder
from utils import get_data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json')

    return parser

def torch_none(x):
    return x

def torch_rot90(x):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x):
    return x.flip(2).flip(3)


def torch_rot270(x):
    return x.transpose(2, 3).flip(3)

def torch_random_crop(x, crop_size=(384, 384)):
    h, w = x.shape[2:]
    x1 = random.randint(0, w - crop_size[1])
    x2 = x1 + crop_size[1]
    y1 = random.randint(0, h - crop_size[0])
    y2 = y1 + crop_size[0]
    return x[:, :, y1:y2, x1:x2]


def make_submit(predicted, labels):
    assert len(labels) % 2 == 0

    submit = []
    for i in range(len(labels) // 2):
        assert labels[i] == labels[i + len(labels) // 2]
        transformed_preds = [p[i] for p in predicted] + [p[i + len(labels) // 2] for p in predicted]
        argmaxes = [p.argmax() for p in transformed_preds]
        sirna, n = Counter(argmaxes).most_common(1)[0]
        if n > 1:
            submit.append((labels[i], sirna))
        else:
            print("sum")
            sirna = sum(transformed_preds, 0).argmax()
            submit.append((labels[i], sirna))

    pd.DataFrame(submit, columns=['id_code','sirna']).to_csv('submission.csv', index=False)


def main(args):
    random.seed(0)

    with open(args.config_path, 'r') as file:
        config = AttrDict(json.load(file))

    data_csv = pd.read_csv(config.test_data_csv_path)
    image_ids, labels = get_data(data_csv, is_train=False)
    dataset = CellsDataset(config.test_images_dir, image_ids)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.n_jobs)

    encoder = Encoder(config.n_image_channels, config.n_emedding_channels, config.n_classes,
                      config.encoder_model, config.encoder_pretrained, config.encoder_dropout, config.encoder_scale)
    
    if config.restore_checkpoint_path is not None:
        state_dict = torch.load(config.restore_checkpoint_path, map_location='cpu')
        encoder.load_state_dict(state_dict)

    device = torch.device('cuda:0')
    encoder = encoder.half().to(device)
    encoder.eval()

    transforms = [torch_none, torch_rot90, torch_rot180, torch_rot270, torch_random_crop, torch_random_crop, torch_random_crop]

    predicted = [[] for _ in range(len(transforms))]
    for images in tqdm(dataloader):
        for i, t in enumerate(transforms):
            transformed_images = t(images)
            transformed_images = transformed_images.half().to(device)
            log_probs = encoder(transformed_images)
            predicted[i].append(log_probs.float().cpu().numpy())

    predicted = [np.concatenate(predicted[i], axis=0) for i in range(len(transforms))]
    make_submit(predicted, labels)


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    with torch.no_grad():
        main(args)
