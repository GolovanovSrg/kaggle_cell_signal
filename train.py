import argparse
import json
import torch

from attrdict import AttrDict

from dataset import CellsDataset, TrainTransform
from model import Encoder, Decoder
from trainer import Trainer
from utils import set_seed, train_test_split, get_data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json')
    parser.add_argument("--local_rank", type=int)

    return parser


def main(args):
    torch.multiprocessing.set_start_method('spawn')
    torch.distributed.init_process_group(backend="nccl")

    with open(args.config_path, 'r') as file:
        config = AttrDict(json.load(file))

    set_seed(config.seed + torch.distributed.get_rank())

    train_data_csv, test_data_csv = train_test_split(config.train_data_csv_path, config.n_test_experiments)
    
    train_image_ids, train_labels = get_data(train_data_csv, is_train=True)
    train_transform = TrainTransform(config.crop_size)
    train_dataset = CellsDataset(config.train_images_dir, train_image_ids, train_labels, train_transform)
    
    test_image_ids, test_labels = get_data(test_data_csv, is_train=True)
    test_dataset = CellsDataset(config.train_images_dir, test_image_ids, test_labels)

    if torch.distributed.get_rank() == 0:
        print(f'Train size: {len(train_dataset)}, test_size: {len(test_dataset)}')

    encoder = Encoder(config.n_image_channels, config.n_emedding_channels, config.n_classes,
                      config.encoder_model, config.encoder_pretrained, config.encoder_dropout, config.encoder_scale)
    
    if config.restore_checkpoint_path is not None:
        state_dict = torch.load(config.restore_checkpoint_path, map_location='cpu')
        encoder.load_state_dict(state_dict)
    
    decoder = Decoder(config.n_emedding_channels, config.n_image_channels, config.n_classes,
                      config.decoder_n_channels)

    trainer = Trainer(encoder=encoder,
                      decoder=decoder,
                      optimizer_params={'lr': config.lr,
                                        'weight_decay': config.weight_decay,
                                        'warmap': config.warmap,
                                        'amsgrad': config.amsgrad},
                      amp_params={'opt_level': config.opt_level,
                                  'loss_scale': config.loss_scale},
                      rank=args.local_rank,
                      n_jobs=config.n_jobs)
    trainer.train(train_data=train_dataset,
                  n_epochs=config.n_epochs,
                  batch_size=config.batch_size,
                  test_data=test_dataset,
                  best_checkpoint_path=config.best_checkpoint_path)


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)