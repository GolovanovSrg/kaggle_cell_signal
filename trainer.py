import os
import random

import torch
import torch.nn as nn
import apex
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from loss import VATLoss, LabelSmoothingLoss
from optim import RAdam
from utils import disable_tracking_bn_stats


class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1

    def __call__(self):
        if self.count:
            return self.sum / self.count
        return 0


class CutMix:
    def __init__(self, n_classes, p=0.5, alpha=1):
        self.n_classes = n_classes
        self.p = p
        self.distribution = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, images, labels):
        assert len(images) == len(labels)

        if random.random() < self.p:
            return images, labels, labels, torch.ones(len(images), device=labels.device)

        indices = torch.randperm(len(images))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        h, w = images.shape[-2:]
        l = self.distribution.sample((len(images),))
        r_w = torch.sqrt(1 - l) * w
        r_h = torch.sqrt(1 - l) * h
        r_x = torch.randint(0, w, size=(len(images),)).float()
        r_y = torch.randint(0, h, size=(len(images),)).float()

        x1 = torch.clamp(r_x - r_w / 2, min=0).long()
        x2 = torch.clamp(r_x + r_w / 2, max=w).long()
        y1 = torch.clamp(r_y - r_h / 2, min=0).long()
        y2 = torch.clamp(r_y + r_h / 2, max=h).long()
        l = 1 - (x2 - x1).float() * (y2 - y1).float() / (w * h)
        l = l.to(labels.device)

        new_images = images
        for i in range(len(images)):
            new_images[i, :, y1[i]:y2[i], x1[i]:x2[i]] = shuffled_images[i, :, y1[i]:y2[i], x1[i]:x2[i]]

        return new_images, labels, shuffled_labels, l


class Trainer:
    def __init__(self, encoder, decoder, optimizer_params={}, amp_params={}, n_jobs=0, rank=0):
        
        lr = optimizer_params.get('lr', 1e-3)
        weight_decay = optimizer_params.get('weight_decay', 0)
        warmap = optimizer_params.get('warmap', 100)
        amsgrad = optimizer_params.get('amsgrad', False)
        opt_level = amp_params.get('opt_level', 'O0')
        loss_scale = amp_params.get('loss_scale', None)

        self.device = torch.device('cuda:' + str(rank))
        self.encoder = encoder.to(self.device)
        #self.decoder = decoder.to(self.device)
        self.num_classes = decoder.num_classes
        self.mse_critetion = nn.L1Loss()
        self.ce_criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1, reduction='none').to(self.device)
        self.vat_criterion = VATLoss()
        self.cutmix = CutMix(self.num_classes)

        param_optimizer = list(self.encoder.named_parameters()) #+ list(self.decoder.named_parameters())
        no_decay = ['bn', 'bias']
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = RAdam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

        self.is_master = torch.distributed.get_rank() == 0
        torch.cuda.set_device(rank)
        [self.encoder], self.optimizer = apex.amp.initialize([self.encoder], self.optimizer,
                                                             opt_level=opt_level, loss_scale=loss_scale, verbosity=1)

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.encoder = apex.parallel.DistributedDataParallel(self.encoder, delay_allreduce=True)
        #self.decoder = apex.parallel.DistributedDataParallel(self.decoder, delay_allreduce=True)

        self.last_epoch = 0
        self.n_jobs = n_jobs

    def _train_epoch(self, train_dataloader):
        if self.is_master:
            pbar = tqdm(desc=f'Train, epoch #{self.last_epoch}',
                        total=len(train_dataloader))

        self.encoder.train()
        #self.decoder.train()

        sum_loss, cls_loss = AvgMeter(), AvgMeter()
        for images, labels in train_dataloader:
            images, labels, shuffled_labels, l = self.cutmix(images, labels)
            images = images.to(self.device)
            labels = labels.to(self.device)
            shuffled_labels = shuffled_labels.to(self.device)
            l = l.to(self.device)

            self.optimizer.zero_grad()

            #loss_vat = self.vat_criterion(self.encoder, images)

            label_preds = self.encoder(images)
            #reconsts_l = self.decoder(latents, labels)
            #with disable_tracking_bn_stats(self.encoder):
            #    latents_l, label_preds_l = self.encoder(reconsts_l)
            #labels_r = torch.randint_like(labels, low=0, high=self.num_classes)
            #reconsts_r = self.decoder(latents, labels_r)
            #with disable_tracking_bn_stats(self.encoder):
            #    latents_r, label_preds_r = self.encoder(reconsts_r)

            loss_c = (l * self.ce_criterion(label_preds, labels) + (1 - l) * self.ce_criterion(label_preds, shuffled_labels)).mean()
            #loss_r = self.mse_critetion(reconsts_l, images)
            #loss_e = self.ce_criterion(label_preds_r, labels_r)
            #loss_i = self.mse_critetion(latents_l, latents_r)
            
            losses = loss_c #+ loss_vat # + loss_r + loss_e + loss_i

            with apex.amp.scale_loss(losses, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                    
            self.optimizer.step()

            sum_loss.update(losses.item())
            cls_loss.update(loss_c.item())

            info_tensor = torch.tensor([sum_loss(), cls_loss()], device=self.device)
            torch.distributed.reduce(info_tensor, dst=0)

            if self.is_master:
                info_tensor = info_tensor / torch.distributed.get_world_size()
                pbar.update(1)
                pbar.set_postfix({'sum_loss': info_tensor[0].item(),
                                  'cls_loss': info_tensor[1].item()})
        
        self.scheduler.step()

    def _test_epoch(self, test_dataloader):
        with torch.no_grad():
            if self.is_master:
                pbar = tqdm(desc=f'Test, epoch #{self.last_epoch}',
                            total=len(test_dataloader))
            
            self.encoder.eval()

            loss, acc, quality_metric = AvgMeter(), AvgMeter(), 0
            for images, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                label_preds = self.encoder(images)
                loss_val = self.ce_criterion(label_preds, labels).mean()
                acc_val = (torch.argmax(label_preds, dim=-1) == labels).float().mean()

                loss.update(loss_val.item())
                acc.update(acc_val.item())

                info_tensor = torch.tensor([loss(), acc()], device=self.device)
                torch.distributed.reduce(info_tensor, dst=0)

                if self.is_master:
                    info_tensor = info_tensor / torch.distributed.get_world_size()
                    quality_metric = info_tensor[1].item()
                    pbar.update(1)
                    pbar.set_postfix({'loss': info_tensor[0].item(),
                                      'acc': info_tensor[1].item()})

            return quality_metric

    def _save_checkpoint(self, checkpoint_path):  
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.encoder.module.state_dict(), checkpoint_path)

    def train(self, train_data, n_epochs, batch_size, test_data=None,
              last_checkpoint_path=None, best_checkpoint_path=None):

        num_replicas = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        batch_size = batch_size // num_replicas

        train_sampler = DistributedSampler(train_data, shuffle=True, num_replicas=num_replicas, rank=rank)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=self.n_jobs)

        if test_data is not None:
            test_sampler = DistributedSampler(test_data, shuffle=False, num_replicas=num_replicas, rank=rank)
            test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, num_workers=self.n_jobs)

        best_metric = float('-inf')
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader)

            if last_checkpoint_path is not None and self.is_master:
                self._save_checkpoint(last_checkpoint_path)

            if test_data is not None:
                torch.cuda.empty_cache()
                metric = self._test_epoch(test_dataloader)

                if best_checkpoint_path is not None and self.is_master:
                    if metric > best_metric:
                        best_metric = metric
                        self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1
