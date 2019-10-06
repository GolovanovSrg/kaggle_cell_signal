import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import disable_tracking_bn_stats


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100, reduction="mean"):
        super().__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            if reduction == "mean":
                reduction = "batchmean"
            self.criterion = nn.KLDivLoss(reduction=reduction)
            n_ignore_idxs = 1 + (ignore_index >= 0)
            one_hot = torch.full(
                (1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs))
            )
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer("one_hot", one_hot)
        else:
            self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        log_inputs = inputs # F.log_softmax(inputs, dim=-1)
        if self.confidence < 1:
            tdata = targets.data

            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp

            if self.criterion.reduction == 'none':
               return self.criterion(log_inputs, targets).sum(dim=-1)

        return self.criterion(log_inputs, targets)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-12
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=0.1, eps=0.1, ip=1):
        """
        VAT loss: https://github.com/lyakaap/VAT-pytorch
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with disable_tracking_bn_stats(model):
            with torch.no_grad():
                #pred = F.softmax(model(x), dim=1)
                pred = torch.exp(model(x))

            # prepare random unit tensor
            d = torch.rand_like(x) - 0.5
            d = _l2_normalize(d)

            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = pred_hat  # F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model((x + r_adv).detach())
            logp_hat = pred_hat  # F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
            lds = lds - (torch.exp(logp_hat) * logp_hat).sum(dim=1).mean()

        return lds