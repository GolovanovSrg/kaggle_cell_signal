import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d, senet154


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()

        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0:
            noise = torch.randn_like(x) * self.sigma
            x = x + noise

        return x


class MoSLayer(nn.Module):
    def __init__(self, in_feature, out_feature, middle_feature=None, prior_feature=None, activation=nn.CELU(inplace=True),
                 scale=64, n_softmax=1, dropout=0):
        super().__init__()

        if middle_feature is None:
            middle_feature = in_feature

        if prior_feature is None:
            prior_feature = in_feature

        self.scale = scale
        self.n_softmax = n_softmax
        self.latent = nn.Linear(in_feature, n_softmax * middle_feature)
        self.prior = nn.Sequential(nn.Linear(in_feature, prior_feature), activation, nn.Linear(prior_feature, n_softmax))
        self.weight = nn.Parameter(torch.Tensor(1, n_softmax, middle_feature, out_feature))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        latent = self.latent(x).view(x.shape[0], self.n_softmax, 1, -1)
        latent = self.dropout(latent)

        logit = self.scale * torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.weight, dim=2)).squeeze(2)
        log_p = F.log_softmax(logit, dim=-1)

        prior_logit = self.prior(x)
        log_prior = F.log_softmax(prior_logit, dim=-1)

        out = torch.logsumexp(log_p + log_prior.unsqueeze(-1), dim=1)

        return out


class DistanceLayer(nn.Module):
    def __init__(self, in_feature, out_feature, middle_feature=None, n_centers=1, scale=64, dropout=0):
        super().__init__()

        if middle_feature is None:
            middle_feature = in_feature

        self.scale = scale
        self.proj = nn.Linear(in_feature, middle_feature)
        self.weight = nn.Parameter(torch.Tensor(out_feature * n_centers, middle_feature))
        self.max_pool = nn.MaxPool1d(kernel_size=n_centers, stride=n_centers)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = self.proj(x)
        x = self.dropout(x)
        out = self.scale * F.linear(F.normalize(x), F.normalize(self.weight))
        out = self.max_pool(out.unsqueeze(1)).squeeze(1)
        out = F.log_softmax(out, dim=-1)

        return out


class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature, out_feature, middle_feature=None, scale_size=64.0, m=0.5, easy_margin=False):
        super().__init__()

        if middle_feature is None:
            middle_feature = in_feature

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = scale_size
        self.m = m
        self.proj = nn.Linear(in_feature, middle_feature)
        self.weight = nn.Parameter(torch.Tensor(out_feature, middle_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        x = self.proj(x)

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = cosine.clamp(-1, 1)

        if self.training:
            # cos(theta + m)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine.type_as(phi))
            else:
                phi = torch.where(cosine > self.th, phi, (cosine - self.mm).type_as(phi))

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = one_hot * phi + (1.0 - one_hot) * cosine
        else:
            output = cosine

        output = output * self.s

        return output


def replace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.CELU(inplace=True))
        else:
            replace_relu(child)


class CatPool2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, model='seresnext50',
                 pretrained=False, dropout=0, scale=64):
        super().__init__()

        assert model in ['seresnext50', 'seresnext101', 'senet154']

        pretrained_dataset = 'imagenet' if pretrained else None
        if model == 'seresnext50':
            self.model = se_resnext50_32x4d(pretrained=pretrained_dataset)
            inplanes = 64
        elif model == 'seresnext101':
            self.model = se_resnext101_32x4d(pretrained=pretrained_dataset)
            inplanes = 64
        elif model == 'senet154':
            self.model = senet154(pretrained=pretrained_dataset)
            self.model.dropout = nn.Identity()
            inplanes = 128
        else:
            assert False


        layer0_modules = [
            ('conv1', nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),

            ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(inplanes)),
            ('relu3', nn.ReLU(inplace=True)),

            ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))
        ]

        self.model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.model.avg_pool = nn.Sequential(CatPool2d(),
                                            GaussianNoise(),
                                            nn.Flatten())

        #self.dist = DistanceLayer(self.model.last_linear.in_features, num_classes, middle_feature=None, scale=64, n_centers=5, dropout=0)
        #self.margin = ArcMarginProduct(self.model.last_linear.in_features, num_classes)
        self.mos = MoSLayer(2 * self.model.last_linear.in_features, num_classes, middle_feature=512, prior_feature=1024, scale=64, n_softmax=10, dropout=0)
        self.model.last_linear = nn.Identity()
        replace_relu(self.model)
        #self.out_proj = nn.Sequential(nn.Conv2d(self.model.layer4[-1].conv3.out_channels, out_channels, kernel_size=3, padding=1),
        #                              nn.Sigmoid())

        self.scale = scale

    def forward(self, x, label=None):
        x = self.model.features(x)
        #out = self.out_proj(x)

        y = self.model.logits(x)
        y = self.mos(y)  # log_softmax
        #y = self.margin(y, label)
        #y = self.dist(y)  # log_softmax

        return y


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,
                 n_channels=[512, 256, 128, 64, 64],
                 activation=nn.ELU(inplace=True)):
        super().__init__()
        self.num_classes = num_classes
        # TODO: insert class embedding for each layer
        self.cls_embedding = nn.Embedding(num_classes, in_channels)

        layers = []
        n_channels = [in_channels] + list(n_channels)
        for i in range(1, len(n_channels)):
            sublayer = nn.Sequential(nn.Conv2d(n_channels[i-1], n_channels[i], kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(n_channels[i]),
                                     activation,
                                     nn.ConvTranspose2d(n_channels[i], n_channels[i], kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(n_channels[i]),
                                     activation,
                                     nn.Conv2d(n_channels[i], n_channels[i], kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(n_channels[i]),
                                     activation)
            layers.append(sublayer)
        
        self.layers = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(nn.Conv2d(n_channels[-1], out_channels, kernel_size=3, padding=1),
                                      nn.Sigmoid())

    def forward(self, x, y):
        emb = F.sigmoid(self.cls_embedding(y))
        x = x + emb.unsqueeze(-1).unsqueeze(-1)
        x = self.layers(x)
        out = self.out_proj(x)

        return out
