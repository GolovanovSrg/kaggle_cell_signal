import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d


class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature, out_feature, scale_size=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = scale_size
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
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


def fix_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            fix_relu(child)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, model='seresnext50',
                 pretrained=False, dropout=0, scale=64):
        super().__init__()

        assert model in ['seresnext50', 'seresnext101']

        # TODO: add Bag of Tricks
        pretrained_dataset = 'imagenet' if pretrained else None
        if model == 'seresnext50':
            self.model = se_resnext50_32x4d(pretrained=pretrained_dataset)
        elif model == 'seresnext101':
            self.model = se_resnext101_32x4d(pretrained=pretrained_dataset)
        else:
            assert False

        # TODO: change activation
        self.norm = nn.BatchNorm2d(in_channels, in_channels, affine=False)
        if in_channels != 3:
            self.model.layer0[0] = nn.Conv2d(in_channels, self.model.layer0[0].out_channels, kernel_size=7,
                                             stride=2, padding=3, bias=False)
        self.model.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                            nn.Flatten(),
                                            nn.BatchNorm1d(self.model.last_linear.in_features),
                                            nn.Dropout(dropout, inplace=True))
        #self.weight = nn.Parameter(torch.Tensor(num_classes, self.model.last_linear.in_features))
        #nn.init.xavier_uniform_(self.weight)
        #self.margin = ArcMarginProduct(self.model.last_linear.in_features, num_classes)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, self.model.last_linear.in_features)
        self.out_proj = nn.Sequential(nn.Conv2d(self.model.layer4[-1].conv3.out_channels, out_channels, kernel_size=3, padding=1),
                                      nn.Tanh())

        self.scale = scale

    def forward(self, x, label=None):
        x = self.norm(x)
        
        x = self.model.features(x)
        out = self.out_proj(x)

        y = self.model.logits(x)
        #y = self.margin(y, label)
        #y = self.scale * F.linear(F.normalize(y), F.normalize(self.weight))

        return out, y


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,
                 n_channels=[512, 256, 128, 64, 64],
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        # TODO: insert class embedding for each layer
        self.cls_embedding = nn.Embedding(num_classes, in_channels)

        layers = []
        n_channels = [in_channels] + list(n_channels)
        for i in range(1, len(n_channels)):
            sublayer = nn.Sequential(nn.Conv2d(n_channels[i-1], n_channels[i], kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(n_channels[i]),
                                     activation,
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(n_channels[i], n_channels[i], kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(n_channels[i]),
                                     activation)
            layers.append(sublayer)
        
        self.layers = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(nn.Conv2d(n_channels[-1], out_channels, kernel_size=3, padding=1),
                                      nn.Sigmoid())

    def forward(self, x, y):
        emb = self.cls_embedding(y)
        x = x + emb.unsqueeze(-1).unsqueeze(-1)
        x = self.layers(x)
        out = self.out_proj(x)

        return out
