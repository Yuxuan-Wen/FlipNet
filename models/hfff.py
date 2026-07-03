import torch.nn as nn
import torch
import torch.nn.functional as F

from models.registry import NET
from .resnet import ResNetWrapper
from .decoder import BUSD, PlainDecoder


def _swap_blocks(seq, groups):

    n = len(seq)
    remainder = n % groups
    if remainder == 0:
        block = n // groups
        out = []
        for b in range(0, groups, 2):
            out.extend(seq[(b + 1) * block:(b + 2) * block])
            out.extend(seq[b * block:(b + 1) * block])
        return out
    prefix, rest = seq[:remainder], seq[remainder:]
    return _swap_blocks(prefix, 2) + _swap_blocks(rest, groups)


def _flip_indices(length, groups_coarse_to_fine):

    seq = list(range(length))
    indices = [torch.tensor(_swap_blocks(seq, groups)) for groups in groups_coarse_to_fine]
    indices.reverse()
    return indices


class HFFF(nn.Module):
    """Hierarchical feature flip fusion module"""

    def __init__(self, cfg):
        super(HFFF, self).__init__()
        self.iter = cfg.hfff.iter
        chan = cfg.hfff.input_channel
        fea_stride = cfg.backbone.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.hfff.alpha
        conv_stride = cfg.hfff.conv_stride

        assert len(cfg.hfff.vert_groups) == self.iter, \
            'cfg.hfff.vert_groups must have cfg.hfff.iter entries'
        assert len(cfg.hfff.hori_groups) == self.iter, \
            'cfg.hfff.hori_groups must have cfg.hfff.iter entries'

        for i in range(self.iter):
            setattr(self, 'conv_vert_' + str(i), nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False))

        for i in range(self.iter):
            setattr(self, 'conv_hori_' + str(i), nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False))

        vert_indices = _flip_indices(self.height, cfg.hfff.vert_groups)
        for i, idx in enumerate(vert_indices):
            setattr(self, 'idx_vert_' + str(i), idx)

        hori_indices = _flip_indices(self.width, cfg.hfff.hori_groups)
        for i, idx in enumerate(hori_indices):
            setattr(self, 'idx_hori_' + str(i), idx)

    def forward(self, x):
        x = x.clone()

        for i in range(self.iter):
            conv = getattr(self, 'conv_vert_' + str(i))
            idx = getattr(self, 'idx_vert_' + str(i))
            x.add_(self.alpha * F.relu(conv(x[..., idx, :])))

        for i in range(self.iter):
            conv = getattr(self, 'conv_hori_' + str(i))
            idx = getattr(self, 'idx_hori_' + str(i))
            x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x


class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)

        stride = cfg.backbone.fea_stride * 2
        self.fc9 = nn.Linear(
            int(cfg.num_classes * cfg.img_width / stride * cfg.img_height / stride), 128)
        self.fc10 = nn.Linear(128, cfg.num_classes - 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)

        x = F.softmax(x, dim=1)
        x = F.avg_pool2d(x, 2, stride=2, padding=0)
        x = x.view(-1, x.numel() // x.shape[0])
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = torch.sigmoid(x)

        return x


@NET.register_module
class FlipNet(nn.Module):
    def __init__(self, cfg):
        super(FlipNet, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper(cfg)
        self.hfff = HFFF(cfg)
        self.decoder = eval(cfg.decoder)(cfg)
        self.heads = ExistHead(cfg)

    def forward(self, batch):
        fea = self.backbone(batch)
        fea = self.hfff(fea)
        seg = self.decoder(fea)
        exist = self.heads(fea)

        output = {'seg': seg, 'exist': exist}

        return output
