from collections import OrderedDict

import torch
import torch.nn as nn
from mmaction.registry import MODELS
from torch.nn import ModuleList


@MODELS.register_module()
class SlowOnly_96win(nn.Module):

    def __init__(self, num_layers=50, freeze_bn=True, freeze_bn_affine=True):
        super(SlowOnly_96win, self).__init__()
        model_name = 'slow_r' + str(num_layers)
        model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
        self.blocks = ModuleList()
        for i in range(5):
            self.blocks.append(model._modules['blocks'][i])
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def train(self, mode=True):
        super(SlowOnly_96win, self).train(mode)
        if self._freeze_bn and mode:
            for name, m in self.named_modules():
                if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.register_hook(lambda grad: torch.zeros_like(grad))
                        m.bias.register_hook(lambda grad: torch.zeros_like(grad))
