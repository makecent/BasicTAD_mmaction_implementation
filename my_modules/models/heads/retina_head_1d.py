# Adapted from https://github.com/MCG-NJU/BasicTAD/
# https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection

import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmdet.models.dense_heads import RetinaHead


@MODELS.register_module()
class RetinaHead1D(RetinaHead):
    r"""Modified RetinaHead to support 1D
    """

    def _init_layers(self):
        super()._init_layers()
        # Change the cls head and reg head to be based on Conv1d instead of Conv2d
        self.retina_cls = nn.Conv1d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv1d(
            self.feat_channels, self.num_base_priors * 2, 3, padding=1)

    def forward_single(self, x):
        cls_score, bbox_pred = super().forward_single(x)
        # add pseudo H dimension
        cls_score, bbox_pred = cls_score.unsqueeze(-2), bbox_pred.unsqueeze(-2)
        # bbox_pred = [N, 2], where 2 is the x, w. Now adding pseudo y, h
        bbox_pred = bbox_pred.unflatten(1, (self.num_base_priors, -1))
        y, h = torch.split(torch.zeros_like(bbox_pred), 1, dim=2)
        bbox_pred = torch.cat((bbox_pred[:, :, :1, :, :], y, bbox_pred[:, :, 1:, :, :], h), dim=2)
        bbox_pred = bbox_pred.flatten(start_dim=1, end_dim=2)
        return cls_score, bbox_pred
