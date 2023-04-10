from typing import Dict
from typing import List, Tuple, Union

import torch
from mmaction.registry import MODELS
from mmdet.models import SingleStageDetector
from mmdet.structures import DetDataSample
from mmdet.structures import OptSampleList, SampleList
from torch import Tensor

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class SegmentDetector(SingleStageDetector):
    """
    Modify the default argument to support temporal action detection on Thumos14
    """

    def __init__(self,
                 backbone=dict(type='SlowOnly_96win'),
                 neck=[dict(type='VDM'), dict(type='FPN')],
                 bbox_head=dict(type='RetinaHead1D'),
                 data_preprocessor=dict(
                     type='ActionDataPreprocessor',
                     mean=[123.675, 116.28, 103.53],
                     std=[58.395, 57.12, 57.375],
                     format_shape='NCTHW'),
                 train_cfg=dict(
                     assigner=dict(
                         type='mmdet.MaxIoUAssigner',
                         pos_iou_thr=0.6,
                         neg_iou_thr=0.4,
                         min_pos_iou=0,
                         ignore_iof_thr=-1,
                         ignore_wrt_candidates=True,
                         iou_calculator=dict(type='SegmentOverlaps')),
                     allowed_border=-1,
                     pos_weight=-1,
                     debug=False),
                 test_cfg=dict(nms_pre=300, score_thr=0.005),
                 **kwargs):
        super(SingleStageDetector, self).__init__(data_preprocessor=data_preprocessor, **kwargs)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                with_nms=False,  # -------------------
                rescale=True) -> ForwardResults:  # -------------------
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, with_nms=with_nms, rescale=rescale)  # -----------------
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'loss_and_predict':  # ---------------------
            return self.loss_and_predict(inputs, data_samples, with_nms=with_nms, rescale=rescale)  # -----------------
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss_and_predict(self, batch_inputs, batch_data_samples, with_nms=True, rescale=False):
        x = self.extract_feat(batch_inputs)
        losses, predictions = self.bbox_head.loss_and_predict(x, batch_data_samples, with_nms=with_nms, rescale=rescale)
        return losses, predictions

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                with_nms: bool = False,  # ---------------------
                rescale: bool = True) -> SampleList:

        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, with_nms=with_nms, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
