import copy
import warnings
from collections import OrderedDict
from typing import Optional, Sequence, Union, List

import numpy as np
import torch
from mmaction.registry import METRICS
from mmcv.ops import batched_nms
from mmdet.evaluation.functional import eval_map, eval_recalls
from mmdet.structures.bbox import bbox_overlaps
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from ..models.task_modules.segments_ops import batched_nmw


@METRICS.register_module()
class BasicTADMetric(BaseMetric):
    """BasicTAD evolution metric.
    This metric differ from the MMDetection metrics: it performs post-processing.
    Specifically, the MMDetection metrics take as input the processed results, by assuming that the post-processing
    was already done in the model's detection head during inference.
    We move the post-processing to the Metric because BasicTAD models often cannot process a test video in one pass,
    but have to split it into several overlapped sub-videos and process them separately.
    Thus, we have to first merge the detection results of sub-videos from the same videos and then perform NMS globally.
    Consequently, this metric receive arguments about post-processing, include:
        (a) the config of NMS (nms_cfg),
        (b) the score threshold for filtering poor detections (score_thr),
        (c) the maximum number of detections in each video (max_per_video).
    In addition to the above common post-processing arguments, we add two extra arguments:
        (d) the duration threshold for filtering short detections (duration_thr),
        (e) the flag indicating whether to perform NMS on overlapped regions in testing videos (nms_in_overlap).
    Furthermore, we support the NMW (Non-Maximum Weighted) NMS, which is used in BasicTAD and claimed to be better.
    However, the NMW is slow compared to NMS as it is not optimized, implemented by native Python.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        nms_cfg (dict): Config of NMS, which is used to remove redundant detections.
            Default: dict(type='nms', iou_thr=0.4).
        max_per_video (int): Maximum number of detections in each video.
            Default: False.
        score_thr (float): Score threshold for filtering poor detections.
            Default: 0.0.
        duration_thr (float): Duration threshold for filtering short detections (in second).
            Default: 0.0.
        nms_in_overlap (bool): Whether to perform NMS on overlapped regions in testing videos.
            Default: False.
    """
    default_prefix: Optional[str] = 'tad'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 # Post-processing arguments:
                 nms_cfg=dict(type='nms', iou_thr=0.4),
                 max_per_video: int = False,
                 score_thr=0.0,
                 duration_thr=0.0,
                 nms_in_overlap=False):
        super().__init__()
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs

        self.nms_cfg = nms_cfg
        self.max_per_video = max_per_video
        self.score_thr = score_thr
        self.duration_thr = duration_thr
        self.nms_in_overlap = nms_in_overlap
        if nms_cfg.get('type') in ['nms', 'soft_nms']:
            self.nms = batched_nms
        elif nms_cfg.get('type') == 'nmw':
            warnings.warn(f'NMW is used, which is slow compared to NMS as it is not optimized, implemented by Python.')
            self.nms = batched_nmw
        else:
            NotImplementedError(f'NMS type {nms_cfg.get("type")} is not implemented.')

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            data = copy.deepcopy(data_sample)
            gts, dets = data['gt_instances'], data['pred_instances']
            gts_ignore = data.get('ignored_instances', dict())
            ann = dict(
                video_name=data['img_id'],  # for the purpose of future grouping detections of same video.
                labels=gts['labels'].cpu().numpy(),
                bboxes=gts['bboxes'].cpu().numpy(),
                bboxes_ignore=gts_ignore.get('bboxes', torch.empty((0, 4))).cpu().numpy(),
                labels_ignore=gts_ignore.get('labels', torch.empty(0, )).cpu().numpy())

            if self.nms_in_overlap:
                ann['overlap'] = data['overlap'],  # for the purpose of NMS on overlapped region in testing videos

            # Convert the format of segment predictions from feature-unit to second-unit (add window-offset back first).
            if 'offset_sec' in data:
                dets['bboxes'] = dets['bboxes'] + data['offset_sec']

            # Set y1, y2 of predictions the fixed value.
            dets['bboxes'][:, 1] = 0.1
            dets['bboxes'][:, 3] = 0.9

            # Filter out predictions with low scores
            valid_inds = dets['scores'] > self.score_thr

            # Filter out predictions with short duration
            valid_inds &= (dets['bboxes'][:, 2] - dets['bboxes'][:, 0]) > self.duration_thr

            dets['bboxes'] = dets['bboxes'][valid_inds].cpu()
            dets['scores'] = dets['scores'][valid_inds].cpu()
            dets['labels'] = dets['labels'][valid_inds].cpu()

            # Format predictions to InstanceData
            dets = InstanceData(**dets)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        logger.info(f'\n Merging the results of same videos...')
        gts, preds = self.merge_results_of_same_video(gts, preds)
        logger.info(f'\n Performing Non-maximum suppression (NMS) ...')
        preds = self.non_maximum_suppression(preds)

        eval_results = OrderedDict()
        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_map(
                preds,
                gts,
                iou_thr=iou_thr,
                dataset=self.dataset_meta['classes'],
                logger=logger)
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results.move_to_end('mAP', last=False)
        return eval_results

    @staticmethod
    def merge_results_of_same_video(gts, preds):
        # Merge prediction results from the same videos because we use sliding windows to crop the testing videos
        # Also known as the Cross-Window Fusion (CWF)
        video_names = list(dict.fromkeys([gt['video_name'] for gt in gts]))

        merged_gts_dict = dict()
        merged_preds_dict = dict()
        for this_gt, this_pred in zip(gts, preds):
            video_name = this_gt.pop('video_name')
            # Computer the mask indicating that if a prediction is in the overlapped regions.
            overlap_regions = this_gt.pop('overlap', np.empty([0]))
            if overlap_regions.size == 0:
                this_pred.in_overlap = np.zeros(this_pred.bboxes.shape[0], dtype=bool)
            else:
                this_pred.in_overlap = bbox_overlaps(this_pred.bboxes, torch.from_numpy(overlap_regions)) > 0

            merged_preds_dict.setdefault(video_name, []).append(this_pred)
            merged_gts_dict.setdefault(video_name, this_gt)  # the gt is video-wise thus no need concatenation

        # dict of list to list of dict
        merged_gts = []
        merged_preds = []
        for video_name in video_names:
            merged_gts.append(merged_gts_dict[video_name])
            # Concatenate detection in windows of the same video
            merged_preds.append(InstanceData.cat(merged_preds_dict[video_name]))
        return merged_gts, merged_preds

    def non_maximum_suppression(self, preds):
        preds_nms = []
        for pred_v in preds:
            if self.nms_cfg is not None:
                if self.nms_in_overlap:
                    if pred_v.in_overlap.sum() > 1:
                        # Perform NMS among predictions in each overlapped region
                        pred_not_in_overlaps = pred_v[~pred_v.in_overlap.max(-1)[0]]
                        pred_in_overlaps = []
                        for i in range(pred_v.in_overlap.shape[1]):
                            pred_in_overlap = pred_v[pred_v.in_overlap[:, i]]
                            if len(pred_in_overlap) == 0:
                                continue
                            bboxes, keep_idxs = self.nms(pred_in_overlap.bboxes,
                                                         pred_in_overlap.scores,
                                                         pred_in_overlap.labels,
                                                         nms_cfg=self.nms_cfg)
                            pred_in_overlap = pred_in_overlap[keep_idxs]
                            pred_in_overlap.bboxes = bboxes[:, :-1]
                            pred_in_overlap.scores = bboxes[:, -1]
                            pred_in_overlaps.append(pred_in_overlap)
                        pred_v = InstanceData.cat(pred_in_overlaps + [pred_not_in_overlaps])
                else:
                    bboxes, keep_idxs = self.nms(pred_v.bboxes,
                                                 pred_v.scores,
                                                 pred_v.labels,
                                                 nms_cfg=self.nms_cfg)
                    pred_v = pred_v[keep_idxs]
                    # Some NMS operations will change the value of scores and bboxes, we track it.
                    pred_v.bboxes = bboxes[:, :-1]
                    pred_v.scores = bboxes[:, -1]
            sort_idxs = pred_v.scores.argsort(descending=True)
            pred_v = pred_v[sort_idxs]
            # keep top-k predictions
            if self.max_per_video:
                pred_v = pred_v[:self.max_per_video]

            # Reformat predictions to meet the requirement of eval_map function: VideoList[ClassList[PredictionArray]]
            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_v.labels == label)[0]
                pred_bbox_with_scores = np.hstack(
                    [pred_v[index].bboxes, pred_v[index].scores.reshape((-1, 1))])
                dets.append(pred_bbox_with_scores)

            preds_nms.append(dets)
        return preds_nms
