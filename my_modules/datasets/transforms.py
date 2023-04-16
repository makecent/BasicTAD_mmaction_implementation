# adapted from basicTAD
# https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection

from typing import Dict, Sequence

import mmcv
import numpy as np
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import BaseDataElement, InstanceData
from numpy import random

from ..models.task_modules.segments_ops import segment_overlaps
from mmengine.logging import print_log

# from mmcv.parallel import DataContainer as DC

@TRANSFORMS.register_module()
class RandSlideAug(BaseTransform):
    """Randomly slide actions' temporal location for data augmentation"""

    def __init__(self, p=0.5, extra=0.5):
        self.p = p
        self.extra = extra

    def slide_and_rearrange_segments(self, segments, total_frames, max_attempts=88):
        segments = np.round(segments).astype(int)
        mask = np.random.choice([True, False], size=segments.shape[0], p=[self.p, 1 - self.p])

        iou = segment_overlaps(segments, segments, mode='iou')
        np.fill_diagonal(iou, 0)
        mask[iou.max(axis=-1) > 0] = False  # segments overlapped with each other will NOT be slided

        images = np.arange(total_frames)
        rearranged_images = np.empty(total_frames, dtype=int)
        filled_positions = np.zeros(total_frames, dtype=bool)
        moved_positions = np.zeros(total_frames, dtype=bool)

        # Initialize rearranged_images and filled_positions with non-moving segments
        for i, (start, end) in enumerate(segments):
            moved_positions[start:end + 1] = True  # non-moving segments are treated as moved with zero offset.
            if not mask[i]:
                rearranged_images[start:end + 1] = images[start:end + 1]
                filled_positions[start:end + 1] = True

        attempt = 0
        while attempt < max_attempts:
            _rearranged_images = rearranged_images.copy()
            _filled_positions = filled_positions.copy()
            _segments = segments.copy()
            _moved_positions = moved_positions.copy()
            try:
                for i, (start, end) in enumerate(segments):
                    if mask[i]:  # Only slide segments with mask=True
                        print(f"\n moving {i}-th segment [{start}, {end}] ...")
                        _moved_positions[start: end + 1] = False
                        segment_length = end - start + 1

                        # Extend the segment by the extra factor
                        extended_start = max(0, start - int(segment_length * self.extra))
                        extended_end = min(total_frames - 1, end + int(segment_length * self.extra))
                        print(f"extended to [{extended_start}, {extended_end}] ...")

                        # Create a boolean array representing the positions of the extended segment
                        extended_indices = np.zeros(total_frames, dtype=bool)
                        extended_indices[extended_start:extended_end + 1] = True

                        # Perform an element-wise AND operation to find intersections
                        intersection = np.logical_and(_moved_positions, extended_indices)

                        # Clip the extended segment by checking the first and last intersections
                        if np.any(intersection):
                            intersecting_indices = np.where(intersection)[0]

                            # Find the closest intersection on the left and right side
                            left_intersection = intersecting_indices[intersecting_indices < start]
                            right_intersection = intersecting_indices[intersecting_indices > end]

                            if left_intersection.size > 0:
                                extended_start = min(start, left_intersection[-1] + 1)

                            if right_intersection.size > 0:
                                extended_end = max(end, right_intersection[0] - 1)
                            print(f"clip the extended segments to [{extended_start}, {extended_end}] ...")

                        # If the extended segment is entirely contained within the fixed_positions, skip this segment
                        extended_length = extended_end - extended_start + 1
                        if extended_length < segment_length:
                            raise ValueError(f"extended length {extended_length} should not be smaller than the original length {segment_length}")

                        # Find all the possible start positions for the extended segment
                        possible_starts = \
                            np.where(np.convolve(~_filled_positions, np.ones(extended_length),
                                                 mode='valid') == extended_length)[0]
                        assert possible_starts.size > 0, "unable to put a segment without overlapping"
                        # Select a random start position and update the new_segments list
                        new_start = random.choice(possible_starts)
                        new_end = new_start + extended_length - 1
                        print(f"moved to [{new_start}, {new_end}] ...")

                        # Update the new_segments array to include only the original segment (excluding extra content)
                        original_start = start + new_start - extended_start
                        original_end = end + new_end - extended_end
                        _segments[i] = [original_start, original_end]

                        # Place the extended segment into the rearranged_images array
                        _rearranged_images[new_start:new_end + 1] = images[extended_start:extended_end + 1]

                        # Update the filled and fixed positions
                        assert new_end - new_start == extended_end - extended_start
                        _filled_positions[new_start:new_end + 1] = True
                        _moved_positions[extended_start:extended_end + 1] = True

                assert np.count_nonzero(_filled_positions) == np.count_nonzero(
                    _moved_positions), f"{segments}, {_segments}, {mask}"

                # Compute the set of background indices
                background_imgs = images[np.where(~_moved_positions)[0]]
                # Fill in the remaining gaps in the rearranged_images array
                _rearranged_images[np.where(~_filled_positions)[0]] = background_imgs
                break  # successful rearrangement, exit the loop

            except AssertionError as e:
                if str(e) == "unable to put a segment without overlapping":
                    attempt += 1
                    continue
                else:
                    raise e
        if attempt == max_attempts:
            raise RuntimeError("Failed to rearrange segments after {} attempts".format(max_attempts))

        return _segments, _rearranged_images

    def transform(self, results: Dict):
        if random.uniform(0, 1) <= self.p:
            try:
                segments, img_idx_mapping = self.slide_and_rearrange_segments(results['segments'], results['total_frames'])
            except RuntimeError:
                pass
            else:
                results['segments_ori'] = results['segments']
                results['segments'] = segments.astype(np.float32)
                results['img_idx_mapping'] = img_idx_mapping
                assert np.array(segments).max() < results['total_frames']
        return results


@TRANSFORMS.register_module()
class Time2Frame(BaseTransform):
    """Switch time point to frame index."""

    def transform(self, results):
        results['segments'] = results['segments'] * results['fps']

        return results


@TRANSFORMS.register_module()
class TemporalRandomCrop(BaseTransform):
    """Temporally crop.

    Args:
        clip_len (int, optional): The cropped frame num. Default: 768.
        iof_th(float, optional): The minimal iof threshold to crop. Default: 0
    """

    def __init__(self, clip_len=96, frame_interval=10, iof_th=0.75):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.iof_th = iof_th

    def get_valid_mask(self, segments, patch, iof_th):
        gt_iofs = segment_overlaps(segments, patch, mode='iof')[:, 0]
        patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        iofs = np.maximum(gt_iofs, patch_iofs)
        mask = iofs > iof_th

        return mask

    def transform(self, results):
        """Call function to random temporally crop video frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Temporally cropped results, 'frame_inds' is updated in
                result dict.
        """
        total_frames = results['total_frames']
        ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        ori_clip_len = min(ori_clip_len, total_frames)
        while True:
            clip = np.arange(self.clip_len) * self.frame_interval
            offset = np.random.randint(0, total_frames - ori_clip_len + 1)
            clip = clip + offset
            clip = clip[clip < total_frames]
            start, end = clip[0], clip[-1]

            segments = results['segments']
            mask = self.get_valid_mask(segments, np.array([[start, end]], dtype=np.float32), self.iof_th)

            # If the cropped clip does NOT have IoF greater than the threshold with any (acknowledged) actions, then re-crop.
            if not np.logical_and(mask, np.logical_not(results['ignore_flags'])).any():
                continue

            segments = segments[mask]
            segments = segments.clip(min=start, max=end)  # TODO: Is this necessary?
            segments -= start  # transform the index of segments to be relative to the cropped segment
            segments = segments / self.frame_interval  # to be relative to the input clip
            assert segments.max() < len(clip)
            assert segments.min() >= 0

            results['segments'] = segments
            results['labels'] = results['labels'][mask]
            results['ignore_flags'] = results['ignore_flags'][mask]
            results['frame_inds'] = clip
            assert max(results['frame_inds']) < total_frames, f"offset: {offset}\n" \
                                                              f"start, end: [{start}, {end}]," \
                                                              f"total frames: {total_frames}"
            results['num_clips'] = 1
            results['clip_len'] = self.clip_len
            results['tsize'] = len(clip)

            if 'img_idx_mapping' in results:
                results['frame_inds'] = results['img_idx_mapping'][clip]
                assert results['frame_inds'].max() < results['total_frames']
                assert results['frame_inds'].min() >= 0
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_len={self.clip_len},'
        repr_str += f'(frame_interval={self.frame_interval},'
        repr_str += f'iof_th={self.iof_th})'

        return repr_str


@TRANSFORMS.register_module()
class SpatialRandomCrop(BaseTransform):
    """Spatially random crop images.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    Notes:
        - If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def transform(self, results):
        """Call function to randomly crop images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'imgs_shape' key in result dict
                is updated according to crop size.
        """
        img_h, img_w = results['img_shape']
        margin_h = max(img_h - self.crop_size[0], 0)
        margin_w = max(img_w - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop images
        imgs = [img[crop_y1:crop_y2, crop_x1:crop_x2] for img in results['imgs']]
        results['imgs'] = imgs

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to images sequentially, every
    transformation is applied with a probability of 0.5. The position of random
    contrast is in second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p

    def transform(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        imgs = np.array(results['imgs']).astype(np.float32)

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        if random.uniform(0, 1) <= self.p:

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                imgs += delta
                imgs = _filter(imgs)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # convert color from BGR to HSV
            imgs = np.array([mmcv.image.bgr2hsv(img) for img in imgs])

            # random saturation
            if random.randint(2):
                imgs[..., 1] *= random.uniform(self.saturation_lower,
                                               self.saturation_upper)

            # random hue
            # if random.randint(2):
            if True:
                imgs[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                imgs[..., 0][imgs[..., 0] > 360] -= 360
                imgs[..., 0][imgs[..., 0] < 0] += 360

            # convert color from HSV to BGR
            imgs = np.array([mmcv.image.hsv2bgr(img) for img in imgs])
            imgs = _filter(imgs)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # randomly swap channels
            if random.randint(2):
                imgs = imgs[..., random.permutation(3)]

            results['imgs'] = list(imgs)  # change back to mmaction-style (list of) imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@TRANSFORMS.register_module()
class Rotate(BaseTransform):
    """Spatially rotate images.

    Args:
        limit (int, list or tuple): Angle range, (min_angle, max_angle).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            Default: bilinear
        border_mode (str): Border mode, accepted values are "constant",
            "isolated", "reflect", "reflect_101", "replicate", "transparent",
            "wrap". Default: constant
        border_value (int): Border value. Default: 0
    """

    def __init__(self,
                 limit,
                 interpolation='bilinear',
                 border_mode='constant',
                 border_value=0,
                 p=0.5):
        if isinstance(limit, int):
            limit = (-limit, limit)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.p = p

    def transform(self, results):
        """Call function to random rotate images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Spatially rotated results.
        """

        if random.uniform(0, 1) <= self.p:
            angle = random.uniform(*self.limit)
            imgs = [
                mmcv.image.imrotate(
                    img,
                    angle=angle,
                    interpolation=self.interpolation,
                    border_mode=self.border_mode,
                    border_value=self.border_value) for img in results['imgs']]

            results['imgs'] = [np.ascontiguousarray(img) for img in imgs]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(limit={self.limit},'
        repr_str += f'interpolation={self.interpolation},'
        repr_str += f'border_mode={self.border_mode},'
        repr_str += f'border_value={self.border_value},'
        repr_str += f'p={self.p})'

        return repr_str


@TRANSFORMS.register_module()
class Pad(BaseTransform):
    """Pad images.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    @staticmethod
    def impad(img, shape, pad_val=0):
        """Pad an image or images to a certain shape.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w).
            pad_val (Number | Sequence[Number]): Values to be filled in padding
                areas. Default: 0.
        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + (img.shape[-1],)
        assert len(shape) == len(img.shape)
        for s, img_s in zip(shape, img.shape):
            assert s >= img_s, f"pad shape {s} should be greater than image shape {img_s}"
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val
        pad[:img.shape[0], :img.shape[1], :img.shape[2], ...] = img
        return pad

    @staticmethod
    def impad_to_multiple(img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.
        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (Number | Sequence[Number]): Same as :func:`impad`.
        Returns:
            ndarray: The padded image.
        """
        pad_shape = tuple(
            int(np.ceil(shape / divisor)) * divisor for shape in img.shape[:-1])
        return Pad.impad(img, pad_shape, pad_val)

    def transform(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        if self.size is not None:
            padded_imgs = self.impad(
                np.array(results['imgs']), shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_imgs = self.impad_to_multiple(
                np.array(results['imgs']), self.size_divisor, pad_val=self.pad_val)
        else:
            raise AssertionError("Either 'size' or 'size_divisor' need to be set, but both None")
        results['imgs'] = list(padded_imgs)  # change back to mmaction-style (list of) imgs
        results['pad_tsize'] = padded_imgs.shape[0]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class MyPackInputs(BaseTransform):
    mapping_table = {'segments': 'bboxes',
                     'labels': 'labels'}

    def __init__(
            self,
            meta_keys: Sequence[str] = ('video_name', 'duration', 'total_frames', 'fps',
                                        'tsize', 'pad_tsize', 'tshift', 'tscale_factor')
    ) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.
        Args:
            results (dict): The result dict.
        Returns:
            dict: The result dict.
        """

        # Pack images
        packed_results = dict()
        packed_results['inputs'] = to_tensor(results['imgs']).squeeze(dim=0)  # squeeze the `num_crops` dimension

        data_sample = BaseDataElement()

        # Pack gt_segments and gt_labels
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        assert len(results['ignore_flags']) == len(results['segments']), \
            f"There are {len(results['segments'])} segments, but {len(results['ignore_flags'])} flags"
        valid_idx = np.where(results['ignore_flags'] == 0)[0]
        ignore_idx = np.where(results['ignore_flags'] == 1)[0]

        for key in self.mapping_table.keys():
            instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
            ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])

        data_sample.gt_instances = instance_data
        # The ignored ground truth currently are not used at all. Input it to the model just for consistency with mmdet.
        data_sample.ignored_instances = ignore_instance_data

        # Pack meta
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class SpatialCenterCrop(BaseTransform):
    """Spatially center crop images.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).

    Notes:
        - If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def transform(self, results):
        """Call function to center crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'imgs_shape' key in result dict
                is updated according to crop size.
        """

        imgs = np.array(results['imgs'])
        margin_h = max(imgs.shape[1] - self.crop_size[0], 0)
        margin_w = max(imgs.shape[2] - self.crop_size[1], 0)
        offset_h = int(margin_h / 2)
        offset_w = int(margin_w / 2)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop images
        imgs = imgs[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
        results['imgs'] = list(imgs)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
