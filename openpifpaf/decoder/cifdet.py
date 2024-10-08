from collections import defaultdict
import logging
import time
from typing import List

import torch
import torchvision

from openpifpaf.decoder import Decoder
from openpifpaf.annotation import AnnotationDet
from openpifpaf import headmeta, visualizer

LOG = logging.getLogger(__name__)


class CifDet(Decoder):
    iou_threshold = 0.5
    instance_threshold = 0.15  # confidence score threshold
    nms_by_category = True
    occupancy_visualizer = visualizer.Occupancy()
    suppression = 0.1  # NMS threshold

    # non-important persons filter
    # can be called directly by 'self.'
    # filter_lw = 0
    # filter_rw = 0
    # filter_th = 280
    # filter_bh = 700


    def __init__(self, head_metas: List[headmeta.CifDet], *, visualizers=None):
        super().__init__()
        self.metas = head_metas

        # prefer decoders with more classes
        self.priority = -1.0  # prefer keypoints over detections
        self.priority += sum(m.n_fields for m in head_metas) / 1000.0

        self.visualizers = visualizers
        if self.visualizers is None:
            self.visualizers = [visualizer.CifDet(meta) for meta in self.metas]

        self.cpp_decoder = torch.classes.openpifpaf_decoder.CifDet()

        self.timers = defaultdict(float)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            CifDet([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.CifDet)
        ]

    def __call__(self, fields):
        start = time.perf_counter()

        if self.visualizers:
            for vis, meta in zip(self.visualizers, self.metas):
                vis.predicted(fields[meta.head_index])

        categories, scores, boxes = self.cpp_decoder.call(
            fields[self.metas[0].head_index],
            self.metas[0].stride,
        )  # cppdecoder  所有score和bbox的坐标全部返回

        # nms
        if self.nms_by_category:
            keep_index = torchvision.ops.batched_nms(boxes, scores, categories, self.iou_threshold)  # 框都得到了再进行nms，score可以视为置信度
        else:
            keep_index = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        pre_nms_scores = scores.clone()
        scores *= self.suppression
        scores[keep_index] = pre_nms_scores[keep_index]
        filter_mask = scores > self.instance_threshold
        categories = categories[filter_mask]
        scores = scores[filter_mask]
        boxes = boxes[filter_mask]

        LOG.debug('cpp annotations = %d (%.1fms)',
                  len(scores),
                  (time.perf_counter() - start) * 1000.0)

        # convert to py
        # packed as annotation object (dict)
        annotations_py = []
        boxes_np = boxes.numpy()
        boxes_np[:, 2:] -= boxes_np[:, :2]  # convert to xywh

        # filter non-important persons -> filter in output step (predictor...) with w and h
        # boxes_np = boxes_np[(boxes_np[:, 1] > self.filter_th) & ((boxes_np[:, 1] + boxes_np[:, 3]) < self.filter_bh)]
        # boxes_np = boxes_np[(boxes_np[:, 1] + boxes_np[:, 3] > 435) & ((boxes_np[:, 1] + boxes_np[:, 3]) < 640)]
        # boxes_np = boxes_np[(boxes_np[:, 1] + boxes_np[:, 3] > 600) & ((boxes_np[:, 1] + boxes_np[:, 3]) < 1000)]

        for category, score, box in zip(categories, scores, boxes_np):
            ann = AnnotationDet(self.metas[0].categories)
            ann.set(int(category), float(score), box)
            annotations_py.append(ann)

        LOG.info('annotations %d, decoder = %.1fms',
                 len(annotations_py),
                 (time.perf_counter() - start) * 1000.0)
        return annotations_py
