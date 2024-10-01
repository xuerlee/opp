import argparse
import logging
import time
from typing import List

import numpy as np
import torch

from openpifpaf.decoder.decoder import Decoder
from openpifpaf.annotation import Annotation
from openpifpaf.decoder import utils
from openpifpaf import headmeta, visualizer

LOG = logging.getLogger(__name__)


class CifCafDense(Decoder):
    dense_coupling = 0.0

    def __init__(self,
                 cif_meta: headmeta.Cif,
                 caf_meta: headmeta.Caf,
                 dense_caf_meta: headmeta.Caf):
        super().__init__()

        self.cif_meta = cif_meta
        self.caf_meta = caf_meta
        self.dense_caf_meta = dense_caf_meta

        # prefer decoders with more keypoints and associations
        self.priority += cif_meta.n_fields / 1000.0
        self.priority += caf_meta.n_fields / 1000.0
        self.priority += dense_caf_meta.n_fields / 1000.0

        # overwrite confidence scale
        self.dense_caf_meta.confidence_scales = [
            self.dense_coupling for _ in self.dense_caf_meta.skeleton
        ]

        concatenated_caf_meta = headmeta.Caf.concatenate(
            [caf_meta, dense_caf_meta])
        self.cifcaf = CifCaf([cif_meta], [concatenated_caf_meta])

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('CifCafDense decoder')
        group.add_argument('--dense-connections', nargs='?', type=float,
                           default=0.0, const=1.0)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.dense_coupling = args.dense_connections

    @classmethod
    def factory(cls, head_metas):
        if len(head_metas) < 3:
            return []
        if not cls.dense_coupling:
            return []
        return [
            CifCafDense(cif_meta, caf_meta, dense_meta)
            for cif_meta, caf_meta, dense_meta in zip(head_metas, head_metas[1:], head_metas[2:])
            if (isinstance(cif_meta, headmeta.Cif)
                and isinstance(caf_meta, headmeta.Caf)
                and isinstance(dense_meta, headmeta.Caf))
        ]

    def __call__(self, fields, initial_annotations=None):
        cifcaf_fields = [
            fields[self.cif_meta.head_index],
            np.concatenate([
                fields[self.caf_meta.head_index],
                fields[self.dense_caf_meta.head_index],
            ], axis=0)
        ]
        return self.cifcaf(cifcaf_fields)


'''
pif, 原始的输出共有4个, 分别为:
    joint_intensity_fields, shape 为 [17, output_h, output_w]. 其实就是输出的每个位置上的confidence map, 17表示channel数, 在pose检测里面表示总共有多少个关键点需要检测.
    joint_offset_fields, shape 为[17, 2, output_h, output_w]. 为对应位置上的离其最近的关节点位置的偏移量. 这个是学习得到的, 2表示是两个方向(x, y)的偏移量. 所以关节点的真正位置需要把该位置的(x, y)和其两个方向的(x_offset, y_offset)相加起来得到.
    joint_b, shape为[17, output_h, output_w]. 论文里提到的spread b,是自适应并且经过网络学习得到的, 用来参与loss计算, 在decode的时候并没有用到.
    joint_scale_fields. shape为[17, output_h, output_w]. 自适应的scale值, 用来表明该关键点的scale大小.不确定是否有用在loss计算里. decode的时候则是作为类似gaussian的sigma值参与decode过程.

paf, 原始的输出共有5个, 按照顺序为: (首先说明下, 论文提出的paf和之前OpenPose及PersonLab提出的连接方式都不一样. 该论文提出的paf连接为, 每个位置预测出哪两个点需要连接在一起, 因此不是单纯的两个关节点之间的直接连接, 而是经过了另外一个位置进行第三方连接)
    joint_intensity_fields, shape为[19, output_h, output_w]. 19表明共有多少个连接需要学习, 对应的是每个输出位置上的paf的confidence值
    joint1_fields, shape为[19, 2, output_h, output_w]. 这个位置表明的两个可以连接在一起的点中的第一个点的信息, 其实就是偏移值, (x_offset, y_offset).
    joint2_fields, shape为[19, 2, output_h, output_w]. 同上, 表示的是一条线段上的第二个点的偏移值.
    joint1_fields_logb, shape为[19, output_h, output_w]. 论文里提到的spread b,是joint1的, 用来参与loss计算和decode. 根据decode的过程来看, 网络输出的这个值是经过log计算后的, 所以叫做logb,在decode的时候需要先exp还原.
    joint2_fields_logb, shape为[19, output_h, output_w]. 同上, 只不过变成是第二个点的b了.
'''


class CifCaf(Decoder):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    nms_before_force_complete = False

    reverse_match = True

    def __init__(self,
                 cif_metas: List[headmeta.Cif],
                 caf_metas: List[headmeta.Caf],
                 *,
                 cif_visualizers=None,
                 cifhr_visualizers=None,
                 caf_visualizers=None):
        super().__init__()

        self.cif_metas = cif_metas  # 提取两个head的meta
        self.caf_metas = caf_metas
        self.score_weights = cif_metas[0].score_weights
        self.confidence_scales = caf_metas[0].decoder_confidence_scales  # 基本都没定义

        self.cif_visualizers = cif_visualizers
        if self.cif_visualizers is None:
            self.cif_visualizers = [visualizer.Cif(meta) for meta in cif_metas]  # cif可视化
        self.cifhr_visualizers = cifhr_visualizers
        if self.cifhr_visualizers is None:
            self.cifhr_visualizers = [
                visualizer.CifHr(stride=meta.stride, field_names=meta.keypoints)  # cif heatmap可视化
                for meta in cif_metas
            ]
        self.caf_visualizers = caf_visualizers  # caf 可视化
        if self.caf_visualizers is None:
            self.caf_visualizers = [visualizer.Caf(meta) for meta in caf_metas]

        self.cpp_decoder = torch.classes.openpifpaf_decoder.CifCaf(
            len(cif_metas[0].keypoints),
            torch.LongTensor(caf_metas[0].skeleton) - 1,
        )  # 直接用torch函数decoder了？
        # prefer decoders with more keypoints and associations
        self.priority += sum(m.n_fields for m in cif_metas) / 1000.0
        self.priority += sum(m.n_fields for m in caf_metas) / 1000.0

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        CppCifCaf = torch.classes.openpifpaf_decoder.CifCaf  # pylint: disable=invalid-name

        group = parser.add_argument_group('CifCaf decoder')
        assert not CppCifCaf.get_force_complete()
        group.add_argument('--force-complete-pose',
                           default=False, action='store_true')
        group.add_argument('--force-complete-caf-th', type=float,
                           default=CppCifCaf.get_force_complete_caf_th(),
                           help='CAF threshold for force complete. Set to -1 to deactivate.')
        assert not cls.nms_before_force_complete
        group.add_argument('--nms-before-force-complete', default=False, action='store_true',
                           help='run an additional NMS before completing poses')

        assert utils.nms.Keypoints.get_keypoint_threshold() == CppCifCaf.get_keypoint_threshold()
        group.add_argument('--keypoint-threshold', type=float,
                           default=CppCifCaf.get_keypoint_threshold(),
                           help='filter keypoints by score')
        group.add_argument('--keypoint-threshold-rel', type=float,
                           default=CppCifCaf.get_keypoint_threshold_rel(),
                           help='filter keypoint connections by relative score')

        assert not CppCifCaf.get_greedy()
        group.add_argument('--greedy', default=False, action='store_true',
                           help='greedy decoding')
        group.add_argument('--connection-method',
                           default=cls.connection_method,
                           choices=('max', 'blend'),
                           help='connection method to use, max is faster')
        assert not CppCifCaf.get_block_joints()
        group.add_argument('--cifcaf-block-joints', default=False, action='store_true',
                           help='block joints')

        assert CppCifCaf.get_reverse_match()
        group.add_argument('--no-reverse-match',
                           default=True, dest='reverse_match', action='store_false')
        group.add_argument('--ablation-cifseeds-nms',
                           default=False, action='store_true')
        group.add_argument('--ablation-cifseeds-no-rescore',
                           default=False, action='store_true')
        group.add_argument('--ablation-caf-no-rescore',
                           default=False, action='store_true')
        group.add_argument('--ablation-independent-kp',
                           default=False, action='store_true')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        CppCifCaf = torch.classes.openpifpaf_decoder.CifCaf  # pylint: disable=invalid-name

        # force complete
        keypoint_threshold_nms = args.keypoint_threshold
        if args.force_complete_pose:
            if not args.ablation_independent_kp:
                args.keypoint_threshold = 0.0
            args.keypoint_threshold_rel = 0.0
            keypoint_threshold_nms = 0.0
        # check consistency
        if args.seed_threshold < args.keypoint_threshold:
            LOG.warning(
                'consistency: decreasing keypoint threshold to seed threshold of %f',
                args.seed_threshold,
            )
            args.keypoint_threshold = args.seed_threshold

        cls.nms_before_force_complete = args.nms_before_force_complete
        utils.nms.Keypoints.set_keypoint_threshold(keypoint_threshold_nms)

        CppCifCaf.set_force_complete(args.force_complete_pose)
        CppCifCaf.set_force_complete_caf_th(args.force_complete_caf_th)
        CppCifCaf.set_keypoint_threshold(args.keypoint_threshold)
        CppCifCaf.set_keypoint_threshold_rel(args.keypoint_threshold_rel)

        CppCifCaf.set_greedy(args.greedy)
        CppCifCaf.set_block_joints(args.cifcaf_block_joints)
        cls.connection_method = args.connection_method

        cls.reverse_match = args.reverse_match
        utils.CifSeeds.set_ablation_nms(args.ablation_cifseeds_nms)
        utils.CifSeeds.set_ablation_no_rescore(args.ablation_cifseeds_no_rescore)
        utils.CafScored.set_ablation_no_rescore(args.ablation_caf_no_rescore)
        if args.ablation_cifseeds_no_rescore and args.ablation_caf_no_rescore:
            utils.CifHr.set_ablation_skip(True)

    @classmethod
    def factory(cls, head_metas):
        if CifCafDense.dense_coupling:
            return []  # --dense-connections is requested, so use the other decoder
        return [
            CifCaf([meta], [meta_next])
            for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
            if (isinstance(meta, headmeta.Cif)
                and isinstance(meta_next, headmeta.Caf))
        ]

    def __call__(self, fields, initial_annotations=None):
        if not initial_annotations:
            initial_annotations_t = None
            initial_ids_t = None
        else:
            initial_annotations_t = torch.empty(
                (len(initial_annotations), self.cif_metas[0].n_fields, 4))
            initial_ids_t = torch.empty((len(initial_annotations),), dtype=torch.int64)
            for i, (ann_py, ann_t) in enumerate(zip(initial_annotations, initial_annotations_t)):
                for f in range(len(ann_py.data)):
                    ann_t[f, 0] = float(ann_py.data[f, 2])
                    ann_t[f, 1] = float(ann_py.data[f, 0])
                    ann_t[f, 2] = float(ann_py.data[f, 1])
                    ann_t[f, 3] = float(ann_py.joint_scales[f])
                initial_ids_t[i] = getattr(ann_py, 'id_', -1)
            LOG.debug('initial annotations = %d', initial_annotations_t.size(0))

        for vis, meta in zip(self.cif_visualizers, self.cif_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.caf_visualizers, self.caf_metas):
            vis.predicted(fields[meta.head_index])

        start = time.perf_counter()
        annotations, annotation_ids = self.cpp_decoder.call_with_initial_annotations(
            fields[self.cif_metas[0].head_index],
            self.cif_metas[0].stride,
            fields[self.caf_metas[0].head_index],
            self.caf_metas[0].stride,
            initial_annotations_t,
            initial_ids_t,
        )
        LOG.debug('cpp annotations = %d (%.1fms)',
                  len(annotations),
                  (time.perf_counter() - start) * 1000.0)
        for vis in self.cifhr_visualizers:
            fields, low = self.cpp_decoder.get_cifhr()
            vis.predicted(fields, low)

        annotations_py = []
        for ann_data, ann_id in zip(annotations, annotation_ids):
            ann = Annotation(self.cif_metas[0].keypoints,
                             self.caf_metas[0].skeleton,
                             score_weights=self.score_weights)
            ann.data[:, :2] = ann_data[:, 1:3]
            ann.data[:, 2] = ann_data[:, 0]
            ann.joint_scales[:] = ann_data[:, 3]
            if ann_id != -1:
                ann.id_ = int(ann_id)
            annotations_py.append(ann)

        LOG.info('annotations %d: %s',
                 len(annotations_py),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations_py])
        return annotations_py  # 输出为ann格式
