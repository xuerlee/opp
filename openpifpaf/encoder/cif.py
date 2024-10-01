import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

from .annrescaler import AnnRescaler

from openpifpaf import headmeta
from openpifpaf. visualizer import Cif as CifVisualizer
from openpifpaf. utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)

'''
定义数据处理方法 eg.: cocokp.py: 
encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
依然用到了plugin.cocokp中设置的headmeta，传递到class cif: meta: headmeta.Cif
该处返回的transform.encoder中调用了call函数
在plugin.cocokp中的train_loader中直接导入image和ann进行数据的处理
该函数的主要作用是生成label
'''
@dataclasses.dataclass
class Cif:
    meta: headmeta.Cif
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4  # 用于预测的grid的边长为side_length的范围，物体所在的中心在该范围的中心
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return CifGenerator(self)(image, anns, meta)


class CifGenerator():
    def __init__(self, config: Cif):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose)  # meta的定义：headmeta.py
        self.visualizer = config.visualizer or CifVisualizer(config.meta)

        self.intensities = None
        self.fields_reg = None
        self.fields_bmin = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0  # 物体中心距离离它最近的负责预测的grid之间的距离

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        # 生成的bg_mask是对那些没有标注任何可见的关节点的区域(以一个人的box为范围)标注为0,
        # 其余为1. 意思就是这个人有box, 但box里面没有任何一个可见的关节点(就是所有关节点的confidence都为0), 那这个box区域内的值就全为0.
        valid_area = self.rescaler.valid_area(meta)  # 好像没有
        # 就是让image处于valid_area之外的区域值为0, 意思就是让网络专注于去学只有人的区域, 非人的区域不要去学
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = len(self.config.meta.keypoints)  # 每个target有多少个keypoint就有几个field
        self.init_fields(n_fields, bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, n_fields, bg_mask):  # 针对某一个target的所有keypoint的fields
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):  # 针对某一个目标的所有keypoints
        scale = self.rescaler.scale(keypoints)
        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.config.v_threshold:
                continue

            joint_scale = (
                scale
                if self.config.meta.sigmas is None
                else scale * self.config.meta.sigmas[f]
            )  # 放缩关键点大小

            self.fill_coordinate(f, xyv, joint_scale)

    '''
    接着, 就是根据ann值生成pif label. 
    对于同一个人的关节点, 其scale值是一样的, 为当前可见的关节点所表示的(maxx-minx, maxy-miny)区域的平方根. 
    主要是函数fill_coordinate是将每一个单独的关节点放入到pif label中. 
    首先是将以关节点(x,y)为中心的, 大小为(self.side_length, self.side_length)范围内的intensities矩阵值设为1, 对应的最后一层背景channel相同位置设置为0. 
    (pif的intensities矩阵更有(n_joints+1)个channel, 最后一个channel为背景类别.) 接着, 计算这个范围内的点到(x, y)的x_offset和y_offset. 
    如果遇到多个关节点范围重合, 那么这个重合范围内的点的offset值应该是其离最近的关节点的offset值, 这个就是函数fill_coordinate最后几行代码的作用. 接着更新对应的field_scale矩阵.
    '''

    def fill_coordinate(self, f, xyv, scale):
        ij = np.round(xyv[:2] - self.s_offset).astype(int) + self.config.padding  # 离物体中心最近的负责预测的grid的坐标
        minx, miny = int(ij[0]), int(ij[1])  # 调整后的keypoint最小位置
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length  # keypoint最大位置  # 用于预测的grid根据self.config.side_length产生
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)  # padding后距离最近的格子中心到实际位置的偏置，即把物体实际中心取整到格点上了。
        # 之后所有的负责预测的Grid都算上这个偏置,这样就得到了关于每一个负责预测的grid的GT
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset  # 离得较近的格子都应该预测该关键点  # 不是偏置，是这个格子的位置到keypoint位置的vector
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        mask_peak = np.logical_and(mask, sink_l < 0.7)
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]  # fields_reg_l：位置偏置的L1norm GT

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0  # 在这里才又把负责预测关键点的地方设置为1，其余都是0
        self.intensities[f, miny:maxy, minx:maxx][mask_peak] = 1.0

        # update regression
        patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
        patch[:, mask] = sink_reg[:, mask]  # # fields_reg：位置偏置的GT

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin[f, miny:maxy, minx:maxx][mask] = bmin

        # update scale
        assert np.isnan(scale) or 0.0 < scale < 100.0
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    '''
    作者可能是为了提高精度, 额外在初始化各个矩阵的时候增加了padding, 因此再取这些矩阵的时候, 需要将padding范围去掉, 就是函数self.fields(self, valid_area)函数的作用. 
    最终, 会返回三个torch tensor, intensities, shape为[18, h, w], 表明pif label 的confidence; 
    fields_reg, shape为[17, 2, h, w], 表明每个位置上的offset；
    fields_scale, shape为[17, h, w], 表明每个位置的scale. 
    发现并没有decode过程中的spread b值, 因为这个值是网络自己学到的, 是通过设计的L1-loss学习的, 因此pif label里并没有这个值.
    '''

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_bmin = self.fields_bmin[:, p:-p, p:-p]
        fields_scale = self.fields_scale[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg,
            np.expand_dims(fields_bmin, 1),
            np.expand_dims(fields_scale, 1),
        ], axis=1))
