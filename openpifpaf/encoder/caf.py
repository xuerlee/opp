import dataclasses
import logging
from typing import ClassVar, List, Tuple

import numpy as np
import torch

from .annrescaler import AnnRescaler

from openpifpaf import headmeta
from openpifpaf. visualizer import Caf as CafVisualizer
from openpifpaf. utils import mask_valid_area

LOG = logging.getLogger(__name__)

'''
生成paf的步骤和之前的生成pif步骤基本一致, 也是需要执行rescalar操作. 主要还是函数fill_associttion那里对每个连接线段生成信息. 每条连接线段的scale值和pif一致
'''
@dataclasses.dataclass
class Caf:
    meta: headmeta.Caf
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CafVisualizer = None
    fill_plan: List[Tuple[int, int, int]] = None

    min_size: ClassVar[int] = 3
    fixed_size: ClassVar[bool] = False
    aspect_ratio: ClassVar[float] = 0.0
    padding: ClassVar[int] = 10

    def __post_init__(self):
        if self.rescaler is None:
            self.rescaler = AnnRescaler(self.meta.stride, self.meta.pose)

        if self.visualizer is None:
            self.visualizer = CafVisualizer(self.meta)

        if self.fill_plan is None:
            self.fill_plan = [
                (caf_i, joint1i - 1, joint2i - 1)
                for caf_i, (joint1i, joint2i) in enumerate(self.meta.skeleton)  # 连接的两个关节
            ]

    def __call__(self, image, anns, meta):
        return CafGenerator(self)(image, anns, meta)


class AssociationFiller:
    def __init__(self, config: Caf):
        self.config = config
        self.rescaler = config.rescaler
        self.visualizer = config.visualizer

        self.sparse_skeleton_m1 = (
            np.asarray(config.meta.sparse_skeleton) - 1  # 也在constant中定义了
            if getattr(config.meta, 'sparse_skeleton', None) is not None
            else None
        )

        if self.config.fixed_size:
            assert self.config.aspect_ratio == 0.0

        LOG.debug('only_in_field_of_view = %s, caf min size = %d',
                  config.meta.only_in_field_of_view,
                  self.config.min_size)

        self.field_shape = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        raise NotImplementedError

    def all_fill_values(self, keypoint_sets, anns):
        """Values in the same order and length as keypoint_sets."""
        raise NotImplementedError

    def fill_field_values(self, field_i, fij, fill_values):
        raise NotImplementedError

    def fields_as_tensor(self, valid_area):
        raise NotImplementedError

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.min_size - 1) / 2)
        self.field_shape = (
            self.config.meta.n_fields,
            bg_mask.shape[0] + 2 * self.config.padding,
            bg_mask.shape[1] + 2 * self.config.padding,
        )
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s', valid_area)

        self.init_fields(bg_mask)
        self.fields_reg_l = np.full(self.field_shape, np.inf, dtype=np.float32)
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0

        fill_values = self.all_fill_values(keypoint_sets, anns)
        self.fill(keypoint_sets, fill_values)
        fields = self.fields_as_tensor(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def fill(self, keypoint_sets, fill_values):
        for keypoints, fill_value in zip(keypoint_sets, fill_values):
            self.fill_keypoints(keypoints, fill_value)

    def shortest_sparse(self, joint_i, keypoints):
        shortest = np.inf
        for joint1i, joint2i in self.sparse_skeleton_m1:
            if joint_i not in (joint1i, joint2i):
                continue

            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            d = np.linalg.norm(joint1[:2] - joint2[:2])
            shortest = min(d, shortest)

        return shortest

    def fill_keypoints(self, keypoints, fill_values):
        for field_i, joint1i, joint2i in self.config.fill_plan:
            joint1 = keypoints[joint1i]  # slekton是按keypoint序号定义的
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            # check if there are shorter connections in the sparse skeleton
            if self.sparse_skeleton_m1 is not None:
                d = (np.linalg.norm(joint1[:2] - joint2[:2])
                     / self.config.meta.dense_to_sparse_radius)  # dense中点的距离
                if self.shortest_sparse(joint1i, keypoints) < d \
                   and self.shortest_sparse(joint2i, keypoints) < d:  # sparse 中点的最小距离
                    continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
            out_field_of_view_1 = (
                joint1[0] < 0
                or joint1[1] < 0
                or joint1[0] > self.field_shape[2] - 1 - 2 * self.config.padding
                or joint1[1] > self.field_shape[1] - 1 - 2 * self.config.padding
            )
            out_field_of_view_2 = (
                joint2[0] < 0
                or joint2[1] < 0
                or joint2[0] > self.field_shape[2] - 1 - 2 * self.config.padding
                or joint2[1] > self.field_shape[1] - 1 - 2 * self.config.padding
            )
            if out_field_of_view_1 and out_field_of_view_2:
                continue
            if self.config.meta.only_in_field_of_view:
                if out_field_of_view_1 or out_field_of_view_2:
                    continue

            self.fill_association(field_i, joint1, joint2, fill_values)

    '''
    **fill_associttion**函数: 
    首先计算joint1和joint2之间的偏移向量offset, 并根据offset的模长动态的生成一个矩阵sink. 
    这个sink和pif那里使用的函数一样, 都是create_sink函数. 这个函数会返回一个(2, s, s)的矩阵, s是传入的参数值, 里面的内容就是s * s大小的, 分别是x方向和y方向的偏移值. 
    (源码里约定s的值最小为3, 最大会根据offset的模厂动态调整).
     然后, 根据计算得到的s_offset长度, 分别用joint1和joint2的坐标值减去s_offset, 得到的其实就是在去除s_offset的偏移喜爱joint1及joint2两个点的新值. 
     得到joint1ij 和 joint2ij(其实发现源码里的offset和offsetij基本上是同一个值, 只不过offsetij可能更准确些)
     
    因为不能只选用一个点作为可以指向joint1和joint2的中间节点, 因此动态的设置中间点的个数, 为num = max(2, int(np.ceil(offset_d))). 
    这样其实是在joint1和joint2之间取了num个点来表示joint1和joint2之间的连接. 
    之后的内容就和pif很相像, 确定label的表示范围, 然后打上对应的标签.

    因为选取的每一个点, 都需要计算下当前位置里joint1和joint2的偏移. 
    文章的做法是首先根据前面计算得到的去除s_offset的joint1ij的值, 加上f*offsetij得到一个坐标(不考虑padding), 然后再加上s_offset得到真正的位置点坐标fxy. 
    接着计算fxy和joint1, joint2的偏移量, 再加上前面得到的偏移矩阵sink, 就是得到以fxy为中心的, [fminx:fmaxx, fminy:fmaxy]区域范围内的每个位置离真正的joint1 joint2的偏移值. (
    后面的步骤就是重复pif 打label的过程, 掠过不表)
    '''

    def fill_association(self, field_i, joint1, joint2, fill_values):
        # offset between joints
        offset = joint2[:2] - joint1[:2]  # 两点之间的偏移向量
        offset_d = np.linalg.norm(offset)  # 求范数
        # 离得最近的负责预测的grid之类，直接算两个点之间的格子都为预测范围了
        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))

        # meshgrid: coordinate matrix
        xyv = np.stack(np.meshgrid(
            np.linspace(-0.5 * (s - 1), 0.5 * (s - 1), s),
            np.linspace(-0.5 * (s - 1), 0.5 * (s - 1), s),
        ), axis=-1).reshape(-1, 2)  # 创建等差数列
        # np.meshgrid代表的是将x中每一个数据和y中每一个数据组合生成很多点,然后将这些点的x坐标放入到X中,y坐标放入Y中,并且相应位置是对应的

        # set fields
        num = max(2, int(np.ceil(offset_d)))  # 中间点的最大个数
        fmargin = (s / 2) / (offset_d + np.spacing(1))
        fmargin = np.clip(fmargin, 0.25, 0.4)
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0 - fmargin, num=num)  # 分布在0和1之间
        if self.config.fixed_size:
            frange = [0.5]
        filled_ij = set()
        for f in frange:  # f: id, 通过f可以遍历到每一个中间点？
            for xyo in xyv:  # 每一个xyv grid?
                fij = np.round(joint1[:2] + f * offset + xyo).astype(int) + self.config.padding  # 通过joint1的位置计算中间点的位置并取整
                if fij[0] < 0 or fij[0] >= self.field_shape[2] or \
                   fij[1] < 0 or fij[1] >= self.field_shape[1]:
                    continue

                # convert to hashable coordinate and check whether
                # it was processed before
                fij_int = (int(fij[0]), int(fij[1]))
                if fij_int in filled_ij:
                    continue
                filled_ij.add(fij_int)

                # mask
                # perpendicular distance computation:
                # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                # Coordinate systems for this computation is such that
                # joint1 is at (0, 0).
                fxy = fij - self.config.padding  # fxy是减去padding的某中间点
                f_offset = fxy - joint1[:2]  # 中间点到joint1的偏置
                sink_l = np.fabs(
                    offset[1] * f_offset[0]
                    - offset[0] * f_offset[1]
                ) / (offset_d + 0.01)  # 逐元素计算绝对值
                if sink_l > self.fields_reg_l[field_i, fij[1], fij[0]]:
                    continue
                self.fields_reg_l[field_i, fij[1], fij[0]] = sink_l  # L1 norm?

                self.fill_field_values(field_i, fij, fill_values)  # 针对某一中间点fij，为其intensity，和两个joint之间的偏置打标签


class CafGenerator(AssociationFiller):
    def __init__(self, config: Caf):
        super().__init__(config)

        self.skeleton_m1 = np.asarray(config.meta.skeleton) - 1

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_bmin1 = None
        self.fields_bmin2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None

    def init_fields(self, bg_mask):
        reg_field_shape = (self.field_shape[0], 2, self.field_shape[1], self.field_shape[2])

        self.intensities = np.zeros(self.field_shape, dtype=np.float32)  # double cif
        self.fields_reg1 = np.full(reg_field_shape, np.nan, dtype=np.float32)
        self.fields_reg2 = np.full(reg_field_shape, np.nan, dtype=np.float32)
        self.fields_bmin1 = np.full(self.field_shape, np.nan, dtype=np.float32)
        self.fields_bmin2 = np.full(self.field_shape, np.nan, dtype=np.float32)
        self.fields_scale1 = np.full(self.field_shape, np.nan, dtype=np.float32)
        self.fields_scale2 = np.full(self.field_shape, np.nan, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def all_fill_values(self, keypoint_sets, anns):
        return [(kps, self.rescaler.scale(kps)) for kps in keypoint_sets]

    def fill_field_values(self, field_i, fij, fill_values):
        joint1i, joint2i = self.skeleton_m1[field_i]
        keypoints, scale = fill_values

        # update intensity
        self.intensities[field_i, fij[1], fij[0]] = 1.0  # fij：joint1和joint2之间的某一中间点

        # update regressions
        fxy = fij - self.config.padding
        self.fields_reg1[field_i, :, fij[1], fij[0]] = keypoints[joint1i][:2] - fxy  # 该中间点到joint1和joiint2之间的偏置
        self.fields_reg2[field_i, :, fij[1], fij[0]] = keypoints[joint2i][:2] - fxy

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin1[field_i, fij[1], fij[0]] = bmin
        self.fields_bmin2[field_i, fij[1], fij[0]] = bmin

        # update scale
        if self.config.meta.sigmas is None:
            scale1, scale2 = scale, scale
        else:
            scale1 = scale * self.config.meta.sigmas[joint1i]
            scale2 = scale * self.config.meta.sigmas[joint2i]
        assert np.isnan(scale1) or 0.0 < scale1 < 100.0
        self.fields_scale1[field_i, fij[1], fij[0]] = scale1
        assert np.isnan(scale2) or 0.0 < scale2 < 100.0
        self.fields_scale2[field_i, fij[1], fij[0]] = scale2

    def fields_as_tensor(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_bmin1 = self.fields_bmin1[:, p:-p, p:-p]
        fields_bmin2 = self.fields_bmin2[:, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin2, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg1,
            fields_reg2,
            np.expand_dims(fields_bmin1, 1),
            np.expand_dims(fields_bmin2, 1),
            np.expand_dims(fields_scale1, 1),
            np.expand_dims(fields_scale2, 1),
        ], axis=1))
