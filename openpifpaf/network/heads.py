"""Head networks."""

import argparse
import functools
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from openpifpaf import headmeta

LOG = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def index_field_torch(shape, *, device=None, unsqueeze=(0, 0)):
    assert len(shape) == 2
    xy = torch.empty((2, shape[0], shape[1]), device=device)
    xy[0] = torch.arange(shape[1], device=device)
    xy[1] = torch.arange(shape[0], device=device).unsqueeze(1)

    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)

    return xy





class PifHFlip(torch.nn.Module):
    def __init__(self, keypoints, hflip):
        super().__init__()

        flip_indices = torch.LongTensor([
            keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i
            for kp_i, kp_name in enumerate(keypoints)
        ])
        LOG.debug('hflip indices: %s', flip_indices)
        self.register_buffer('flip_indices', flip_indices)

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)

        # flip the x-coordinate of the vector component
        out[1][:, :, 0, :, :] *= -1.0

        return out


class PafHFlip(torch.nn.Module):
    def __init__(self, keypoints, skeleton, hflip):
        super().__init__()
        skeleton_names = [
            (keypoints[j1 - 1], keypoints[j2 - 1])
            for j1, j2 in skeleton
        ]
        flipped_skeleton_names = [
            (hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2)
            for j1, j2 in skeleton_names
        ]
        LOG.debug('skeleton = %s, flipped_skeleton = %s',
                  skeleton_names, flipped_skeleton_names)

        flip_indices = list(range(len(skeleton)))
        reverse_direction = []
        for paf_i, (n1, n2) in enumerate(skeleton_names):
            if (n1, n2) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n1, n2))
            if (n2, n1) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n2, n1))
                reverse_direction.append(paf_i)
        LOG.debug('hflip indices: %s, reverse: %s', flip_indices, reverse_direction)

        self.register_buffer('flip_indices', torch.LongTensor(flip_indices))
        self.register_buffer('reverse_direction', torch.LongTensor(reverse_direction))

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)

        # flip the x-coordinate of the vector components
        out[1][:, :, 0, :, :] *= -1.0
        out[2][:, :, 0, :, :] *= -1.0

        # reverse direction
        for paf_i in self.reverse_direction:
            cc = torch.clone(out[1][:, paf_i])
            out[1][:, paf_i] = out[2][:, paf_i]
            out[2][:, paf_i] = cc

        return out


class HeadNetwork(torch.nn.Module):
    """Base class for head networks.

    :param meta: head meta instance to configure this head network
    :param in_features: number of input features which should be equal to the
        base network's output features
    """
    def __init__(self, meta: headmeta.Base, in_features: int):
        super().__init__()
        self.meta = meta
        self.in_features = in_features

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, x):
        raise NotImplementedError


class CompositeField3(HeadNetwork):
    dropout_p = 0.0
    inplace_ops = True

    def __init__(self,
                 meta: headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
        out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        self.conv = torch.nn.Conv2d(in_features, out_features * (meta.upsample_stride ** 2),
                                    kernel_size, padding=padding, dilation=dilation)

        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CompositeField3')
        group.add_argument('--cf3-dropout', default=cls.dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--cf3-no-inplace-ops', dest='cf3_inplace_ops',
                           default=True, action='store_false',
                           help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.dropout_p = args.cf3_dropout
        cls.inplace_ops = args.cf3_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.meta.n_confidences + self.meta.n_vectors * 3 + self.meta.n_scales,
            feature_height,
            feature_width
        )



        if not self.training and self.inplace_ops:
            # classification
            classes_x = x[:, :, 0:self.meta.n_confidences]
            torch.sigmoid_(classes_x)

            # regressions x: add index
            if self.meta.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = self.meta.n_confidences
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)

            # remove width in the middle and add one to the front (v4 style)
            first_width_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            x = torch.cat([
                x[:, :, first_width_feature:first_width_feature + 1],
                x[:, :, :first_width_feature],
                x[:, :, self.meta.n_confidences + self.meta.n_vectors * 3:],
            ], dim=2)
        elif not self.training and not self.inplace_ops:
            # TODO: CoreMLv4 does not like strided slices.
            # Strides are avoided when switching the first and second dim
            # temporarily.
            x = torch.transpose(x, 1, 2)

            # classification
            classes_x = x[:, 0:self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)

            # regressions x
            first_reg_feature = self.meta.n_confidences
            regs_x = [
                x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)
            ]
            # regressions x: add index
            index_field = index_field_torch(
                (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            # TODO: coreml export does not work with the index_field creation in the graph.
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [reg_x + index_field if do_offset else reg_x
                      for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

            # regressions logb
            first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            single_reg_logb = x[:, first_reglogb_feature:first_reglogb_feature + 1]

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            # concat with width in front (v4 style)
            x = torch.cat([single_reg_logb, classes_x, *regs_x, scales_x], dim=1)

            # TODO: CoreMLv4 problem (see above).
            x = torch.transpose(x, 1, 2)

        return x


class CompositeField4(HeadNetwork):
    dropout_p = 0.0
    inplace_ops = True

    def __init__(self,
                 meta: headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)
        # 从plugin中对headmeta的设置得来：headmeta.Base: meta:Cif(name='cif'(caf), dataset='wholebody', ..., keypoints=[几个keypoints的名称],
        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,  # 直接在headmeta中定义好了，111或122
                  kernel_size, padding, dilation)

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
        self.n_components = 1 + meta.n_confidences + meta.n_vectors * 2 + meta.n_scales  # 需要预测的内容 b 是预测的内容？
        self.conv = torch.nn.Conv2d(
            in_features, meta.n_fields * self.n_components * (meta.upsample_stride ** 2),  # num keypoints * num prediction items for each keypoint
            kernel_size, padding=padding, dilation=dilation,
        )  # in_channels, out_channels, kernel_size, stride, padding, dilation
        # 1x1 convolution
        # meta.n_fields = len(self.keypoints)
        # output 'keypoints' coarse fileds (each location of the filed points to the location, and predicts the confidences and sclaes)
        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CompositeField4')
        group.add_argument('--cf4-dropout', default=cls.dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--cf4-no-inplace-ops', dest='cf4_inplace_ops',
                           default=True, action='store_false',
                           help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.dropout_p = args.cf4_dropout
        cls.inplace_ops = args.cf4_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ  # the inference flow (basenet -> heads) are defined in nets.py
        # torch.size(1, 1392, 46, 81)
        # feature_map = x.detach().cpu().numpy()[0]
        # feature_map = np.sum(feature_map, axis=0)
        # plt.imshow(feature_map)
        # plt.savefig('feature_map_volleyball2.jpg')

        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,  # keypoint或category的数量
            self.n_components,  # c x y b sigma (w h)
            feature_height,
            feature_width  # 这里预测时是每个grid都要输出预测的conf offset scale/wh
        )  # 类似yolo, 输出最后的卷积头预测结果
        # 如果是从checkpoint中直接加载网络，则此处的数字与dataset处设置的是cocokp还是cocodet无关，predict时本就用不着dataset
        #  1 91 6 55 81  detection checkpoint
        #  1 91 6 28 41  1 17 5 55 81 1 19 8 55 81 kp-det checkpoint
        # print('image shape', x.shape, x.size())
        # print('x_view', batch_size, self.meta.n_fields, self.n_components, feature_height, feature_width)
        # # plt.imshow(x.detach().cpu().numpy())
        # plt.figure()
        # for i in range(0, self.meta.n_fields):
        #     feature_map_combination = []
        #     feature_map = x.detach().cpu().numpy()[0, i, :, :, :]
        #     for j in range (0, self.n_components):
        #         feature_map_split = feature_map[j, :, :]
        #         feature_map_combination.append(feature_map_split)
        #         # plt.subplot(2, 3, j + 1)
        #         # plt.imshow(feature_map_split)
        #     feature_map_sum = sum(ele for ele in feature_map_combination)
        #     plt.imshow(feature_map_sum)
        #     if self.meta.n_fields == 91:
        #         plt.savefig(f'featuremap_kp-det-train-det/feature_map_cls{i}.pdf')
        #     elif self.meta.n_fields == 17:
        #         plt.savefig(f'featuremap_kp-det-train-cif/feature_map_cls{i}.pdf')
        #     elif self.meta.n_fields == 19:
        #         plt.savefig(f'featuremap_kp-det-train-caf/feature_map_cls{i}.pdf')

        if not self.training and self.inplace_ops:
            # classification
            classes_x = x[:, :, 1:1 + self.meta.n_confidences]
            torch.sigmoid_(classes_x)  # 类别+conf 类别指的是前景和背景

            # regressions x: add index
            if self.meta.n_vectors > 0:  # offset
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = 1 + self.meta.n_confidences  # conf结束idx
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2  # offset结束idx
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)  # det和cif的区别是det没有scale
        elif not self.training and not self.inplace_ops:
            # TODO: CoreMLv4 does not like strided slices.
            # Strides are avoided when switching the first and second dim
            # temporarily.
            x = torch.transpose(x, 1, 2)

            # width
            width_x = x[:, 0:1]

            # classification
            classes_x = x[:, 1:1 + self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)  # 类别+conf

            # regressions x
            first_reg_feature = 1 + self.meta.n_confidences  # conf结束idx
            regs_x = [
                x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)  # offset
            ]
            # regressions x: add index
            index_field = index_field_torch(
                (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            # TODO: coreml export does not work with the index_field creation in the graph.
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [reg_x + index_field if do_offset else reg_x
                      for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2  # offset结束idx
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            # concat
            x = torch.cat([width_x, classes_x, *regs_x, scales_x], dim=1)  # 非training, 输出x

            # TODO: CoreMLv4 problem (see above).
            x = torch.transpose(x, 1, 2)

        return x
