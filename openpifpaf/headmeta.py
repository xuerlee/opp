"""Head meta objects contain meta information about head networks.

This includes the name, the name of the individual fields, the composition, etc.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Tuple

import numpy as np

# dataclass 相当于类中的 def __init__(self, xxx, xxx)
@dataclass
class Base:  # 也在plugin中定义，两个双引号内的名称就是name 和 dataset
    name: str
    dataset: str

    head_index: int = field(default=None, init=False)  # filed:使用复合形式初始化字段 If init is True, the field will be a parameter to the class's __init__() function.
    base_stride: int = field(default=None, init=False)
    upsample_stride: int = field(default=1, init=False)

    @property
    def stride(self) -> int:
        if self.base_stride is None:
            return None
        return self.base_stride // self.upsample_stride

    @property
    def n_fields(self) -> int:
        raise NotImplementedError


@dataclass
class Cif(Base):  # 检测关节 定义卷积层信息和label处理信息
    """
    Headmeta is a class that holds configuration data about a head network.
    It is instantiated in a DataModule (above) and used throughout OpenPifPaf to configure various other parts.
    """
    """Head meta data for a Composite Intensity Field (CIF)."""

    keypoints: List[str]
    sigmas: List[float]
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    score_weights: List[float] = None
    # c; x; y; b; σ
    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 1
    n_scales: ClassVar[int] = 1

    vector_offsets = [True]
    decoder_min_scale = 0.0
    decoder_seed_mask: List[int] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.keypoints)


@dataclass
class Caf(Base):  # 检测连接
    """Head meta data for a Composite Association Field (CAF)."""

    keypoints: List[str]
    sigmas: List[float]
    skeleton: List[Tuple[int, int]]
    pose: Any = None
    sparse_skeleton: List[Tuple[int, int]] = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False
    # c; x1; y1; x2; y2; σ1; σ2
    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.skeleton)

    @staticmethod
    def concatenate(metas):
        # TODO: by keypoint name, update skeleton indices if meta.keypoints
        # is not the same for all metas.
        """
        When a new network is created, information from the head metas will be used to
        create the appropriate torch graph for the heads.
        It will use the type of the head meta (openpifpaf.headmeta.Cif, openpifpaf.headmeta.Caf, …)
        and information like the number of keypoints in Cif or the number of skeleton connections in Caf to know how many feature maps to create.
        """
        concatenated = Caf(
            name='_'.join(m.name for m in metas),
            dataset=metas[0].dataset,
            keypoints=metas[0].keypoints,
            sigmas=metas[0].sigmas,
            pose=metas[0].pose,
            skeleton=[s for meta in metas for s in meta.skeleton],
            sparse_skeleton=metas[0].sparse_skeleton,
            only_in_field_of_view=metas[0].only_in_field_of_view,
            decoder_confidence_scales=[
                s
                for meta in metas
                for s in (meta.decoder_confidence_scales
                          if meta.decoder_confidence_scales
                          else [1.0 for _ in meta.skeleton])
            ]
        )
        concatenated.head_index = metas[0].head_index
        concatenated.base_stride = metas[0].base_stride
        concatenated.upsample_stride = metas[0].upsample_stride
        return concatenated


@dataclass
class CifDet(Base):
    """Head meta data for a Composite Intensity Field (CIF) for Detection."""

    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2  # localization、 wh
    n_scales: ClassVar[int] = 0

    vector_offsets = [True, False]
    decoder_min_scale = 0.0

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.categories)


@dataclass
class TSingleImageCif(Cif):
    """Single-Image CIF head in tracking models."""


@dataclass
class TSingleImageCaf(Caf):
    """Single-Image CAF head in tracking models."""


@dataclass
class Tcaf(Base):  # 跟踪
    """Tracking Composite Association Field."""

    keypoints_single_frame: List[str]
    sigmas_single_frame: List[float]
    pose_single_frame: Any
    draw_skeleton_single_frame: List[Tuple[int, int]] = None
    keypoints: List[str] = None
    sigmas: List[float] = None
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    training_weights: List[float] = None

    vector_offsets = [True, True]

    def __post_init__(self):
        if self.keypoints is None:
            self.keypoints = np.concatenate((
                self.keypoints_single_frame,
                self.keypoints_single_frame,
            ), axis=0)
        if self.sigmas is None:
            self.sigmas = np.concatenate((
                self.sigmas_single_frame,
                self.sigmas_single_frame,
            ), axis=0)
        if self.pose is None:
            self.pose = np.concatenate((
                self.pose_single_frame,
                self.pose_single_frame,
            ), axis=0)
        if self.draw_skeleton is None:
            self.draw_skeleton = np.concatenate((
                self.draw_skeleton_single_frame,
                self.draw_skeleton_single_frame,
            ), axis=0)

    @property
    def skeleton(self):
        return [(i + 1, i + 1 + len(self.keypoints_single_frame))
                for i, _ in enumerate(self.keypoints_single_frame)]

    @property
    def n_fields(self):
        return len(self.keypoints_single_frame)
