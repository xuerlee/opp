
from collections import defaultdict


def skeleton_mapping(kps_mapping):
    """Map the subset of keypoints from 0 to n-1"""   # 对关键点重新排序？
    map_sk = defaultdict(lambda: 100)  # map to 100 the keypoints not used
    for i, j in zip(kps_mapping, range(len(kps_mapping))):
    #  [5, 2, 4, 3, 1] -> [5, 2, 3, 4, 1]   # kps_mapping指的是每个关节点的id在原列表的位次
        map_sk[i] = j
    return map_sk


def transform_skeleton(skeleton_orig, kps_mapping):
    """
    Transform the original apollo skeleton of 66 joints into a skeleton from 1 to n
    """
    map_sk = skeleton_mapping(kps_mapping)
    # skeleton = [[dic_sk[i], dic_sk[j]] for i, j in SKELETON]  # TODO
    skeleton = []
    for i, j in skeleton_orig:
        skeleton.append([map_sk[i] + 1, map_sk[j] + 1])   # skeleton starts from 1
    return skeleton  # 返回每个关节点的id在原列表的位次
