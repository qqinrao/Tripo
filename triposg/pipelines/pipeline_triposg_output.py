"""
TripoSG 系统中定义输出数据结构的模块。
它提供了一个专门的数据类来封装和标准化 TripoSG 生成管道的输出，使输出结果结构化且易于使用。
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from diffusers.utils import BaseOutput


@dataclass
class TripoSGPipelineOutput(BaseOutput):
    r"""
    Output class for ShapeDiff pipelines.
    """

    samples: torch.Tensor
    meshes: List[trimesh.Trimesh]
