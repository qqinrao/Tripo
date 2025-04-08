"""
该Python文件定义了一个数据类 `Transformer1DModelOutput`。
此数据类属于 `triposg.models.transformers` 模块,用于表示1D 变换器模型的输出，其中包含一个类型为 `torch.FloatTensor` 的属性 `sample` 。
依赖 `torch` 库和Python内置的 `dataclasses` 模块。 
"""
"""
modeling_outputs.py 文件的作用是定义 TripoSG 系统中 Transformer 模型的输出数据结构。
这是一个非常简洁但很重要的组件，为模型输出提供了标准化的接口和类型安全保障。
"""
from dataclasses import dataclass

import torch


@dataclass
class Transformer1DModelOutput:
    sample: torch.FloatTensor
