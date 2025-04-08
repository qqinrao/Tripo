"""
这是一个Python文件,定义在TripoSG项目的变分自动编码器(VAE)模块中。
文件定义了 `DiagonalGaussianDistribution` 类,用于处理高斯分布。该类能计算均值、标准差、方差等统计量,并实现从分布中采样、计算Kullback-Leibler散度(KL)、负对数似然(NLL)以及获取分布的众数等功能。
代码使用了PyTorch进行张量计算,并借助 `diffusers.utils.torch_utils` 中的 `randn_tensor` 函数生成随机张量。
"""
from typing import Optional, Tuple

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters: torch.Tensor,
        deterministic: bool = False,
        feature_dim: int = 1,
    ):
        self.parameters = parameters
        self.feature_dim = feature_dim
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=feature_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(
        self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
