"""ValueNorm."""
import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):  # TODO
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(
        self,
        input_shape,
        norm_axes=1,
        beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"),
    ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )
        pass

    def running_mean_var(self):
        """Get running mean and variance.
        计算数据的滚动均值和滚动方差"""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        """将输入的向量进行归一化处理"""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        # 计算出运行时的均值和方差
        mean, var = self.running_mean_var()
        # 将输入向量减去均值，然后除以方差的平方根
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]
        # 将使输入向量的数据分布归一化为均值为0，方差为1的标准正态分布
        return out

    def denormalize(self, input_vector):
        """Transform normalized data back into original distribution
        用于将归一化后的数据还原为原始数据分布"""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        # 计算出运行时的均值和方差
        mean, var = self.running_mean_var()
        # 将输入向量乘以方差的平方根，然后加上均值
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        # 将归一化后的数据还原为原始数据分布
        return out
