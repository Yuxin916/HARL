"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method


class FixedCategorical(torch.distributions.Categorical):
    """
    扩展了标准的PyTorch Categorical分布的功能
    Modify standard PyTorch Categorical.
    包括了对从分布中抽样、计算log_probs和找到probs.argmax的修改。

    """

    def sample(self):
        """
        离散动作
        从分类分布中进行抽样
        调用了超类（Categorical）的sample()方法，然后在最后一个维度（-1）上进行unsqueeze操作
        """
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        """
        接受一个输入张量actions，根据分布计算这些动作的对数概率
        """
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        """
        连续动作
        通过查找分布的probs张量中的最大概率所对应的index，然后将结果保持为带有额外维度的tensor
        """
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        # 获取网络权重初始化方法函数
        init_method = get_init_method(initialization_method)

        # 定义了一个初始化神经网络权重的函数 （特别是 nn.Linear 层的权重）
        def init_(m):
            """
            m: nn.Linear 层 - 想要初始化神经网络权重的层
            init_method: 权重初始化方法
            lambda x: nn.init.constant_(x, 0): 用于初始化偏置的函数
            gain: 权重初始化的增益
            """
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        # 初始化了一个 nn.Linear 层 (self.linear)，该层将 {并行环境数量 x hidden_sizes[-1]} 个输入连接到 action_dim 个输出。
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10

        # 将线性层的输出转换为一个 FixedCategorical 分布
        a = FixedCategorical(logits=x)
        return a


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
