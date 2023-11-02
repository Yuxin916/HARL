import torch.nn as nn
from harl.utils.models_tools import init, get_active_func, get_init_method

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小obs_shape，输出大小hidden_sizes[-1]

        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func) # 激活函数
        init_method = get_init_method(initialization_method) # 初始化方法

        # 调整权重的尺度，以确保层中的激活在训练期间不会变得太小（梯度消失）或太大（梯度爆炸）
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 创建第一个隐藏层(include len(obs))
        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        # 循环，用于创建多个隐藏层
        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]  # 是否使用特征归一化 TODO: 什么是特征归一化，batchnorm和layernorm的区别
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.activation_func = args["activation_func"]  # 激活函数
        self.hidden_sizes = args["hidden_sizes"]  # 隐藏层大小

        obs_dim = obs_shape[0]  # 获取观测空间的形状，integer. eg: 18

        # 如果使用特征归一化，则使用LayerNorm
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        # 创建MLP层，输入obs_dim，隐藏层大小，网络权重初始化方法，激活函数
        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )
        pass

    def forward(self, x):
        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小obs_shape，输出大小hidden_sizes[-1]
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
