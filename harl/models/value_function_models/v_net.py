import torch
import torch.nn as nn
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class VNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"] # MLP隐藏层神经元数量
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]  # RNN的层数
        self.tpdv = dict(dtype=torch.float32, device=device) # dtype和device

        # 获取网络权重初始化方法函数
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space) # 获取观测空间的形状，tuple of integer. eg: （54，）

        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小cent_obs_shape，输出大小hidden_sizes[-1]
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        # 如果使用RNN，初始化RNN层 #TODO: 暂时还没看
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # 定义了一个初始化神经网络权重的函数 （特别是 nn.Linear 层的权重）
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 初始化了一个 nn.Linear 层 (self.linear)，该层将输入的self.hidden_sizes[-1]个特征映射到一个值，即state value
        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        critic_features = self.base(cent_obs)

        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        # 将特征输入v_out层，得到值函数预测值
        values = self.v_out(critic_features)

        return values, rnn_states
