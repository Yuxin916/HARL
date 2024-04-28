import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical, Normal

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, masked=False):
        super(SelfAttention, self).__init__()
        # d_q = d_k = d_v = d_model/h
        # d_model = n_embd --> 确保d_model可以被n_head整除
        assert n_embd % n_head == 0
        # 使用mask, 使得attention只能看到前面的信息
        self.masked = masked
        # 多头
        self.n_head = n_head
        # key, query, value 线性projection for all heads (进入attention前的线性变换)
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # 从多头attention输出再concat后的线性变换
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.n_embd = n_embd
        self.dropout = nn.Dropout(0.1)
        self.att_bp = None

    def forward(self, key, value, query):
        """
        输入: key, value, query
        计算: scaled dot-product self-attention
        输出: self-attention后的value
        """
        # query size (n_rollout_threads, n_embd)
        # B: batch size - n_rollout_threads
        # D: hidden dimension - n_embd
        B, D = query.size()

        # Reshape key, query, value to be (B, n_head, D // n_head) for multi-head attention
        # Since there's no sequence length L, we treat the whole D as the sequence of features for each head
        k = self.key(key).view(B, self.n_head, -1).transpose(0, 1)  # (n_head, B, hs)
        q = self.query(query).view(B, self.n_head, -1).transpose(0, 1)  # (n_head, B, hs)
        v = self.value(value).view(B, self.n_head, -1).transpose(0, 1)  # (n_head, B, hs)

        # Attention（Q,K,V）= softmax(QK^T/sqrt(d_k))V
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(D // self.n_head))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            mask = torch.triu(torch.ones(B, B, device=query.device), diagonal=1).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, nh, L, hs)

        # 把multi-head的结果拼接起来 --> (B, L, D)
        attn_output = attn_output.transpose(0, 1).contiguous().view(B, D)  # (B, D)

        # output projection --> (B, D)
        y = self.proj(attn_output)

        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # mask关掉了，说明所有的agent都可以看到所有的agent的obs
        self.attn = SelfAttention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        # x [n_rollout_thread, n_embd]
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x



class Encoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd,
                 n_head, action_type='Discrete'):
        super(Encoder, self).__init__()


        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd

        self.action_type = action_type

        # obs_encoder都是单层的MLP
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)

        # n_block代表EncodeBlock的数量
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head) for _ in range(n_block)])

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, obs):
        # obs: (n_rollout_thread, obs_dim)

        # 所有agent共用同一个obs_encoder，这是第一个embedding
        # 分别提取每个agent的obs feature
        obs_embeddings = self.obs_encoder(obs)
        # obs_embeddings: (n_rollout_thread, n_embd)
        x = obs_embeddings

        # 在做完layer norm之后，进入multi-head attention, 每一个都是EncodeBlock
        # rep: (n_rollout_thread, n_embd)
        rep = self.blocks(self.ln(x))


        return rep