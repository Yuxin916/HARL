"""Base class for on-policy algorithms."""

import torch
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.models_tools import update_linear_schedule


class OnPolicyBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Base class.
        Args:
            args: (dict) arguments.  # yaml里面的model和algo的config打包作为args进入OnPolicyBase
            obs_space: (gym.spaces or list) observation space. # 单个智能体的观测空间 eg: Box (18,)
            act_space: (gym.spaces) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        # "model" and "algo" sections in $Algorithm config file
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)   # dtype和device

        self.data_chunk_length = args["data_chunk_length"]  # TODO：这是什么 rnn相关
        self.use_recurrent_policy = args["use_recurrent_policy"]  # TODO：这两个的区别，基于rnn chunck是什么
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]  # TODO：这是什么
        self.action_aggregation = args["action_aggregation"]  # TODO：这是什么

        self.lr = args["lr"]  # actor学习率
        self.opti_eps = args["opti_eps"]  # optimizer的epsilon
        self.weight_decay = args["weight_decay"] # optimizer的权重衰减

        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space

        # 建立actor网络结构
        self.actor = StochasticPolicy(args, self.obs_space, self.act_space, self.device)

        # 建立actor的优化器optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        episode是当前episode的index，episodes是总共需要跑多少个episode
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def get_actions(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor. 当前时刻obs
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
                                上一时刻的rnn_state
            masks: (np.ndarray) denotes points at which RNN states should be reset. #TODO: 这是什么，有点像episode结束的信号
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 当前智能体的可用动作 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
                                有没有available_actions，离散还是连续动作
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states_actor

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray / torch.Tensor) if actor has RNN layer, RNN states for actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            masks: (np.ndarray / torch.Tensor) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        """

        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        pass

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        """
        pass

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        # 测试actor网络结构
        self.actor.eval()
