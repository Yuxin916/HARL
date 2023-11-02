"""On-policy buffer for critic that uses Environment-Provided (EP) state."""
import torch
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.utils.trans_tools import _flatten, _sa_cast


class OnPolicyCriticBufferEP:
    """On-policy buffer for critic that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space):
        """Initialize on-policy critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
        """
        self.episode_length = args["episode_length"]  # 每个环境的episode长度 TODO：检查env config
        self.n_rollout_threads = args["n_rollout_threads"] # 多进程环境数量
        self.hidden_sizes = args["hidden_sizes"] # critic网络的隐藏层大小
        self.rnn_hidden_size = self.hidden_sizes[-1] # rnn隐藏层大小
        self.recurrent_n = args["recurrent_n"] # rnn的层数

        self.gamma = args["gamma"]  # 折扣因子
        self.gae_lambda = args["gae_lambda"]  # GAE的参数
        self.use_gae = args["use_gae"]  # 是否使用GAE

        self.use_proper_time_limits = args["use_proper_time_limits"]  # 是否考虑episode的提前结束

        share_obs_shape = get_shape_from_obs_space(share_obs_space)  # 获取单个智能体状态空间的形状，tuple of integer. eg: （54，）

        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]

        """
        Critic Buffer里储存了： ALL (np.ndarray) NOTE： 所有agent的全局状态是一样的
        1. self.share_obs: 全局状态 [episode_length + 1, 进程数量, share_obs_shape]
        2. self.rnn_states_critic: critic的rnn状态 [episode_length + 1, 进程数量, recurrent_n, rnn_hidden_size]
        3. self.value_preds: critic的value预测 [episode_length + 1, 进程数量, 1]
        4. self.returns: 每一步的计算的return [episode_length + 1, 进程数量, 1]
        5. self.rewards: 每一步的reward [episode_length, 进程数量, 1]
        6. self.masks: 每一步的mask,环境是否done [episode_length + 1, 进程数量, 1]
        7. self.bad_masks: 每一步的bad_mask,是否提前结束truncation [episode_length + 1, 进程数量, 1] 和6一起看
        """

        # Buffer for share observations

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of critic
        self.rnn_states_critic = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for value predictions made by this critic
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for returns calculated at each timestep
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for masks indicating whether an episode is done at each timestep
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for bad masks indicating truncation and termination.
        # If 0, trunction; if 1 and masks is 0, termination; else, not done yet.
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self, share_obs, rnn_states_critic, value_preds, rewards, masks, bad_masks
    ):
        """Insert data into buffer."""
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        """Get mean rewards for logging."""
        return np.mean(self.rewards)

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        # next_value --- np.array shape=(环境数, 1) -- 最后一个状态的状态值
        # self.value_normalizer --- ValueNorm
        """
        # consider the difference between truncation and termination
        if self.use_proper_time_limits:
            if self.use_gae:  # use GAE
                # 把最后一个状态的状态值放到value_preds的最后一个位置, index是200
                self.value_preds[-1] = next_value
                gae = 0  # 可以看成gae(200) = 0
                # timestep从后往前--倒推的方式
                for step in reversed(range(self.rewards.shape[0])):  # 从step199到0
                    # use ValueNorm
                    # 在GAE计算中，将值函数的估计值denormalize，然后再计算GAE，最后再normalize
                    if value_normalizer is not None:
                        # 计算delta[step]
                        # delta[step] = r([step]) + gamma * V(s[step+1]) * mask - V(s[step]) -- 如果下一个step done
                        # delta[step] = r([step]) + gamma * 0 * mask - V(s[step]) -- 如果下一个step done
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )

                        # gae递归公式，查看https://zhuanlan.zhihu.com/p/651944382和笔记
                        # gae[step] = delta[step] + gamma * lambda * mask[t+1] * gae[t+1]
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )

                        # 因为分离了terminated和truncated
                        gae = self.bad_masks[step + 1] * gae

                        # V网络的标签值 = GAE(step) + V网络(step) -- 标量
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])

                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )

                        # 因为分离了terminated和truncated
                        gae = self.bad_masks[step + 1] * gae

                        # V网络的标签值 = GAE + V网络的估计值 -- 标量
                        self.returns[step] = gae + self.value_preds[step]

            else:  # do not use GAE
                # 把最后一个状态的状态值放到value_preds的最后一个位置, index是200
                self.returns[-1] = next_value

                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        # Q递归公式 - V网络的标签值
                        # Q[step] = r([step]) + gamma * V(s[step+1]) * mask
                        discounted_cum_reward = (self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step])
                        #   TODO 没看完
                        discounted_value = (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                        self.returns[step] = discounted_cum_reward * self.bad_masks[step + 1] + discounted_value

                    else:  # do not use ValueNorm
                        self.returns[step] = (self.returns[step + 1] *
                                              self.gamma *
                                              self.masks[step + 1] +
                                              self.rewards[step]) \
                                             * \
                                              self.bad_masks[step + 1] + \
                                             (1 - self.bad_masks[step + 1]) * \
                                             self.value_preds[step]
        # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
        else:
            if self.use_gae:  # use GAE
                # 把最后一个状态的状态值放到value_preds的最后一个位置, index是200
                self.value_preds[-1] = next_value
                gae = 0  # 可以看成gae(200) = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        # 计算delta(step) = r(step) + gamma * V(step+1) * mask(step+1) - V(step)
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        # 计算GAE(step) = delta(step) + gamma * lambda * mask(step+1) * gae(step+1)
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )

                        # V网络的标签值 = GAE(step) + V网络(step) -- 标量
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )

                        # V网络的标签值 = GAE + V网络的估计值 -- 标量
                        self.returns[step] = gae + self.value_preds[step]

            else:  # do not use GAE
                # 把最后一个状态的状态值放到value_preds的最后一个位置, index是200
                self.returns[-1] = next_value

                for step in reversed(range(self.rewards.shape[0])):
                    # V网络的标签值 = r + gamma * V(s[t+1]) * mask
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1]
                                          + self.rewards[step])

    def feed_forward_generator_critic(
        self, critic_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for critic that uses MLP network.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            mini_batch_size: (int) Size of mini batch for critic.
        """

        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) = {n_rollout_threads * episode_length} "
                f"is required to be greater than or equal to the number of critic mini batches ({critic_num_mini_batch})."
            )
            mini_batch_size = batch_size // critic_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (episode_length * n_rollout_threads, *share_obs_shape)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )  # actually not used, just for consistency
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            # share_obs shape:
            # (episode_length * n_rollout_threads, *share_obs_shape) --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def naive_recurrent_generator_critic(self, critic_num_mini_batch):
        """Training data generator for critic that uses RNN network.
        This generator does not split the trajectories into chunks,
        and therefore maybe less efficient than the recurrent_generator_critic in training.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
        """

        # get n_rollout_threads and num_envs_per_batch
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= critic_num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({critic_num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // critic_num_mini_batch

        # shuffle indices
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(critic_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            share_obs_batch = _flatten(T, N, self.share_obs[:-1, ids])
            value_preds_batch = _flatten(T, N, self.value_preds[:-1, ids])
            return_batch = _flatten(T, N, self.returns[:-1, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            rnn_states_critic_batch = self.rnn_states_critic[0, ids]

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def recurrent_generator_critic(self, critic_num_mini_batch, data_chunk_length):
        """Training data generator for critic that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            data_chunk_length: (int) Length of data chunks.
        """

        # get episode_length, n_rollout_threads, and mini_batch_size
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // critic_num_mini_batch

        assert (
            episode_length % data_chunk_length == 0
        ), f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (n_rollout_threads, episode_length, *share_obs_shape) --> (n_rollout_threads * episode_length, *share_obs_shape)
        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.share_obs.shape[2:])
            )
        else:
            share_obs = _sa_cast(self.share_obs[:-1])
        value_preds = _sa_cast(self.value_preds[:-1])
        returns = _sa_cast(self.returns[:-1])
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        # generate mini-batches
        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(
                    rnn_states_critic[ind]
                )  # only the beginning rnn states are needed

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_critic_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch
