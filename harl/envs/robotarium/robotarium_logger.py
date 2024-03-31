from harl.common.base_logger import BaseLogger
import time
from functools import reduce
import numpy as np


class RobotariumLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(RobotariumLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )

        # declare some variables

        self.total_num_steps = None
        self.episode = None
        self.start = None
        self.episodes = None
        self.episode_lens = None
        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        self.train_episode_rewards = None
        self.one_episode_len = None
        self.done_episode_infos = None
        self.done_episodes_rewards = None
        self.done_episode_lens = None


    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"

    def init(self, episodes):
        # 记录训练开始时间
        self.start = time.time()
        # episodes总个数
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(self.n_rollout_threads)
        self.one_episode_len = np.zeros(self.n_rollout_threads, dtype=int)
        self.done_episodes_rewards = np.zeros(self.n_rollout_threads)
        self.done_episode_lens = np.zeros(self.n_rollout_threads)
        self.done_episode_infos = [{} for _ in range(self.n_rollout_threads)]

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        # 当前是第几个episode
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        # 并行环境中的每个环境是否done （n_env_threads, ）
        dones_env = np.all(dones, axis=1)
        # 并行环境中的每个环境的step reward （n_env_threads, ）
        reward_env = np.mean(rewards, axis=1).flatten()
        # 并行环境中的每个环境的episode reward （n_env_threads, ）累积
        self.train_episode_rewards += reward_env
        # 并行环境中的每个环境的episode len （n_env_threads, ）累积
        self.one_episode_len += 1

        for t in range(self.n_rollout_threads):
            # 如果这个环境的episode结束了
            if dones_env[t]:
                # 已经done的episode的总reward
                self.done_episodes_rewards[t] = self.train_episode_rewards[t]
                self.train_episode_rewards[t] = 0  # 归零这个以及done的episode的reward

                # 存一下这个已经done的episode的terminated step的信息
                self.done_episode_infos[t] = infos[t][0]

                # 存一下这个已经done的episode的episode长度
                self.done_episode_lens[t] = self.one_episode_len[t]
                self.one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度

                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                # assert round(self.done_episode_infos[t]['episode_return'], 2) == \
                #        round(self.done_episodes_rewards[t], 2) or \
                #        round(self.done_episode_infos[t]['episode_return'],2) == \
                #        round(self.done_episodes_rewards[t],2), 'episode reward not match'
                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                assert self.done_episode_infos[t]['episode_steps'] == self.done_episode_lens[t], 'episode len not match'


    def episode_log(
            self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""

        # 记录训练结束时间
        self.end = time.time()

        # 当前跑了多少time steps
        self.total_num_steps = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()

        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        # 检查哪个环境done了
        a = [index for index, info in enumerate(self.done_episode_infos) if
                                       "episode_return" in info]
        b = [index for index, value in enumerate(self.done_episode_lens) if value != 0]
        c = [index for index, value in enumerate(self.done_episodes_rewards) if value != 0]
        assert a == b == c
        indices = a


        # # 记录每个episode的平均total overlap
        # average_total_overlap = np.mean([info["total_overlap"] for info in self.done_episode_infos])
        # self.writter.add_scalars(
        #     "average_total_overlap",
        #     {"average_total_overlap": average_total_overlap},
        #     self.total_num_steps,
        # )
        # 记录每个episode的平均total reward 和 total step
        episode_returns = [self.done_episode_infos[index]["episode_return"] for index in indices]
        episode_step = [self.done_episode_infos[index]["episode_steps"] for index in indices]

        # 记录每个episode的平均avergae reward 和 average step
        average_episode_return = np.mean(episode_returns) if episode_returns else 0
        average_episode_step = np.mean(episode_step) if episode_step else 0

        self.writter.add_scalars(
            "average_episode_length",
            {"average_episode_length": average_episode_step},
            self.total_num_steps,
        )
        print(
            "Some episodes done, average episode length is {}.\n".format(
                average_episode_step
            )
        )

        print(
            "Some episodes done, average episode reward is {}.\n".format(
                average_episode_return*self.num_agents
            )
        )
        self.writter.add_scalars(
            "train_episode_rewards",
            {"aver_rewards": average_episode_return*self.num_agents},
            self.total_num_steps,
        )

        for index in indices:
            self.done_episode_infos[index] = {}
            self.done_episode_lens[index] = 0
            self.done_episodes_rewards[index] = 0

        # 记录每个episode的平均 step reward
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)
        self.writter.add_scalars(
            "average_step_rewards",
            {"average_step_rewards": critic_train_info["average_step_rewards"]},
            self.total_num_steps,
        )
        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )
