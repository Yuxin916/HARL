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
        self.done_episodes_rewards = None
        self.train_episode_rewards = None
        self.start = None
        self.episodes = None
        self.episode_lens = None
        self.one_episode_len = None
        self.done_episode_infos = None

    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"

    def init(self, episodes):
        self.start = time.time()
        # episodes总个数
        self.episodes = episodes
        self.episode_lens = []
        self.one_episode_len = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=np.int)
        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_rewards = []
        self.done_episode_infos = []

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

        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            # 如果这个环境的episode结束了
            if dones_env[t]:
                # 已经done的episode的总reward
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0  # 归零这个以及done的episode的reward

                # 存一下这个已经done的episode的terminated step的信息
                self.done_episode_infos.append(infos[t][0])

                # 存一下这个已经done的episode的episode长度
                self.episode_lens.append(self.one_episode_len[t].copy())
                self.one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度

                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                # if not self.done_episode_infos[t]['episode_return'] * self.env_args['n_agents'] == \
                #        self.done_episodes_rewards[t]:
                #     print('stop here')
                assert self.done_episode_infos[t]['episode_return'] * self.env_args['n_agents'] == \
                       self.done_episodes_rewards[t], 'episode reward not match'
                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                assert self.done_episode_infos[t]['episode_steps'] == self.episode_lens[t], 'episode len not match'

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
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

        # 记录每个episode的平均total overlap
        average_total_overlap = np.mean([info["total_overlap"] for info in self.done_episode_infos])
        self.writter.add_scalars(
            "average_total_overlap",
            {"average_total_overlap": average_total_overlap},
            self.total_num_steps,
        )
        # 记录每个episode的平均total reward
        average_total_reward = np.mean([info["episode_return"] for info in self.done_episode_infos])

        # 记录每个episode的平均edge count
        average_edge_count = np.mean([info["edge_count"] for info in self.done_episode_infos])
        self.writter.add_scalars(
            "average_edge_count",
            {"average_edge_count": average_edge_count},
            self.total_num_steps,
        )

        # 记录每个episode的平均violations
        average_violations = np.mean([info["violation_occurred"] for info in self.done_episode_infos])
        self.writter.add_scalars(
            "average_violations",
            {"average_violations": average_violations},
            self.total_num_steps,
        )

        self.done_episode_infos = []

        # 记录每个episode的平均长度
        average_episode_len = (
            np.mean(self.episode_lens) if len(self.episode_lens) > 0 else 0.0
        )
        self.episode_lens = []

        self.writter.add_scalars(
            "average_episode_length",
            {"average_episode_length": average_episode_len},
            self.total_num_steps,
        )

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

        # 记录每个episode的平均 episode reward
        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []