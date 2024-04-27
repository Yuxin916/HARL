"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        # 读取算法相关config
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]  # MLP隐藏层神经元数量
        self.rnn_hidden_size = self.hidden_sizes[-1]  # RNN隐藏层神经元数量
        self.recurrent_n = algo_args["model"]["recurrent_n"]  # RNN的层数
        self.action_aggregation = algo_args["algo"]["action_aggregation"]  # 多维动作空间的聚合方式，如mean/prod
        self.share_param = algo_args["algo"]["share_param"]  # actor是否共享参数
        self.fixed_order = algo_args["algo"]["fixed_order"]  # 是否固定agent的策略更新顺序
        set_seed(algo_args["seed"])  # 设置随机种子
        self.device = init_device(algo_args["device"])  # 设置设备

        # train, not render 说明在训练，不在eval
        if not self.algo_args["render"]["use_render"]:
            # 初始化运行路径，日志路径，保存路径，tensorboard路径
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            # 保存algo，env args，algo args所有config
            save_config(args, algo_args, env_args, self.run_dir)

        # set the title of the process
        # 设置进程的标题
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # 使用env tools中的函数创建训练/测试/render环境 （调取环境+插入env config）
        if self.algo_args["render"]["use_render"]:
            # 创建单线程render环境
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:
            # 创建多线程训练环境
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            # 创建多线程测试环境
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        # 默认使用EP作为state_type
        # EP：EnvironmentProvided global state (EP)：环境提供的全局状态
        # FP：Featured-Pruned Agent-Specific Global State (FP)： 特征裁剪的特定智能体全局状态(不同agent的全局状态不同, 需要agent number)
        self.state_type = env_args.get("state_type", "EP")

        # 智能体数量
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)
        print("num_agents: ", self.num_agents)

        # actor相关
        # actor共享参数
        if self.share_param:
            self.actor = []
            # 初始化actor网络，进入mappo.py
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},  # yaml里model和algo的config打包作为args进入OnPolicyBase
                self.envs.observation_space[0], # 单个agent的观测空间
                self.envs.action_space[0], # 单个agent的动作空间
                device=self.device,
            )
            # 因为共享参数，所以self.actor列表中只有一个actor，即所有agent共用一套actor网络
            self.actor.append(agent)

            # 因为共享参数，agent之间的观测空间和动作空间都要同构
            for agent_id in range(1, self.num_agents):
                # 所以self.envs.observation_space作为a list of obs space for each agent应该保持一致
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                # 所以self.envs.action_space list of act space for each agent应该保持一致
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
                # self.actor是一个list，里面有N个一模一样的actor，

        # actor不共享参数
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                # 给每一个agent初始化actor网络，进入mappo.py 【根据其不同的obs_dim和act_dim】
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                # 因为不共享参数，所以self.actor列表中有N个actor，所有agent每人一套actor网络
                self.actor.append(agent)

        # 训练
        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            # 给每一个agent创立buffer，初始化buffer，进入OnPolicyActorBuffer
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    # yaml里model和algo的config打包作为args进入OnPolicyActorBuffer
                    {**algo_args["train"], **algo_args["model"]},
                    # 【根据其不同的obs_dim和act_dim】
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                # self.actor_buffer列表中有N个buffer，所有agent每人一套buffer
                self.actor_buffer.append(ac_bu)

            # 单个agent的share obs space eg: Box(-inf, inf, (54,), float32)
            share_observation_space = self.envs.share_observation_space[0]

            # 创建centralized critic网络
            self.critic = VCritic(
                # yaml里model和algo的config打包作为args进入VCritic
                {**algo_args["model"], **algo_args["algo"]},
                # 中心式的值函数centralized critic的输入是单个agent拿到的share_observation_space dim
                share_observation_space,
                device=self.device,
            )

            # 创建centralized critic网络的buffer（1个）
            # MAPPO trick: 原始论文section 5.2
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                # EP的全局状态是所有agent的状态的拼接，所以所有agent的share_observation_space dim是一样的
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                )
            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.

                # FP的全局状态是EP+IND的prune版本（包含全局状态，agent specific状态，并且删除了冗余状态），因此每个agent不一样 #TODO：还没看
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                # TODO： 或许EP+IND的非prune版本？
                raise NotImplementedError

            # MAPPO trick: 原始论文 section 5.1 - PopArt？
            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            # 环境的logger
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # 可以restore之前训练到一半的模型继续训练
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def run(self):
        """Run the training (or rendering) pipeline."""

        # render,不是训练
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return

        # 开始训练
        print("start running")

        # 在环境reset之后返回的obs，share_obs，available_actions存入每一个actor的replay buffer 以及 集中式critic的replay buffer
        self.warmup()

        # 计算总共需要跑多少个episode = 总训练时间步数 / 每个episode的时间步数 / 并行的环境数 (int)
        episodes = (
                # 训练总时间步数 / 每个episode的时间步数 / 并行的环境数
                int(self.algo_args["train"]["num_env_steps"])
                // self.algo_args["train"]["episode_length"]
                // self.algo_args["train"]["n_rollout_threads"]
        )

        # 初始化logger
        self.logger.init(episodes)  # logger callback at the beginning of training

        # 开始训练！！！！！！
        # 对于每一个episode
        for episode in range(1, episodes + 1):
            # 学习率是否随着episode线性递减
            if self.algo_args["train"]["use_linear_lr_decay"]:
                # 是否共享actor网络
                if self.share_param:
                    # 在mappo继承的OnPolicyBase类中，episode是当前episode的index，episodes是总共需要跑多少个episode
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                # critic的lr_decay函数在VCritic类中，episode是当前episode的index，episodes是总共需要跑多少个episode
                self.critic.lr_decay(episode, episodes)

            # 每个episode开始的时候更新logger里面的episode index
            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            # 把actor和critic网络都切换到eval模式
            self.prep_rollout()  # change to eval mode

            # 对于所有并行环境一个episode的每一个时间步
            for step in range(self.algo_args["train"]["episode_length"]):
                """
                采样动作 - 进入actor network 
                values: (n_threads, 1) - 所有并行环境在这一个timestep的critic网络的输出
                actions: (n_threads, n_agents, 1) 
                action_log_probs: (n_threads, n_agents, 1)
                rnn_states: (进程数量, n_agents, rnn层数, rnn_hidden_dim)
                rnn_states_critic: (n_threads, rnn层数, rnn_hidden_dim)
                """
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,  # rnn_states是actor的rnn的hidden state
                    rnn_states_critic,  # rnn_states_critic是critic的rnn的hidden state
                ) = self.collect(step)

                """
                在得到动作后，执行动作 - 进入环境 ShareVecEnv | step
                与环境交互一个step，得到obs，share_obs，rewards，dones，infos，available_actions
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                """
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                """每个step更新logger里面的per_step data"""
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
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
                )

                self.logger.per_step(data)  # logger callback at each step

                """把这一步的数据存入每一个actor的replay buffer 以及 集中式critic的replay buffer"""
                self.insert(data)  # insert data into buffer

            # 收集完了一个episode的所有timestep data，开始计算return，更新网络
            # compute Q and V using GAE or not
            self.compute()

            # 结束这一个episode的交互数据收集
            # 把actor和critic网络都切换回train模式
            self.prep_training()

            # 从这里开始，mappo和happo不一样了
            actor_train_infos, critic_train_info = self.train()

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            # 把上一个episode产生的最后一个timestep的state放入buffer的新的episode的第一个timestep
            self.after_update()

    def warmup(self):
        """
        Warm up the replay buffer.
        在环境reset之后返回的obs，share_obs，available_actions存入每一个actor的replay buffer 以及 集中式critic的replay buffer
        """
        """
        reset所有的并行环境，返回
        obs: (n_threads, n_agents, obs_dim)
        share_obs: (n_threads, n_agents, share_obs_dim)
        available_actions: (n_threads, n_agents, action_dim)
        """
        obs, share_obs, available_actions = self.envs.reset()

        # 准备阶段---每一个actor的replay buffer
        for agent_id in range(self.num_agents):
            # self.actor_buffer[agent_id].obs是[episode_length+1, 进程数量, obs_shape]
            # self.actor_buffer[agent_id].obs[0]是episode在t=0时的obs [进程数量, obs_shape]
            # 更多细节看OnPolicyActorBuffer
            # 在环境reset之后，把所有并行环境下专属于agent_id的obs放入专属于agent_id的buffer的self.obs的第一步里
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()

            if self.actor_buffer[agent_id].available_actions is not None:
                # 在环境reset之后
                # 把所有并行环境下的专属于agent_id的available_actions放入专属于agent_id的buffer的self.available_actions的第一步里
                self.actor_buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

        # 准备阶段---集中式critic的replay buffer
        # 更多细节看OnPolicyCriticBufferEP/FP
        if self.state_type == "EP":
            # 在环境reset之后
            # 把所有并行环境下的专属于agent_id的share_obs放入专属于agent_id的buffer的self.share_obs的第一步里
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()  # 前向，没有反向传播，不需要计算梯度
    def collect(self, step):
        """
        Collect actions and values from actors and critics.
        从actor和critic中收集actions和values
        Args:
            step: step in the episode. 这一个episode的第几步
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
            输出values, actions, action_log_probs, rnn_states（actor）, rnn_states_critic
        """

        # 从n个actor中收集actions, action_log_probs, rnn_states
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []

        # 从critic中收集values, rnn_states_critic
        values = []
        rnn_states_critic = []

        """
        首先是actor的收集 - 伪代码12-13行
        # 对于每一个agent对应的self.actor[agent_id]
        # 给actor[agent_id]输入:  (有关输入可以参考OnPolicyActorBuffer的初始化)
            - 当前时刻obs
            - 上一时刻输出的的rnn_state
            - mask (done or not)
            - 当前智能体的可用动作
            - bool(有没有available_actions)
        # 输出:
            - action
            - action_log_prob
            - rnn_state(actor)
        """
        # 对于每一个agent来说
        for agent_id in range(self.num_agents):
            # self.actor[agent_id].get_actions参考OnPolicyBase
            # actions: (torch.Tensor) actions for the given inputs. 【thread_num, 1】
            # action_log_probs: (torch.Tensor) log probabilities of actions. 【thread_num, 1】
            # rnn_states_actor: (torch.Tensor) updated RNN states for actor. 【thread_num, rnn层数，rnn_state_dim】
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            # tensor转numpy
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))

        # 转置 (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        """
        然后是critic的收集 - 伪代码14行
        两种情况：critic的输入是所有agent obs的concate起来(EP)还是经过处理(FP)
        # 给critic输入:
            - 当前时刻的share_obs
            - 上一时刻的rnn_state_critic
            - mask
        # 输出:
            - value
            - rnn_state_critic
        """
        # collect values, rnn_states_critic from 1 critic
        # 参考v_critics.py
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)

            # tensor转numpy
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)

        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """把这一个time step的数据插入到buffer中"""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # tuple of list of Dict, shape: (n_threads, n_agents, 4)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_dim)
            values,  # EP: (n_threads, 1), FP: (n_threads, n_agents, 1)
            actions,  # (n_threads, n_agents, 1)
            action_log_probs,  # (n_threads, n_agents, 1)
            rnn_states,  # (n_threads, n_agents, rnn层数, hidden_dim)
            rnn_states_critic,  # EP: (n_threads, rnn层数, hidden_dim), FP: (n_threads, n_agents, dim)
        ) = data

        # 检查所有env thread是否done (n_threads, )
        dones_env = np.all(dones, axis=1)

        """
        重置actor和critic的rnn_state
        rnn_states: (n_threads, n_agents, rnn层数, hidden_dim)
        rnn_states_critic: (n_threads, rnn层数, hidden_dim)
        """
        # 如果哪个env done了，那么就把那个环境的rnn_state (所有actor)置为0
        rnn_states[
            dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(), #dones_env里有几个true，几个并行环境done了
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        # 如果哪个env done了，那么就把那个环境的rnn_state (critic)置为0
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        """
        重置masks
        把已经done了的env的mask由1置为0
        这个mask是表示着什么时候哪一个并行环境的rnn_state要重置
        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        array of shape (n_rollout_threads, n_agents, 1)
        """
        # 初始化所有环境的mask是1
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        # 如果哪个env done了，那么就把那个环境的mask置为0
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )
        """
        重置active_masks
        把已经死掉的agent的mask由1置为0
        # active_masks use 0 to mask out agents that have died
        array of shape (n_rollout_threads, n_agents, 1)
        """
        # 初始化所有环境的mask是1
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        # 如果哪个agent done了，那么就把那个agent的mask置为0
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        # 如果哪个env done了，那么就把那个环境的mask置为1
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        """
        重置bad_masks
        array of shape (n_rollout_threads, 1)
        """
        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        # 插入actor_buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        # 插入critic_buffer
        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs,
                rnn_states_critic,
                values,
                rewards,
                masks,
                bad_masks
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        训练开始之前，首先调用self.compute()函数计算这个episode的折扣回报
        在计算折扣回报之前，先算这个episode最后一个状态的状态值函数next_values，其shape=(环境数, 1)然后调用compute_returns函数计算折扣回报
        Compute critic evaluation of the last state, V（s-T）
        and then let buffer compute returns, which will be used during training.
        """
        # 计算critic的最后一个state的值
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )

        # next_value --- np.array shape=(环境数, 1) -- 最后一个状态的状态值
        # self.value_normalizer --- ValueNorm
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if np.all(eval_dones):
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions[0] is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def prep_rollout(self):
        """Prepare for rollout.
        把actor和critic网络都切换到eval模式
        """

        # 每一个actor
        for agent_id in range(self.num_agents):
            # 测试actor网络结构 actor_policy.eval()
            self.actor[agent_id].prep_rollout()

        # 集中式critic
        # 测试critic网络结构 critic_policy.eval()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training.
        把actor和critic网络都切换回train模式"""
        for agent_id in range(self.num_agents):
            # 开始准备训练 actor_policy.train()
            self.actor[agent_id].prep_training()
        # 开始准备训练 critic_policy.train()
        self.critic.prep_training()

    def save(self):
        """Save model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def restore(self):
        """Restore model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        if not self.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"]) + "/critic_agent" + ".pt"
            )
            self.critic.critic.load_state_dict(policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()
