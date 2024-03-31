import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
import yaml
from harl.envs.robotarium.multiagentenv import MultiAgentEnv


class TimeLimit(GymTimeLimit):
    # 主要是为了限制每个episode的最大步数
    # Issue a truncated signal if a maximum number of timesteps is exceeded.
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            for dict in info:
                dict["TimeLimit.truncated"] = not all(done)
            # info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents.
    Example::
        import gymnasium as gym
        from gymnasium.wrappers import FlattenObservation
        env = gym.make("CarRacing-v2")
        env.observation_space.shape
        (96, 96, 3)
        env = FlattenObservation(env)
        env.observation_space.shape
        (27648,)
        obs, _ = env.reset()
        obs.shape
        (27648,)
    """

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    # _GymmaWrapper类继承自MultiAgentEnv类
    # TimeLimit类和FlattenObservation类服务于_GymmaWrapper类
    def __init__(self, key, time_limit, **kwargs):
        self.episode_limit = time_limit
        # 使用gym.make()函数创建环境
        # 这里的key是在main.py的gymma.py里面读取的 - 指定环境名称
        # 先进入了robotarium_gym的__init__函数，把所有scenarios的名称都注册了
        # 然后进入了robotarium_gym的make函数  - robotarium的wrapper和HSN的环境
        # robotarium的wrapper读取了HSN的config文件，然后把config文件中的参数传递给HSN的环境
        self._env = gym.make(f"{key}")

        # 环境的第一层wrapper - 限制每个episode的最大步数
        # self._env = TimeLimit(self._env, max_episode_steps=time_limit)

        # # 环境的第二层wrapper - 把每个agent的observation进行了flatten
        # self._env = FlattenObservation(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None

        # 为了防止出现action_space和observation_space的维度不一致的情况，
        # 输出的action_space和observation_space的维度都是最大的
        # 可能未来会对action_space和observation_space进行了padding
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)
        self._seed = kwargs["seed"]
        # self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        # reward = [x * len(reward) for x in reward]
        reward = [x for x in reward]
        return [[item] for item in reward], np.array(done), info

    def get_num_agents(self):
        return self.n_agents

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_space(self):
        return self.longest_observation_space

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        # assume the state is the same as the concatenated observations
        return [np.concatenate(self._obs, axis=0).astype(np.float32)] * self.n_agents

    def get_state_space(self):
        return np.concatenate(self.get_obs_space, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_action_space(self):
        return self.longest_action_space

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def get_adj_matrix(self):
        """Returns the adjacency matrix of the environment """
        return self._env.get_adj_matrix()

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_spaces(self):
        return self._env.observation_space, self._env.state_space, self._env.action_space