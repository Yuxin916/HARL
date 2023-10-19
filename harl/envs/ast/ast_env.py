from harl.envs.ast.Prison_Escape.environment.prisoner_perspective_envs import PrisonerBlueEnv
from harl.envs.ast.Prison_Escape.fugitive_policies.a_star_avoid import AStarAdversarialAvoid

from harl.envs.ast.load_env import load_environment

class ASTEnv():

    def __init__(self, env_args):
        """
        self.env = ...
        self.n_agents = ...
        self.share_observation_space = ...
        self.observation_space = ...
        self.action_space = ...
        """
        raw_env = load_environment(env_args)
        red_policy = AStarAdversarialAvoid(raw_env,
                                           max_speed=15,  # FIXED
                                           cost_coeff=1000,
                                           visualize=False)

        env = PrisonerBlueEnv(raw_env, red_policy)

        # env = PrisonerEmbedEnv(env)

        self.env = env
        # 智能体prefix
        self.agents = self.env.agents
        # 智能体数量
        self.n_agents = self.env.num_agents
        # 单个智能体状态空间
        single_observation_space = self.env.observation_space
        # 团队状态空间
        self.observation_space = self.repeat(single_observation_space)

        # 团队动作空间
        self.action_space = self.unwrap(self.env.action_space)

        # 共享状态空间
        single_share_observation_space = self.env.share_observation_space
        self.share_observation_space = self.repeat(single_share_observation_space)

    def step(self, actions):
        """
        return obs, state, rewards, dones, info, available_actions
        """
        blue_obs, share_obs, reward, done, info = self.env.step(actions)

        blue_obs = self.repeat(blue_obs)
        share_obs = self.repeat(share_obs)
        info = self.repeat(info)

        # obs: (n_agents, obs_dim)
        # share_obs: (n_agents, share_obs_dim)
        # rewards: (n_agents, 1)
        # dones: (n_agents,)
        # infos: (n_agents,)
        # available_actions: None or (n_agents, action_number)
        return blue_obs, share_obs, reward, done, info, self.get_avail_actions()

    def reset(self):
        """
        return
        obs --> list of n_agents. Each is (obs_dim 79, ) array
        share_obs --> list of n_agents. Each is (share_obs_dim 316, )
        available_actions --> can be none
        """
        obs, share_obs = self.env.reset()

        obs = self.repeat(obs)
        share_obs = self.repeat(share_obs)

        return obs, share_obs, self.get_avail_actions()

    def seed(self, seed):
        pass

    def render(self):
        self.env.render('heuristic',
                   show=True,
                   fast=True,
                   scale=3,  # fast canvas window size FIXED
                   show_delta=True,  # show a square around evader
                   show_grid=True  # show grid
                   )

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def close(self):
        self.env.close()

    def unwrap(self, d):
        """
        将字典d中的值取出来，组成一个列表
        """
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        """
        重复agents的状态空间和动作空间
        #FIXME： pursuer团队内部每个agent的observation是一致的，这在异构/通信受阻情况下是不成立的
        """
        return [a for _ in range(self.n_agents)]