import sys
sys.path.append('/home/tsaisplus/MuRPE_base/Heterogenous-MARL/harl/envs/ast/Prison_Escape')

from Prison_Escape.environment.prisoner_env import PrisonerBothEnv
from Prison_Escape.fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
import numpy as np
import gym

"""
PrisonerBlueEnv and PrisonerEnv are essentially wrappers for the PrisonerBothEnv class such that 
given the policy of the other team, we just return the observations for the desired team (red or blue).
"""

class PrisonerBlueEnv(gym.Wrapper):
    """ This environment return blue observations and takes in blue actions """
    def __init__(self,
                 env: PrisonerBothEnv,
                 fugitive_policy):
        super().__init__(env)
        self.env = env

        # evader策略
        self.fugitive_policy = fugitive_policy
        # 单个智能体状态空间
        self.observation_space = self.env.blue_observation_space
        # 单个智能体动作空间
        self.action_space = self.env.blue_team_action_space
        # 团队智能体状态空间
        self.share_observation_space = self.env.blue_team_observation_space
        # 智能体数量
        self.num_agents = self.env.num_agents
        # 智能体prefix
        self.search_parties = ["search_parties_{}".format(i) for i in range(self.env.num_search_parties)]
        self.helicopters = ["helicopters_{}".format(i) for i in range(self.env.num_helicopters)]
        self.agents = self.search_parties + self.helicopters
        # 时间步
        self.max_timesteps = self.env.max_timesteps

        self.obs_names = self.env.blue_obs_names

        # evader当前位置
        self.prisoner_location = env.get_prisoner_location()

    def reset(self, seed=None):
        self.env.reset(seed)

        if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
            self.fugitive_policy.reset()
            raise NotImplementedError("RRTStarAdversarialAvoid not test for blue policy, use astar instead")

        return self.env.blue_observation, self.env.blue_team_state
        
    def step(self, blue_action):
        # get red observation for policy
        red_obs_in = self.env.fugitive_observation
        red_action = self.fugitive_policy.predict(red_obs_in)
        # red_action: a speed and direction vector for the red agent (not important) -- red_action[0] -- ndarray(2,)
        # blue_action: a triple of [dx, dy, speed] where dx and dy is the direction vector (a norm of 1) -- blue_action
        # list of arrays. (N agent), each array is ndarray(3,)
        _, blue_obs, share_obs, reward, done, i = self.env.step_both(red_action[0], blue_action)

        done = np.full((self.num_agents, ), done)
        reward = np.full((self.num_agents, 1), reward)

        return blue_obs, share_obs, reward, done, i