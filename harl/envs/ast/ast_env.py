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
        red_policy = AStarAdversarialAvoid(raw_env, max_speed=15, cost_coeff=1000, visualize=False)

        env = PrisonerBlueEnv(raw_env, red_policy)

        # env = PrisonerEmbedEnv(env)

        self.env = env
        self.n_agents = env.num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.share_observation_space = env.share_observation_space

    def step(self, actions):
        """
        return obs, state, rewards, dones, info, available_actions
        """

        pass

    def reset(self):
        """
        return obs, state, available_actions
        """
        obs, share_obs = self.env.reset()

        return obs, share_obs, None

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        self.env.close()