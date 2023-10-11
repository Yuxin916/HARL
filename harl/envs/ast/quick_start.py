# from Prison_Escape.environment.obs_embedding_wrapper import PrisonerGNNEnv
# from Prison_Escape.environment.gnn_wrapper import PrisonerGNNEnv
from Prison_Escape.environment.load_environment import load_environment
from Prison_Escape.environment.prisoner_perspective_envs import PrisonerBlueEnv

# evader策略
# from Prison_Escape.fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
# from Prison_Escape.fugitive_policies.heuristic import HeuristicPolicy
from Prison_Escape.fugitive_policies.a_star_avoid import AStarAdversarialAvoid

# pursuer策略
# from blue_policies.heuristic import BlueHeuristic
from Prison_Escape.blue_policies.random_multi_discrete import create_random_policy

# 其他
import numpy as np


# config for raw env
raw_env_path = "/home/tsaisplus/MuRPE_base/Opponent-Modeling-Env/Prison_Escape/environment/configs/mytest.yaml"

# 原始环境
raw_env = load_environment(raw_env_path)

# 原文提供三种策略，使用AStarAdversarialAvoid
# red_policy = HeuristicPolicy(raw_env, epsilon=0.0)
red_policy = AStarAdversarialAvoid(raw_env, max_speed=15, cost_coeff=1000, visualize=False)
# red_policy = RRTStarAdversarialAvoid(raw_env,
#                                      n_iter=1500,
#                                      step_len=150,
#                                      search_radius=150,
#                                      max_speed=9,
#                                      terrain_cost_coef=500,
#                                      visualize=False,
#                                      gamma=15,
#                                      goal_sample_rate=0.1,
#                                      epsilon=0.1
#                                      )

# wrap原始环境，使用PrisonerBlueEnv输入red_policy
env = PrisonerBlueEnv(raw_env, red_policy)

DEBUG = False

if DEBUG:

    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(f'num_agents: {env.num_agents}')
    # share_observation_space
    pass
#
#     env_instance, info = env
#     obs = env_instance.reset()
#     # print(obs['search_parties_0'])
#     obp = env_instance.observation_space
#     # print(obp)
#     print(obp.contains(obs['search_parties_0']))
#     # pass
#     #
#
#     for i in range(1000):
#         blue_actions = [np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 127])]
#         # transform blue_actions to try_action_dict, with prefix added search_parties_ and helicopters_
#         try_action_dict = {}
#         num_search_parties = 3
#         num_helicopters = 1
#         try_action_dict = {f'search_parties_{i}': blue_actions[i] for i in range(num_search_parties)}
#         try_action_dict.update({f'helicopters_{j}': blue_actions[-j-1] for j in range(num_helicopters)})
#         # print(try_action_dict)
#         #
#         obs, rewards, dones, info = env_instance.step(try_action_dict)
#         # print(obs['search_parties_0'])
#         print(obp.contains(obs['search_parties_0']))
#         env_instance.render('heuristic', show=True, fast=True, show_delta=True)

# env = PrisonerGNNEnv(env)
# print(env.action_space)
# print(env.observation_space)
# print(env.total_agents_num)

# Reset the environment and policy
# gnn_obs, blue_obs = env.reset()


blue_obs = env.reset()

# # what MARL should do. now use Heuristic policy for blue
# blue_policy = BlueHeuristic(env, debug=False)
# blue_policy.reset()
# blue_policy.reset()
# blue_policy.init_behavior()



for i in range(1000):
    # blue_actions = blue_policy.predict(blue_obs)
    # blue_actions = [np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 15]), np.array([0.0, 1.0, 127])]
    # blue_actions = [np.array([20, 20]), np.array([20, 20]), np.array([20, 20]), np.array([20, 20])]
    blue_actions = create_random_policy(env.num_agents,
                                       action_dim=2,
                                       min_action=-100,
                                       max_action=100)

    blue_obs, reward, done, _ = env.step(blue_actions)
    # print("blue_obs", blue_obs)
    env.render('heuristic',
               show=True,
               fast=True,
               scale=3,  # fast canvas window size FIXED
               show_delta=True,  # show a square around evader
               show_grid=True  # show grid
               )

    pass

