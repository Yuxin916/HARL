import copy
import gym
import sys
import numpy as np
from .env_utils.veh_env import VehEnvironment
from .env_utils.veh_env_wrapper import VehEnvWrapper
from tshub.utils.get_abs_path import get_abs_path

# 获得全局路径
path_convert = get_abs_path(__file__)

def make_bottleneck_envs():
    # TODO: temp write the arguments fixed

    # base env
    sumo_cfg = path_convert("env_utils/bottleneck_map_small/scenario.sumocfg")
    num_seconds = 200  # 秒
    vehicle_action_type = 'lane_continuous_speed'
    use_gui = False
    trip_info = None

    # for veh wrapper
    scene_name = "Env_Bottleneck"
    num_HDVs = 40
    num_CAVs = 10
    warmup_steps = 0
    ego_ids = ['CAV_0', 'CAV_1', 'CAV_2',
               'CAV_3', 'CAV_4',
               'CAV_5', 'CAV_6', 'CAV_7',
               'CAV_8', 'CAV_9',
               ]
    edge_ids = ['E0', 'E1', 'E2', 'E3', 'E4', ]
    edge_lane_num = {'E0': 4,
                     'E1': 4,
                     'E2': 4,
                     'E3': 2,
                     'E4': 4,
                     }  # 每一个 edge 对应的车道数
    bottle_necks = ['E4']  # bottleneck 的 edge id
    bottle_neck_positions = (496, 0)  # bottle neck 的坐标, 用于计算距离
    calc_features_lane_ids = ['E0_0', 'E0_1',
                              'E0_2', 'E0_3',
                              'E1_0', 'E1_1',
                              'E1_2', 'E1_3',
                              'E2_0', 'E2_1',
                              'E2_2', 'E2_3',
                              'E3_0', 'E3_1',
                              'E4'
                              ]  # 计算对应的 lane 的信息
    log_path = path_convert('./log/check_veh_env')
    delta_t = 1.0

    veh_env = VehEnvironment(
        sumo_cfg=sumo_cfg,
        num_seconds=num_seconds,
        vehicle_action_type=vehicle_action_type,
        use_gui=use_gui,
        trip_info=trip_info,
    )
    veh_env = VehEnvWrapper(
        env=veh_env,
        name_scenario=scene_name,
        num_HDVs=num_HDVs,
        num_CAVs=num_CAVs,
        warmup_steps=warmup_steps,
        ego_ids=ego_ids,
        edge_ids=edge_ids,
        edge_lane_num=edge_lane_num,
        bottle_necks=bottle_necks,
        bottle_neck_positions=bottle_neck_positions,
        calc_features_lane_ids=calc_features_lane_ids,
        filepath=log_path,
        use_gui=use_gui,
        delta_t=delta_t
    )
    return veh_env


class BOTTLENECKEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.env = make_bottleneck_envs()
        self.n_agents = self.env.num_CAVs
        self.share_observation_space = list(self.env.share_observation_space.values())
        self.observation_space = list(self.env.observation_space.values())
        self.action_space = list(self.env.action_space.values())

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        action_dict = {ego_id: action[0] for ego_id, action in zip(self.env.ego_ids, actions)}
        obs, rew, truncated, done, info = self.env.step(action_dict)

        s_obs = self.convert_shared_obs(obs)
        obs = list(obs.values())
        s_obs = list(s_obs.values())
        rew = np.array(list(rew.values())).reshape((-1, 1))
        done = np.array(list(done.values()))

        return obs, s_obs, rew, done, self.repeat(info), self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        obs, _ = self.env.reset()
        s_obs = self.convert_shared_obs(obs)
        obs = list(obs.values())
        s_obs = list(s_obs.values())

        return obs, s_obs, self.get_avail_actions()

    def seed(self, seed):
        pass

    def get_avail_actions(self):

        avail_actions = [[1] * self.action_space[0].n]*self.n_agents
        return np.array(avail_actions)
        # TODO: 换道的动作被mask掉

    def close(self):
        self.env.close()

    def convert_shared_obs(self, obs_dict):
        # Concatenate all observations into one list
        all_observations = []
        for cav, observations in obs_dict.items():
            all_observations.extend(observations)

        # Create shared_obs with the same keys but with the concatenated list for each
        shared_obs = {cav: all_observations for cav in obs_dict.keys()}

        return shared_obs

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

