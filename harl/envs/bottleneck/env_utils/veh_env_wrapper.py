import time
from typing import Any, SupportsFloat, Tuple, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from loguru import logger

# from .generate_scene import generate_scenario
from .generate_scene_MTF import generate_scenario
from .wrapper_utils import (
    analyze_traffic,
    compute_ego_vehicle_features,
    compute_centralized_vehicle_features,
    check_collisions_based_pos,
    check_collisions
)
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
# 获得全局路径
path_convert = get_abs_path(__file__)
# 设置日志 -- tshub自带的给环境的
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level='ERROR')

# TODO: 在MARL actor中可以考虑分开encode - 地图信息 - 单车信息 - 所有车的信息 （借鉴MAPPO对state的处理

# TODO: sumo 底层 允许换道碰撞 + 判断

# TODO: 我明天需要确认一下我的evaluation指标是什么

# TODO: 如何在时序上或者空间上让他知道后面的bottleneck需要减速慢行  - 距离bottleneck的距离作为一个factor
# TODO： 如何在时序或者空间上提高bottleneck区域的重要性

# TODO: 在E3全速前进 - rule based

# TODO: reward design is related to the CAV penetration rate, if the CAV penetration rate is higher, the weight of the global reward is higher???

GAP_THRESHOLD = 1.5
WARN_GAP_THRESHOLD = 3.0


class VehEnvWrapper(gym.Wrapper):
    """Vehicle Env Wrapper for vehicle info
    """

    def __init__(self, env: Env,
                 name_scenario: str,  # 场景的名称
                 CAV_penetration: float,  # HDV 的数量
                 num_CAVs: int,  # CAV 的数量
                 num_HDVs: int,  # HDV 的数量
                 ego_ids: List[str],  # ego vehicle id
                 edge_ids: List[str],  # 路网中所有路段的 id
                 edge_lane_num: Dict[str, int],  # 每个 edge 的车道数
                 calc_features_lane_ids: List[str],  # 需要统计特征的 lane id
                 bottle_necks: List[str],  # 路网中 bottleneck
                 bottle_neck_positions: Tuple[float],  # bottle neck 的坐标
                 filepath: str,  # 日志文件的路径
                 delta_t: int,  # 动作之间的间隔时间
                 warmup_steps: int,  # reset 的时候仿真的步数, 确保 ego vehicle 可以全部出现
                 use_gui: bool,  # 是否使用 GUI
                 ) -> None:
        super().__init__(env)
        self.name_scenario = name_scenario
        self.CAV_penetration = CAV_penetration
        self.num_CAVs = num_CAVs
        self.num_HDVs = num_HDVs
        self.edge_ids = edge_ids
        self.edge_lane_num = edge_lane_num
        self.ego_ids = ego_ids  # 控制车辆的 id
        self.bottle_necks = bottle_necks
        self.calc_features_lane_ids = calc_features_lane_ids  # 需要统计特征的 lane id
        self.bottle_neck_positions = bottle_neck_positions  # bottle neck 的坐标
        self.warmup_steps = warmup_steps
        self.use_gui = use_gui
        self.delta_t = delta_t

        # 记录当前速度
        self.current_speed = {key: 0 for key in self.ego_ids}
        # 记录当前的lane
        self.current_lane = {key: 0 for key in self.ego_ids}

        self.action_pointer = {
            0: (0, 0),  # 不换道
            1: (1, 0),  # 左换道
            2: (2, 0),  # 右换道
            3: (0, 0),  # 加速度+2
            4: (0, 0),  # 减速-2
        }

        self.congestion_level = 0  # 初始是不堵车的
        self.vehicles_info = {}  # 记录仿真内车辆的 (初始 lane index, travel time)
        self.agent_mask = {ego_id: True for ego_id in self.ego_ids}  # RL控制的车辆是否在路网上

        # #######
        # Writer
        # #######
        logger.info(f'RL: Log Path, {filepath}')
        self.t_start = time.time()
        # self.results_writer = ResultsWriter(
        #     filepath,
        #     header={"t_start": self.t_start},
        # )
        self.rewards_writer = list()

    # #####################
    # Obs and Action Space
    # #####################
    @property
    def action_space(self):
        """直接控制 ego vehicle 的速度
        """
        return {_ego_id: gym.spaces.Discrete(5) for _ego_id in self.ego_ids}

    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,)
        )
        return {_ego_id: obs_space for _ego_id in self.ego_ids}

    @property
    def share_observation_space(self):
        share_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(132,)
        )
        return {_ego_id: share_obs_space for _ego_id in self.ego_ids}

    # ##################
    # Tools for observations
    # ##################
    def append_surrounding(self, state):
        surrounding_vehicles = {}
        """
        ^ x (-)
        |
        |
        ------> y (+)
        | 
        |
        v x (-)

        [surround_vehicle_id, relative x, relative y, relative speed]
        """
        for vehicle_id in state['vehicle'].keys():
            # 对于所有RL控制的车辆
            if vehicle_id in self.ego_ids:
                if self.use_gui:
                    import traci as traci
                else:
                    import libsumo as traci
                surrounding_vehicle = {}

                modes_follow = {
                    'left_followers': 0b000,  # Left and followers
                    'right_followers': 0b001,  # Left and leaders
                }
                # ego车的左右车道的后面的车辆
                for key, mode in modes_follow.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    for n in neighbors:
                        if n[0][:3] == 'HDV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是HDV）
                            # 3.2是lane的宽度，1.5是HDV的minGap
                            surrounding_vehicle[key] = (n[0], -3.2, -(n[1] + 1.5), relative_speed)
                        elif n[0][:3] == 'CAV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是CAV）
                            # 3.2是lane的宽度，1.0是CAV的minGap
                            surrounding_vehicle[key] = (n[0], -3.2, -(n[1] + 1.0), relative_speed)
                        else:
                            raise ValueError('Unknown vehicle type')

                modes_lead = {
                    'left_leaders': 0b010,  # Right and followers
                    'right_leaders': 0b011  # Right and leaders
                }
                # ego车的左右车道的前面的车辆
                for key, mode in modes_lead.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    for n in neighbors:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                        # traci得到的距离是ego到前车尾部减去minGap的距离，所以要加上minGap（此时后车是ego）
                        # 3.2是lane的宽度，1.0是CAV的minGap
                        surrounding_vehicle[key] = (n[0], -3.2, n[1] + 1.0, relative_speed)

                # 在当前车道上的前车
                front_vehicle = traci.vehicle.getLeader(vehicle_id)
                if front_vehicle not in [None, ()] and front_vehicle[0] != '':  # 有可能是空的
                    front_vehicle_lane = traci.vehicle.getLaneID(front_vehicle[0])
                    front_vehicle_lane_index = int(front_vehicle_lane.split('_')[-1])
                    front_vehicle_road_id = front_vehicle_lane.split('_')[0]

                    ego_lane = state['vehicle'][vehicle_id]['lane_id']
                    ego_lane_index = int(ego_lane.split('_')[-1])
                    ego_road_id = ego_lane.split('_')[0]
                    # if front_vehicle_road_id[:3] == ':J3' or ego_road_id[:3] == ':J3':
                    #     print('debug')
                    if front_vehicle_lane_index != ego_lane_index:
                        pass
                    else:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(front_vehicle[0])
                        # traci得到的距离是ego到前车尾部减去minGap的距离，所以要加上minGap（此时后车是ego）
                        surrounding_vehicle['front'] = (front_vehicle[0], 0, front_vehicle[1] + 1.0, relative_speed)

                # 在当前车道上的后车
                back_vehicle = traci.vehicle.getFollower(vehicle_id)
                if back_vehicle not in [None, ()] and back_vehicle[0] != '':  # 有可能是空的
                    back_vehicle_lane = traci.vehicle.getLaneID(back_vehicle[0])
                    back_vehicle_lane_index = int(back_vehicle_lane.split('_')[-1])
                    back_vehicle_road_id = back_vehicle_lane.split('_')[0]

                    ego_lane = state['vehicle'][vehicle_id]['lane_id']
                    ego_lane_index = int(ego_lane.split('_')[-1])
                    ego_road_id = ego_lane.split('_')[0]

                    if back_vehicle_lane_index != ego_lane_index:
                    # if back_vehicle_lane != ego_lane:
                        pass
                    else:
                        if back_vehicle[0][:3] == 'HDV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(back_vehicle[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是HDV）
                            surrounding_vehicle['back'] = (back_vehicle[0], 0, -(back_vehicle[1] + 1.5), relative_speed)
                        elif back_vehicle[0][:3] == 'CAV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(back_vehicle[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是CAV）
                            surrounding_vehicle['back'] = (back_vehicle[0], 0, -(back_vehicle[1] + 1.0), relative_speed)

                surrounding_vehicles[vehicle_id] = surrounding_vehicle

                pass
        for vehicle_id in surrounding_vehicles.keys():
            state['vehicle'][vehicle_id]['surround'] = surrounding_vehicles[vehicle_id]

        return state

    # ##################
    # Tools for actions
    # ##################
    def __init_actions(self, raw_state):
        """初始化所有车辆(CAV+HDV)的速度:
        1. 所有车辆的速度保持不变, (0, -1) --> 0 表示不换道, -1 表示速度不变 [(-1, -1)表示HDV不接受RL输出]
        """
        self.actions = dict()
        for _veh_id, veh_info in raw_state['vehicle'].items():
            self.actions[_veh_id] = (-1, -1)

    def __update_actions(self, raw_action):
        """更新 ego 车辆的速度
        """
        for _veh_id in raw_action:
            if _veh_id in self.actions:  # 只更新 ego vehicle 的速度
                # 只有换道动作，不需要更新速度
                if raw_action[_veh_id] in range(0, 3):
                    self.action_command = (self.action_pointer[raw_action[_veh_id]][0], self.current_speed[_veh_id])

                elif raw_action[_veh_id] in range(3, 5):
                    if raw_action[_veh_id] == 3:
                        # the speed should not exceed 15
                        self.action_command = (0, min(15, self.current_speed[_veh_id] + self.delta_t * 2))
                    else:
                        # the speed should not be negative
                        self.action_command = (0, max(0, self.current_speed[_veh_id] - self.delta_t * 2))
                else:
                    raise ValueError(f'Action {raw_action[_veh_id]} is not in the range of 0-5')

                self.actions[_veh_id] = self.action_command

        return self.actions

    # ##########################
    # State and Reward Wrappers
    # ##########################
    def state_wrapper(self, state):
        """对原始信息的 state 进行处理, 分别得到:
        - 车道的信息
        - ego vehicle 的属性
        """
        state = state['vehicle'].copy()  # 只需要分析车辆的信息

        # 计算车辆和地图的数据
        lane_statistics, ego_statistics, reward_statistics = analyze_traffic(
            state=state, lane_ids=self.calc_features_lane_ids
        )

        # # 计算 bottle neck 处车辆的数量
        # bottleneck_veh_num = count_bottleneck_vehicles(
        #     lane_statistics=lane_statistics,
        #     bottle_necks=self.bottle_necks
        # )

        # # 计算 bottle neck 处的拥堵程度
        # self.congestion_level = calculate_congestion(
        #     bottleneck_veh_num,
        #     length=300,  # ['E3'] 的长度
        #     num_lane=4
        # ) # TODO： 他们使用cogestion level来认为调整速度

        # 计算每个 ego vehicle 的 state 拼接为向量
        feature_vectors = compute_ego_vehicle_features(
            lane_statistics=lane_statistics,
            ego_statistics=ego_statistics,
            unique_edges=self.edge_ids,
            edge_lane_num=self.edge_lane_num,
            bottle_neck_positions=self.bottle_neck_positions,

        )
        shared_feature_vectors = compute_centralized_vehicle_features(lane_statistics,
                                                                      feature_vectors,
                                                                      self.bottle_neck_positions)

        return feature_vectors, shared_feature_vectors, lane_statistics, ego_statistics, reward_statistics

    def reward_wrapper(self, lane_statistics, ego_statistics, reward_statistics) -> float:
        """
        根据 ego vehicle 的状态信息和 reward 的统计信息计算 reward
        我希望：
            global reward
                1. 所有车CAV+HDV都能够尽可能时间短的到达终点
                2. 所有CAV的平均速度尽可能的接近最快速度
                3.

            special reward (near bottleneck area)
                1. 在通过bottleneck的时候尽量全速通过
                2. CAV车辆尽可能的减速
                2. CAV车辆尽可能的保持距离


            local reward
                1. 每个单独的CAV尽可能不要和其他车辆碰撞
                    a. 警告距离
                    b. 碰撞距离
                2. 每个单独的CAV尽可能的保持最快速度
                3. 每个单独的CAV离开路网的奖励

        """
        max_speed = 15  # ego vehicle 的最快的速度

        # 先把reward_statistics中所有的车辆的信息都全局记录下来self.vehicles_info
        for veh_id, (road_id, distance, speed, position_x, position_y, waiting_time,
                     accumulated_waiting_timecollision) in reward_statistics.items():
            self.vehicles_info[veh_id] = [
                self.vehicles_info.get(veh_id, [0, None])[0] + 1,  # travel time
                road_id,
                distance,
                speed,
                position_x,
                position_y,
                waiting_time,
                accumulated_waiting_timecollision
            ]
        # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
        if len(self.out_of_road) > 0:
            for veh_id in self.out_of_road:
                if veh_id in self.vehicles_info:
                    del self.vehicles_info[veh_id]

        # 开始计算reward
        inidividual_rew_ego = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}
        range_reward_ego = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}

        all_vehicle_speed = []
        all_ego_vehicle_speed = []
        all_vehicle_accumulated_waiting_time = []
        all_ego_vehicle_accumulated_waiting_time = []

        for veh_id, (veh_travel_time, road_id, distance, speed, position_x,
                     position_y, waiting_time, accumulated_waiting_time) in list(self.vehicles_info.items()):

            all_vehicle_speed.append(distance / veh_travel_time)
            all_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)

            # 把CAV单独取出来
            if veh_id in self.ego_ids:
                # 计算CAV车辆的累积平均速度
                ego_mean_speed = distance / veh_travel_time  # TODO: 这个和target speed
                all_ego_vehicle_speed.append(ego_mean_speed)

                # CAV车辆的速度越靠近最大速度，reward越高 - [0, 5]
                individual_speed_r = -abs(ego_mean_speed - max_speed) / max_speed * 5 + 5
                inidividual_rew_ego[veh_id] += individual_speed_r

                # 计算CAV车辆的累积平均等待时间
                ego_mean_waiting_time = accumulated_waiting_time
                all_ego_vehicle_accumulated_waiting_time.append(ego_mean_waiting_time)
                # CAV车辆的等待时间越短，reward越高 - [0, 5]
                individual_waiting_time_r = -ego_mean_waiting_time / 50 * 5 + 5
                inidividual_rew_ego[veh_id] += individual_waiting_time_r

                # 警告距离和碰撞距离
                if veh_id in self.warn_ego_ids.keys():
                    individual_warn_r = 0
                    for dis in self.warn_ego_ids[veh_id]:
                        individual_warn_r += -(WARN_GAP_THRESHOLD - dis) / (WARN_GAP_THRESHOLD - GAP_THRESHOLD) * 10
                    inidividual_rew_ego[veh_id] += individual_warn_r

                if veh_id in self.coll_ego_ids.keys():
                    individual_coll_r = 0
                    for dis in self.coll_ego_ids[veh_id]:
                        individual_coll_r += -(GAP_THRESHOLD - dis) / GAP_THRESHOLD * 500 - 5
                    inidividual_rew_ego[veh_id] += individual_coll_r

                else:
                    inidividual_rew_ego[veh_id] += 0

                # 计算局部地区的reward
                if road_id in self.bottle_necks + ['E3']:
                    # 快速穿过bottleneck区域 和最后一个lane
                    individual_botte_neck_r = -abs(speed - max_speed) / max_speed * 5 + 5
                    range_reward_ego[veh_id] = individual_botte_neck_r
                else:
                    range_reward_ego[veh_id] = 0

        # 计算全局reward
        all_mean_speed = np.mean(all_vehicle_speed)
        all_ego_mean_speed = np.mean(all_ego_vehicle_speed)
        all_mean_accumulated_waiting_time = np.mean(all_vehicle_accumulated_waiting_time)
        all_ego_vehicle_accumulated_waiting_time = np.mean(all_ego_vehicle_accumulated_waiting_time)

        global_speed_r = -abs(all_mean_speed - max_speed) / max_speed * 5 + 5
        global_ego_speed_r = -abs(all_ego_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        global_waiting_time_r = -all_mean_accumulated_waiting_time / 10
        global_ego_waiting_time_r = -all_ego_vehicle_accumulated_waiting_time / 10  # [0, 5]

        # TODO： lane_statistics  在E2的等待时间
        # rewards = {key: inidividual_rew_ego[key] + range_reward_ego[key] + global_speed_r + global_waiting_time_r for
        #            key in inidividual_rew_ego}
        # rewards = {key: inidividual_rew_ego[key] + range_reward_ego[key] + global_speed_r + global_waiting_time_r + \
        #               global_ego_speed_r + global_ego_waiting_time_r for key in inidividual_rew_ego}

        rewards = {key: inidividual_rew_ego[key] + range_reward_ego[key] + \
                        global_ego_speed_r + global_ego_waiting_time_r for key in inidividual_rew_ego}

        return rewards

    # ############
    # Collision
    # #############

    def check_collisions(self, init_state):

        ################# 碰撞检查 ###########################################
        # 简单版本 - 根据车头的两两位置计算是否碰撞
        collisions_head_vehs, collisions_head_info = check_collisions_based_pos(init_state['vehicle'],
                                                                                gap_threshold=GAP_THRESHOLD)

        # print('point to point collision:', collisions_head_vehs, collisions_head_info)

        # 稍微复杂的版本 - 根据neighbour位置计算是否碰撞
        collisions_neigh_vehs, warn_neigh_vehs, collisions_neigh_info = check_collisions(init_state['vehicle'],
                                                                                         self.ego_ids,
                                                                                         gap_threshold=GAP_THRESHOLD,
                                                                                         gap_warn_collision=WARN_GAP_THRESHOLD
                                                                                         # 给reward的警告距离
                                                                                         )
        # print('neighbour collision:', collisions_neigh_vehs, collisions__neigh_info)

        collisions_for_reward = {
            'collision': collisions_head_vehs + collisions_neigh_vehs,
            'warn': warn_neigh_vehs,
            'info': collisions_neigh_info + collisions_head_info
        }

        self.warn_ego_ids = {}
        self.coll_ego_ids = {}

        for key, value in collisions_for_reward.items():
            if key == 'warn' and len(value) != 0:
                for element in collisions_for_reward['info']:
                    if 'warn' in element:
                        if not element['CAV_key'] in self.warn_ego_ids:
                            self.warn_ego_ids.update({element['CAV_key']: [element['distance']]})
                        else:
                            # append the distance
                            self.warn_ego_ids[element['CAV_key']].append(element['distance'])

            if key == 'collision' and len(value) != 0:
                for element in collisions_for_reward['info']:
                    if 'collision' in element:
                        if not element['CAV_key'] in self.coll_ego_ids:
                            self.coll_ego_ids.update({element['CAV_key']: [element['distance']]})
                        else:
                            self.coll_ego_ids[element['CAV_key']].append(element['distance'])

    # ############
    # reset & step
    # #############

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化
        """
        # 初始化超参数
        # bottleneck 处的拥堵程度 # TODO: 根据lane statastics来计算
        self.congestion_level = 0
        # 记录仿真内所有车辆的信息 - 在reward wrapper中更新
        self.vehicles_info = {}
        # 记录行驶出路网的车辆
        self.out_of_road = []
        # 假设这些车初始化都在路网上 活着
        self.agent_mask = {ego_id: True for ego_id in self.ego_ids}
        self.current_speed = {key: 10 for key in self.ego_ids}

        # 初始化环境
        init_state = self.env.reset()
        # 生成车流
        # generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario, HDV_num=self.num_HDVs,
        #                   CAV_num=self.num_CAVs)  # generate_scene.py
        generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario, 
                          CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
                          distribution="uniform")  # generate_scene_MTF.py - "random" or "uniform" distribution
        # 初始化车辆的速度
        self.__init_actions(raw_state=init_state)

        # 对于warmup step = 0也适用
        for _ in range(self.warmup_steps + 1):
            init_state, _, _, _, _ = super().step(self.actions)
            init_state = self.append_surrounding(init_state)

            # 检查是否有碰撞
            collisions_vehs, warn_vehs, collision_infos = check_collisions(init_state['vehicle'],
                                                                           self.ego_ids,
                                                                           gap_threshold=GAP_THRESHOLD,
                                                                           gap_warn_collision=WARN_GAP_THRESHOLD)
            # reset 时不应该有碰撞
            assert len(collisions_vehs) == 0, f'Collision with {collisions_vehs} at reset!!! Regenerate the flow'
            assert len(warn_vehs) == 0, f'Warning with {warn_vehs} at reset!!! Regenerate the flow'

            # 对 state 进行处理
            feature_vectors, shared_feature_vectors, _, _, _ = self.state_wrapper(state=init_state)
            self.__init_actions(raw_state=init_state)

        return feature_vectors, shared_feature_vectors, {'step_time': self.warmup_steps}

    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        """
        # 已经死了的车辆不控制 - 从 action 中删除
        for ego_id, ego_live in self.agent_mask.items():
            if not ego_live:
                del action[ego_id]

        # 更新 action
        action = self.__update_actions(raw_action=action).copy()
        # 在环境里走一步
        init_state, rewards, truncated, _dones, infos = super().step(action)
        init_state = self.append_surrounding(init_state)
        self.current_speed = {key: init_state['vehicle'][key]['speed'] if key in init_state['vehicle'] else 0 for key in self.ego_ids}
        self.current_lane = {key: init_state['vehicle'][key]['lane_id'] if key in init_state['vehicle'] else 0 for key in self.ego_ids}

        ################# 碰撞检查 ###########################################
        self.check_collisions(init_state)
        ####################################################################

        # 对 state 进行处理 (feature_vectors的长度是没有行驶出CAV的数量)
        feature_vectors, shared_feature_vectors, lane_statistics, ego_statistics, reward_statistics = self.state_wrapper(state=init_state)

        # 处理离开路网的车辆 agent_mask 和 out_of_road
        for _ego_id in self.ego_ids:
            if _ego_id not in feature_vectors:
                assert _ego_id not in ego_statistics, f'ego vehicle {_ego_id} should not be in ego_statistics'
                assert _ego_id not in reward_statistics, f'ego vehicle {_ego_id} should not be in reward_statistics'
                self.agent_mask[_ego_id] = False  # agent 离开了路网, mask 设置为 False
                if _ego_id not in self.out_of_road:
                    self.out_of_road.append(_ego_id)

        # 初始化车辆的速度
        self.__init_actions(raw_state=init_state)

        # 处理 dones 和 infos
        if len(self.coll_ego_ids) == 0 and len(feature_vectors) > 0:  # 还有车在路上 且还没有碰撞发生
            # 计算此时的reward （这里的reward只有还在路网上的车的reward）
            rewards = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)

        elif len(self.coll_ego_ids) > 0 and len(feature_vectors) > 0:  # 还有车在路上 但有车辆碰撞
            # 计算此时的reward
            rewards = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)  # 更新 veh info
            for collid_ego_id in self.coll_ego_ids:
                infos['collision'].append(collid_ego_id)
                self.agent_mask[collid_ego_id] = False
            infos['done_reason'] = 'collision'

        else:  # 所有RL车离开的时候, 就结束
            assert len(feature_vectors) == 0, f'All RL vehicles should leave the environment'
            infos['done_reason'] = 'all RL vehicles leave the environment'
            rewards = {}
            while self.vehicles_info:  # 仿真到没有车在 self.edge_ids
                init_state, _, _, _, _ = super().step(self.actions)
                init_state = self.append_surrounding(init_state)
                feature_vectors, shared_feature_vectors, _, _, reward_statistics = self.state_wrapper(state=init_state)
                reward = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)  # 更新 veh info
                self.__init_actions(raw_state=init_state)
                if not rewards:
                    rewards = reward.copy()
                else:
                    ego_id = list(ego_statistics.keys())[0]
                    rewards[ego_id] += reward[ego_id]

        # 处理以下reward
        if len(self.out_of_road) > 0:
            for out_of_road_ego_id in self.out_of_road:
                rewards[out_of_road_ego_id] = 0.0  # 离开路网之后 reward 也是 0  # TODO: 注意一下dead mask MARL
                infos['out_of_road'].append(out_of_road_ego_id)
                self.agent_mask[out_of_road_ego_id] = False
                feature_vectors[out_of_road_ego_id] = [0.0] * 40
                pass

        # 处理以下 infos
        if len(self.warn_ego_ids) > 0:
            infos['warning'].append(self.warn_ego_ids)

        # 处理以下done
        dones = {}
        for _ego_id in self.ego_ids:
            dones[_ego_id] = not self.agent_mask[_ego_id]

        # 只要有一个车辆碰撞，就结束所有车辆的仿真
        if len(self.coll_ego_ids) > 0:
            for ego_id in self.ego_ids:
                dones[ego_id] = True

        # # 只要有一个车辆碰撞，不要结束所有车辆的仿真
        # if len(self.coll_ego_ids) > 0:
        #     for ego_id in self.coll_ego_ids:
        #         dones[ego_id] = True

        # 超出时间 结束仿真
        if infos['step_time'] >= 200:
            for ego_id in self.ego_ids:
                dones[ego_id] = True
                infos['done_reason'] = 'time out'
                # TODO： 是否需要更新reward time penalty

        # 记录结果 #TODO
        # self.rewards_writer.append(float(sum(rewards.values())))
        # if all(dones.values()):  # 所有结束才算结束
        #     ep_rew = sum(self.rewards_writer)
        #     ep_len = len(self.rewards_writer)
        #     ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
        #     self.results_writer.write_row(ep_info)
        #     self.rewards_writer = list()

        # For DEBUG
        # if all([dones[_ego_id] for _ego_id in self.ego_ids]):
        #     print('stop here')

        return feature_vectors, shared_feature_vectors, rewards, dones.copy(), dones.copy(), infos

    def close(self) -> None:
        return super().close()

    # TODO: MARL最好知道上一刻的动作
