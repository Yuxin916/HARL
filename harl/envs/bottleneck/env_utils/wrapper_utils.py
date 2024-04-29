import itertools
import math
from typing import List, Dict, Tuple, Union

import numpy as np

LANE_LENGTHS = {
    'E0': 250,
    'E1': 150,
    'E2': 96,
    'E3': 200,
    'E4': 4,
}


def analyze_traffic(state, lane_ids):
    """
    输入：当前所有车的state  &&   需要计算feature的lane id
    输出：
    1. lane_statistics: 每一个 lane 的统计信息 - Dict, key: lane_id, value:
        - vehicle_count: 当前车道的车辆数 1
        - lane_density: 当前车道的车辆密度 1
        - lane_length: 这个车道的长度 1
        - speeds: 在这个车道上车的速度 3 (mean, max, min)
        - waiting_times: 一旦车辆开始行驶，等待时间清零 3  (mean, max, min)
        - accumulated_waiting_times: 车的累积等待时间 3 (mean, max, min)

    2. ego_statistics: ego vehicle 的统计信息
        - speed: vehicle当前车速
        - position: vehicle当前位置
        - heading: vehicle当前朝向
        - road_id: vehicle当前所在道路的 ID
        - lane_index: vehicle当前所在车道的 index
        - surroundings: vehicle周围车辆的相对位置

    3. reward_statistics: Dict, key: vehicle_id, value:
        - 这个车行驶道路的 ID. eg: 'E0'
        - 累积行驶距离
        - 当前车速
        - position[0]: 1
        - position[1]: 1
        - waiting_time: 1

    """

    # 初始化每一个 lane 的统计信息
    lane_statistics = {
        lane_id: {
            'vehicle_count': 0,  # 当前车道的车辆数
            'lane_density': 0,  # 当前车道的车辆密度
            'speeds': [],  # 在这个车道上车的速度
            'waiting_times': [],  # 一旦车辆开始行驶，等待时间清零
            'accumulated_waiting_times': [],  # 车的累积等待时间
            # 'crash_assessment': 0,  # 这个车道碰撞了多少次
            'lane_length': 0,  # 这个车道的长度
        } for lane_id in lane_ids
    }
    # 初始化 ego vehicle 的统计信息
    ego_statistics = {}
    # 初始化 hdv vehicle 的统计信息
    hdv_statistics = {}
    # 初始化 reward_statistics, 用于记录每一辆车所在的 edge, 后面用于统计车辆的 travel time
    reward_statistics = {}

    # 先收录所有车的信息 - （HDV + CAV）
    for vehicle_id, vehicle in state.items():
        lane_id = vehicle['lane_id']  # 这个车所在车道的 ID. eg: 'E0_2'
        if lane_id[:3] in [':J3']:
            lane_id = 'E4'
        elif lane_id[:3] in [':J1']:
            lane_id = 'E1'
        elif lane_id[:3] in [':J2']:
            lane_id = 'E2'

        road_id = vehicle['road_id']  # 这个车行驶道路的 ID. eg: 'E0'
        if road_id[:3] in [':J3']:
            road_id = 'E4'
        elif road_id[:3] in [':J1']:
            road_id = 'E1'
        elif road_id[:3] in [':J2']:
            road_id = 'E2'

        lane_index = vehicle['lane_index']  # 这个车所在车道的 index eg: 2
        speed = vehicle['speed']  # vehicle当前车速
        position = vehicle['position']  # vehicle当前位置
        heading = vehicle['heading']  # vehicle当前朝向

        leader_info = vehicle['leader']  # vehicle前车的信息
        distance = vehicle['distance']  # 这个车的累积行驶距离
        waiting_time = vehicle['waiting_time']  # vehicle的等待时间
        accumulated_waiting_time = vehicle['accumulated_waiting_time']  # vehicle的累积等待时间
        lane_position = vehicle['lane_position']  # vehicle 前保险杠到车道起点的距离

        # 感兴趣的lane数据统计
        if lane_id in lane_ids:
            # reward计算需要的信息
            # 记录每一个 vehicle 的 (edge id, distance, speed), 用于统计 travel time 从而计算平均速度
            reward_statistics[vehicle_id] = [road_id, distance, speed,
                                             position[0], position[1],
                                             waiting_time, accumulated_waiting_time]

        # ego车需要的信息
        if vehicle['vehicle_type'] == 'ego':
            # TODO: when will the state['leader'] be not None?
            surroundings = vehicle['surround']

            ego_statistics[vehicle_id] = [speed, position,
                                          heading, road_id, lane_index,
                                          surroundings
                                          ]
        # HDV车需要的信息
        else:
            hdv_statistics[vehicle_id] = [speed, position,
                                          heading, road_id, lane_index]

        # 根据lane_id把以下信息写进lane_statistics
        if lane_id in lane_statistics:
            stats = lane_statistics[lane_id]
            stats['lane_length'] = LANE_LENGTHS[road_id],  # 这个车道的长度
            stats['vehicle_count'] += 1
            stats['speeds'].append(vehicle['speed'])
            stats['accumulated_waiting_times'].append(vehicle['accumulated_waiting_time'])
            stats['waiting_times'].append(vehicle['waiting_time'])
            stats['lane_density'] = stats['vehicle_count'] / LANE_LENGTHS[road_id]

    # lane_statistics计算
    for lane_id, stats in lane_statistics.items():
        speeds = np.array(stats['speeds'])
        waiting_times = np.array(stats['waiting_times'])
        accumulated_waiting_times = np.array(stats['accumulated_waiting_times'])
        vehicle_count = stats['vehicle_count']
        lane_length = stats['lane_length']
        lane_density = stats['lane_density']

        if vehicle_count > 0:  # 当这个车道上有车的时候
            lane_statistics[lane_id] = [
                vehicle_count,
                lane_density,
                lane_length[0],
                np.mean(speeds), np.max(speeds), np.min(speeds),
                np.mean(waiting_times), np.max(waiting_times), np.min(waiting_times),
                np.mean(accumulated_waiting_times), np.max(accumulated_waiting_times), np.min(accumulated_waiting_times)
            ]
        else:
            lane_statistics[lane_id] = [0] * 12
    # lane_statistics转换成dict
    lane_statistics = {lane_id: stats for lane_id, stats in lane_statistics.items()}

    return lane_statistics, ego_statistics, reward_statistics, hdv_statistics


def check_collisions_based_pos(vehicles, gap_threshold: float):
    """输出距离过小的车辆的 ID, 直接根据 pos 来进行计算是否碰撞 (比较简单)

    Args:
        vehicles: 包含车辆部分的位置信息
        gap_threshold (float): 最小的距离限制
    """
    collisions = []
    collisions_info = []

    _distance = {}  # 记录每辆车之间的距离
    for (id1, v1), (id2, v2) in itertools.combinations(vehicles.items(), 2):
        dist = math.sqrt(
            (v1['position'][0] - v2['position'][0]) ** 2 \
            + (v1['position'][1] - v2['position'][1]) ** 2
        )
        _distance[f'{id1}-{id2}'] = dist
        if dist < gap_threshold:
            collisions.append((id1, id2))
            collisions_info.append({'collision': True,
                                    'CAV_key': id1,
                                    'surround_key': id2,
                                    'distance': dist,
                                    })

    return collisions, collisions_info


def check_collisions(vehicles, ego_ids, gap_threshold: float, gap_warn_collision: float):
    ego_collision = []
    ego_warn = []

    info = []

    # 把ego的state专门拿出来
    def filter_vehicles(vehicles, ego_ids):
        # Using dictionary comprehension to create a new dictionary
        # by iterating over all key-value pairs in the original dictionary
        # and including only those whose keys are in ego_ids
        filtered_vehicles = {key: value for key, value in vehicles.items() if key in ego_ids}
        return filtered_vehicles

    filtered_ego_vehicles = filter_vehicles(vehicles, ego_ids)

    for ego_key, ego_value in filtered_ego_vehicles.items():
        for surround_direction, content in filtered_ego_vehicles[ego_key]['surround'].items():
            c_info = None
            w_info = None

            # print(ego_key, 'is surrounded by:', content[0], 'with direction', surround_direction,
            # 'at distance', content[1])
            # TODO: 同一个车道和不同车道的车辆的warn gap应该是不一样！！！！11
            distance = math.sqrt(content[1] ** 2 + content[2] ** 2)
            # print('distance:', distance)
            if distance < gap_threshold:
                # print(ego_key, 'collision with', content[0])
                ego_collision.append((ego_key, content[0]))
                c_info = {'collision': True,
                          'CAV_key': ego_key,
                          'surround_key': content[0],
                          'distance': distance,
                          'relative_speed': content[3],
                          }

            elif gap_threshold <= distance < gap_warn_collision:
                ego_warn.append((ego_key, content[0]))
                w_info = {'warn': True,
                          'CAV_key': ego_key,
                          'surround_key': content[0],
                          'distance': distance,
                          'relative_speed': content[3],
                          }
            if c_info:
                info.append(c_info)
            elif w_info:
                info.append(w_info)

    return ego_collision, ego_warn, info


def check_prefix(a: str, B: List[str]) -> bool:
    """检查 B 中元素是否有以 a 开头的

    Args:
        a (str): lane_id
        B (List[str]): bottle_neck ids

    Returns:
        bool: 返回 lane_id 是否是 bottleneck
    """
    return any(a.startswith(prefix) for prefix in B)


def count_bottleneck_vehicles(lane_statistics, bottle_necks) -> int:
    """
    统计 bottleneck 处车辆的个数
    """
    veh_num = 0
    for lane_id, lane_info in lane_statistics.items():
        if check_prefix(lane_id, bottle_necks):
            veh_num += lane_info[0]  # lane_statistics的第一个是vehicle_count
    return veh_num


def calculate_congestion(vehicles: int, length: float, num_lane: int, ratio: float = 1) -> float:
    """计算 bottle neck 的占有率, 我们假设一辆车算上车间距是 10m, 那么一段路的。占有率是
        占有率 = 车辆数/(车道长度*车道数/10)
    于是可以根据占有率计算拥堵系数为:
        拥堵程度 = min(占有率, 1)

    Args:
        vehicles (int): 在 bottle neck 处车辆的数量
        length (float): bottle neck 的长度, 单位是 m
        num_lane (int): bottle neck 的车道数

    Returns:
        float: 拥堵系数 in (0,1)
    """
    capacity_used = ratio * vehicles / (length * num_lane / 10)  # 占有率
    congestion_level = min(capacity_used, 1)  # Ensuring congestion level does not exceed 100%
    return congestion_level


def calculate_speed(congestion_level: float, speed: int) -> float:
    """根据拥堵程度来计算车辆的速度

    Args:
        congestion_level (float): 拥堵的程度, 通过 calculate_congestion 计算得到
        speed (int): 车辆当前的速度

    Returns:
        float: 车辆新的速度
    """
    if congestion_level > 0.2:
        speed = speed * (1 - congestion_level)
        speed = max(speed, 1)
    else:
        speed = -1  # 不控制速度
    return speed


def one_hot_encode(value, unique_values):
    """Create an array with zeros and set the corresponding index to 1
    """
    one_hot = np.zeros(len(unique_values))
    index = unique_values.index(value)
    one_hot[index] = 1
    return one_hot.tolist()


def euclidean_distance(point1, point2):
    # Convert points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the Euclidean distance
    distance = np.linalg.norm(point1 - point2)
    return distance


def compute_ego_vehicle_features(
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        bottle_neck_positions: Tuple[float]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        bottle_neck_positions (Tuple[float]): bottle neck 的坐标
    """
    feature_vectors = {}

    for ego_id, ego_info in ego_statistics.items():
        # ############################## ego_statistics 的信息 ##############################
        speed, position, heading, road_id, lane_index, surroundings = ego_info

        # 速度归一化  - 1
        normalized_speed = speed / 15.0

        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y

        # 转向归一化 - 1
        normalized_heading = heading / 360

        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))

        # ############################## 周车信息 ##############################
        # 提取surrounding的信息 -18
        surround = []
        for index, (_, statistics) in enumerate(surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            surround.append([relat_x, relat_y, relat_v])  # relat_x, relat_y, relat_v
        flat_surround = [item for sublist in surround for item in sublist]
        # 如果周围车辆的信息不足6*3个, 补0 对齐到最多数量的周车信息
        if len(flat_surround) < 18:
            flat_surround += [0] * (18 - len(flat_surround))

        # ############################## 当前车道信息 ##############################
        # ego_id当前所在的lane
        ego_lane_id = f'{road_id}_{lane_index}'
        # 每个obs只需要一个lane的信息，不需要所有lane的信息， shared obs可以拿到所有lane的信息
        ego_lane_stats = lane_statistics[ego_lane_id]

        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)
        ego_lane_stats = ego_lane_stats[:4] + ego_lane_stats[6:7] + ego_lane_stats[9:10]

        # ############################## bottle_neck 的信息 ##############################
        # 车辆距离bottle_neck
        bottle_neck_position_x = bottle_neck_positions[0] / 700
        bottle_neck_position_y = bottle_neck_positions[1]

        distance = euclidean_distance(position, bottle_neck_positions)
        normalized_distance = distance / 700

        # ############################## 合并所有 ##############################
        # feature_vector = [normalized_speed, normalized_position_x, normalized_position_y, normalized_heading,
        #                   bottle_neck_position_x, bottle_neck_position_y, normalized_distance] + \
        #                   road_id_one_hot + lane_index_one_hot + flat_surround + all_lane_stats

        feature_vector = [normalized_speed, normalized_position_x, normalized_position_y, normalized_heading,
                          bottle_neck_position_x, bottle_neck_position_y, normalized_distance] + \
                         road_id_one_hot + lane_index_one_hot + flat_surround + ego_lane_stats

        # Assign the feature vector to the corresponding ego vehicle
        feature_vectors[ego_id] = [float(item) for item in feature_vector]

    # 保证每一个 ego vehicle 的特征长度一致
    assert all(len(feature_vector) == 40 for feature_vector in feature_vectors.values())

    # # Create a new dictionary to hold the updated feature lists
    # updated_feature_vectors = {}
    #
    # # Iterate over each key-value pair in the dictionary
    # for cav, features in feature_vectors.items():
    #     # Start with the original list of features
    #     new_features = features.copy()
    #     # Iterate over all other CAVs
    #     for other_cav, other_features in feature_vectors.items():
    #         if other_cav != cav:
    #             # Append the first four elements of the other CAV's features
    #             new_features.extend(other_features[:4])
    #     # Update the dictionary with the new list
    #     updated_feature_vectors[cav] = new_features
    #
    # for updated_feature_vector in updated_feature_vectors.values():
    #     if not len(updated_feature_vector) == 56:
    #         updated_feature_vector += [0] * (56 - len(updated_feature_vector))
    #
    # # assert all(len(updated_feature_vector) == 56 for updated_feature_vector in updated_feature_vectors.values())

    return feature_vectors


def compute_hierarchical_ego_vehicle_features(
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        bottle_neck_positions: Tuple[float],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        bottle_neck_positions (Tuple[float]): bottle neck 的坐标
    """

    # ############################## 所有HDV的信息 ############################## 13
    hdv_stats = {}
    for hdv_id, hdv_info in hdv_statistics.items():
        speed, position, heading, road_id, lane_index = hdv_info

        # 速度归一化  - 1
        normalized_speed = speed / 15.0

        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y

        # 转向归一化 - 1
        normalized_heading = heading / 360

        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))

        hdv_stats[hdv_id] = [normalized_speed, normalized_position_x, normalized_position_y,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot

    # convert to 2D array (12 * 13)  - 12 is max number of HDVs
    hdv_stats = np.array(list(hdv_stats.values()))
    if 0 < hdv_stats.shape[0] <= 12:
        # add 0 to make sure the shape is (12, 13)
        hdv_stats = np.vstack([hdv_stats, np.zeros((12 - hdv_stats.shape[0], 13))])
    elif hdv_stats.shape[0] == 0:
        hdv_stats = np.zeros((12, 13))

    # ############################## 所有CAV的信息 ############################## 13
    cav_stats = {}
    ego_stats = {}
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        speed, position, heading, road_id, lane_index, surroundings = ego_info

        # 速度归一化  - 1
        normalized_speed = speed / 15.0

        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y

        # 转向归一化 - 1
        normalized_heading = heading / 360

        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))

        # ############################## 周车信息 ############################## 18
        # 提取surrounding的信息 -18
        surround = []
        for index, (_, statistics) in enumerate(surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            surround.append([relat_x, relat_y, relat_v])  # relat_x, relat_y, relat_v
        flat_surround = [item for sublist in surround for item in sublist]
        # 如果周围车辆的信息不足6*3个, 补0 对齐到最多数量的周车信息
        if len(flat_surround) < 18:
            flat_surround += [0] * (18 - len(flat_surround))

        cav_stats[ego_id] = [normalized_speed, normalized_position_x, normalized_position_y,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot
        ego_stats[ego_id] = [normalized_speed, normalized_position_x, normalized_position_y,
                             normalized_heading] + road_id_one_hot + lane_index_one_hot
    # convert to 2D array (5 * 5)
    cav_stats = np.array(list(cav_stats.values()))
    if 0 < cav_stats.shape[0] <= 12:
        # add 0 to make sure the shape is (12, 13)
        cav_stats = np.vstack([cav_stats, np.zeros((12 - cav_stats.shape[0], 13))])
    elif cav_stats.shape[0] == 0:
        cav_stats = np.zeros((12, 13))

    if len(ego_stats) != len(ego_ids):
        for ego_id in ego_ids:
            if ego_id not in ego_stats:
                ego_stats[ego_id] = [0.0] * 13

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}

    # Iterate over all possible lanes to get their statistics
    for lane_id, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean)

        all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[9:10]

    # convert to 2D array (18 * 6)
    all_lane_stats = np.array(list(all_lane_stats.values()))

    # ############################## bottle_neck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    feature_vector = {}
    feature_vector['hdv_stats'] = hdv_stats
    feature_vector['cav_stats'] = cav_stats
    feature_vector['all_lane_stats'] = all_lane_stats
    feature_vector['bottle_neck_position'] = np.array([bottle_neck_position_x, bottle_neck_position_y])

    feature_vectors = {}
    for ego_id in ego_statistics.keys():
        feature_vectors[ego_id] = feature_vector.copy()
        feature_vectors[ego_id]['self_stats'] = ego_stats[ego_id]

    # A function to flatten a dictionary structure into 1D array
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        return np.array(flat_list)

    # Flatten the dictionary structure
    feature_vectors_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               feature_vectors.items()}

    return feature_vectors, feature_vectors_flatten


def compute_centralized_vehicle_features(lane_statistics, feature_vectors, bottle_neck_positions):
    shared_features = {}

    # ############################## 所有车的速度 位置 转向信息 ##############################
    all_vehicle = []
    for _, ego_feature in feature_vectors.items():
        all_vehicle += ego_feature[:4]

    # ############################## lane_statistics 的信息 ##############################
    # Initialize a list to hold all lane statistics
    all_lane_stats = []

    # Iterate over all possible lanes to get their statistics
    for _, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)

        all_lane_stats += lane_info[:4] + lane_info[6:7] + lane_info[9:10]

    # ############################## bottleneck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    for ego_id in feature_vectors.keys():
        shared_features[ego_id] = [bottle_neck_position_x, bottle_neck_position_y] + all_vehicle + all_lane_stats

    # assert all(len(shared_feature) == 130 for shared_feature in shared_features.values())

    return shared_features


def compute_centralized_vehicle_features_hieratical_version(lane_statistics, feature_vectors, feature_vectors_flatten):
    shared_features = {}

    for ego_id in feature_vectors.keys():
        shared_features[ego_id] = feature_vectors[ego_id].copy()

    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        return np.array(flat_list)

    shared_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               shared_features.items()}
    if len(shared_features_flatten) != 5:
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in shared_features_flatten:
                shared_features_flatten[ego_id] = np.zeros(435)

    return shared_features, shared_features_flatten
