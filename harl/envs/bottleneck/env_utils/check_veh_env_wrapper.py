import time

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from veh_env import VehEnvironment
from veh_env_wrapper import VehEnvWrapper

# 获得全局路径
path_convert = get_abs_path(__file__)
# 设置日志 -- tshub自带的给环境的
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level='INFO')

if __name__ == '__main__':
    # base env
    sumo_cfg = path_convert("bottleneck_map_small/scenario.sumocfg")
    num_seconds = 1500  # 秒
    vehicle_action_type = 'lane_continuous_speed'
    use_gui = True
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

    ac_env = VehEnvironment(
        sumo_cfg=sumo_cfg,
        num_seconds=num_seconds,
        vehicle_action_type=vehicle_action_type,
        use_gui=use_gui,
        trip_info=trip_info,
    )

    ac_env_wrapper = VehEnvWrapper(
        env=ac_env,
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

    for constant_speed in range(1, 5):  # 测试不同的速度
        """
        在simple中
        0： 不换道
        1： 左换道
        2： 右换道
        3： 加速
        4： 减速
        """
        logger.info(f'SIM: TEST Constant Speed : {constant_speed}')
        done = False
        ac_env_wrapper.reset()
        while not done:
            # 获取环境中所有车辆的ID
            # 为每个车辆生成一个动作
            action = {ego_id: constant_speed for ego_id in ac_env_wrapper.ego_ids}
            states, rewards, truncated, dones, infos = ac_env_wrapper.step(action=action)
            done = all([dones[_ego_id] for _ego_id in ac_env_wrapper.ego_ids])
            logger.info(f'SIM: Applied action: {ac_env_wrapper.action_command}')
            logger.info(f'SIM: Speed: {ac_env_wrapper.current_speed}')
            logger.info(f'SIM: Lane: {ac_env_wrapper.current_lane}')
            logger.info(f'SIM: Reward: {rewards}')
            logger.info(f'SIM: Info: {infos}')
            logger.info(f'SIM: Warn Vehicle: {ac_env_wrapper.warn_ego_ids}')
            logger.info(f'SIM: Collision Vehicle: {ac_env_wrapper.coll_ego_ids}')
            time.sleep(10)
            if done:
                print('done')

    ac_env_wrapper.close()
