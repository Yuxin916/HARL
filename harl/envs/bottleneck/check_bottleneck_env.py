import yaml
from bottleneck_env import BOTTLENECKEnv
import time
import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger


# 获得全局路径
path_convert = get_abs_path(__file__)
# 设置日志 -- tshub自带的给环境的
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level='INFO')


if __name__ == '__main__':
    args = yaml.load(open('/home/tsaisplus/generalized_traffic/HARL/harl/configs/envs_cfgs/bottleneck.yaml', 'r'), Loader=yaml.FullLoader)
    env = BOTTLENECKEnv(args)

    for constant_speed in range(0, 5):  # 测试不同的速度
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
        env.reset()
        while not done:
            # 获取环境中所有车辆的ID
            # 为每个车辆生成一个动作
            action = {ego_id: constant_speed for ego_id in env.ego_ids}
            states, share_states, rewards, dones, infos, _ = env.step(action)
            done = np.all(dones)
            logger.info(f'SIM: Applied action: {env.action_command}')
            logger.info(f'SIM: Speed: {env.current_speed}')
            logger.info(f'SIM: Lane: {env.current_lane}')
            logger.info(f'SIM: Reward: {rewards}')
            logger.info(f'SIM: Info: {infos[0]}')
            logger.info(f'SIM: Warn Vehicle: {env.warn_ego_ids}')
            logger.info(f'SIM: Collision Vehicle: {env.coll_ego_ids}')
            logger.info(f'SIM: Total Timesteps: {env.total_timesteps}')
            time.sleep(0.1)
            if done:
                print('done')

    env.close()
    print('debugging...')
