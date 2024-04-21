from typing import Dict

import gymnasium as gym
from tshub.tshub_env.tshub_env import TshubEnvironment

class VehEnvironment(gym.Env):
    def __init__(self,
                 sumo_cfg: str,
                 num_seconds: int,
                 vehicle_action_type: str,
                 use_gui: bool = False,
                 trip_info: str = None,  # 添加 trip_info 参数
                 ) -> None:
        super().__init__()
        self.num_seconds = num_seconds  # 一个episode的时间长度
        self.traffic_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            is_vehicle_builder_initialized=True,  # 只需要获得车辆的信息
            is_aircraft_builder_initialized=False,
            is_traffic_light_builder_initialized=False,
            is_map_builder_initialized=False,
            is_person_builder_initialized=False,

            vehicle_action_type=vehicle_action_type,  # 车辆的action类型
            num_seconds=num_seconds,  # 一个episode的时间长度
            use_gui=use_gui,
            is_libsumo=(not use_gui),  # 如果不开界面, 就是用 libsumo
            trip_info=trip_info,
            collision_action='warn'
        )

    def reset(self):
        state_infos = self.traffic_env.reset()
        return state_infos

    def step(self, action: Dict[str, Dict[str, int]]):
        action = {'vehicle': action}
        states, rewards, infos, dones = self.traffic_env.step(action)
        truncated = dones
        # 添加一些额外的信息
        infos['collision'] = []
        infos['warning'] = []
        infos['out_of_road'] = []

        return states, rewards, truncated, dones, infos

    def close(self) -> None:
        self.traffic_env._close_simulation()



if __name__ == '__main__':
    # 无法测试 因为车流文件需要在veh_wrapper中生成
    pass