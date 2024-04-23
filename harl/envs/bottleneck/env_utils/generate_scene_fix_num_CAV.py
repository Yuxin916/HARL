import numpy as np
import traci
import libsumo as ls
import random
import math

def generate_scenario(use_gui: bool, sce_name: str, CAV_num: int, CAV_penetration: float, distribution: str):
    """
    Mixed Traffic Flow (MTF) scenario generation: v_0 = 10 m/s, v_max = 15 m/s
    use_gui: false for libsumo, true for traci
    sce_name: scenario name, e.g., "Env_Bottleneck"
    CAV_num: number of CAVs
    CAV_penetration: CAV penetration rate, only 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 can be used
    - the number of vehicles in the scenario is determined by veh_num=CAV_num/CAV_penetration
    - the number of HDVs is determined by HDV_num=veh_num-CAV_num
    - 3 types of HDVs are randomly generated, and the parameters of each type of HDVs are defined in .rou.xml
    --- HDV_0: aggressive, 0.2 probability
    --- HDV_1: cautious, 0.2 probability
    --- HDV_2: normal, 0.6 probability
    distribution: "random" or "uniform"
    """
    veh_num = math.ceil(CAV_num / CAV_penetration)
    HDV_num = veh_num - CAV_num
    # generate HDVs with different driving behaviors
    random_numbers_HDV = [random.random() for _ in range(HDV_num)]
    random_HDVs = []
    for i in range(HDV_num):
        if random_numbers_HDV[i] < 0.2:
            random_HDVs.append(0)
        elif random_numbers_HDV[i] < 0.4:
            random_HDVs.append(1)
        else:
            random_HDVs.append(2)
    # routes for HDVs and CAVs
    if sce_name == "Env_Bottleneck":
        random_route_HDVs = np.floor(1 * np.random.rand(HDV_num))
        random_route_CAVs = np.floor(1 * np.random.rand(CAV_num))

    if use_gui:
        scene_change = traci.vehicle
    else:
        scene_change = ls.vehicle

    # random - CAVs are randomly distributed
    if distribution == "random":
        for i_HDV in range(HDV_num):
            scene_change.add(
                vehID=f'HDV_{i_HDV}',
                typeID=f'HDV_{int(random_HDVs[i_HDV])}',
                routeID=f'route_{int(random_route_HDVs[i_HDV])}',
                depart="now",
                departPos="random",
                departLane="random",
                departSpeed='10',
            )

        for i_CAV in range(CAV_num):
            scene_change.add(
                vehID=f'CAV_{i_CAV}',
                typeID='ego',
                routeID=f'route_{int(random_route_CAVs[i_CAV])}',
                depart="now",
                departPos="random",
                departLane="random",
                departSpeed='10',
            )

    # uniform - CAVs are uniformly distributed
    else:
        i_CAV = 0
        i_HDV = 0
        if CAV_penetration == 0.1:
            distribution = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif CAV_penetration == 0.2:
            distribution = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        elif CAV_penetration == 0.3:
            distribution = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        elif CAV_penetration == 0.4:
            distribution = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        elif CAV_penetration == 0.5:
            distribution = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        elif CAV_penetration == 0.6:
            distribution = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        elif CAV_penetration == 0.7:
            distribution = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]
        elif CAV_penetration == 0.8:
            distribution = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        elif CAV_penetration == 0.9:
            distribution = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        else:
            distribution = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for i_all in range(veh_num):
            pos_change = -5 if i_all % 4 == 1 else 5
            pos_change = 10 if i_all % 4 == 2 else pos_change
            pos_change = 0 if i_all % 4 == 0 else pos_change
            if distribution[i_all % 10] == 1:
                scene_change.add(
                    vehID=f'CAV_{i_CAV}',
                    typeID='ego',
                    routeID=f'route_{int(random_route_CAVs[i_CAV])}',
                    depart="now",
                    departPos=f'{float(250 - i_all / 4 * 20 + pos_change)}',
                    departLane=f'{int(i_all % 4)}',
                    departSpeed='10',
                )
                i_CAV += 1
            else:
                scene_change.add(
                    vehID=f'HDV_{i_HDV}',
                    typeID=f'HDV_{int(random_HDVs[i_HDV])}',
                    routeID=f'route_{int(random_route_HDVs[i_HDV])}',
                    depart="now",
                    departPos=f'{float(250 - i_all / 4 * 20 + pos_change)}',
                    departLane=f'{int(i_all % 4)}',
                    departSpeed='10',
                )
                i_HDV += 1