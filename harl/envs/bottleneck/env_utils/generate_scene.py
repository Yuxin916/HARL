import numpy as np
import traci
import libsumo as ls

def generate_scenario(use_gui:bool, sce_name:str, HDV_num:int, CAV_num:int):
    
    random_HDVs = np.floor(3*np.random.rand(HDV_num))
    if sce_name == "Env_SingleLane":
        random_route = np.floor(4*np.random.rand(HDV_num))
    elif sce_name == "Env_Round":
        random_route = np.floor(4*np.random.rand(HDV_num))
    elif sce_name == "Env_Bottleneck":
        random_route = np.floor(1*np.random.rand(HDV_num))
    elif sce_name == "Env_Merging":
        random_route = np.floor(1.3*np.random.rand(HDV_num))
    else:
        random_route = np.floor(5*np.random.rand(HDV_num))
    if use_gui:
        scene_change=traci.vehicle
    else:
        scene_change=ls.vehicle

    '''random flow'''
    # for i in range(HDV_num):
    #     scene_change.add(
    #         vehID=f'HDV_{i}',
    #         typeID=f'HDV_{int(random_HDVs[i])}', 
    #         routeID=f'route_{int(random_route[i])}',
    #         depart="now",
    #         departPos="random", 
    #         departLane="random",
    #         departSpeed='0',
    #     )
        
    # for i in range(CAV_num):
    #     scene_change.add(
    #         vehID=f'CAV_{i}',
    #         typeID='ego',
    #         routeID=f'route_{int(random_route[i])}',
    #         depart="now",
    #         departPos="random",
    #         departLane="random",
    #         departSpeed='0',
    #     )

    '''fixed flow'''
    i_CAV = 0
    i_HDV = 0

    for i in range(HDV_num+CAV_num):
        scene_change.add(
            vehID=f'CAV_{i}',
            typeID='ego',
            routeID='route_0',
            depart="now",
            departPos=f'{float(250 - i / 4 * 20)}',
            departLane=f'{int(i % 4)}',
            departSpeed='10',
        )

        # if i % 5 == 0:
        #     scene_change.add(
        #     vehID=f'CAV_{i_CAV}',
        #     typeID='ego',
        #     routeID='route_0',
        #     depart="now",
        #     departPos=f'{float(250-i/4*20)}',
        #     departLane=f'{int(i%4)}',
        #     departSpeed='10',
        # )
        #     i_CAV += 1
        # else:
        #     scene_change.add(
        #     vehID=f'HDV_{i_HDV}',
        #     typeID='HDV_2',
        #     routeID='route_0',
        #     depart="now",
        #     departPos=f'{float(250-i/4*20)}',
        #     departLane=f'{int(i%4)}',
        #     departSpeed='10',
        # )
        #     i_HDV += 1