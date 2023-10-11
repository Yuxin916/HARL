""" Generate a list of cameras to cover n percentage of the map, continuously sample"""

import numpy as np

from Prison_Escape.environment.terrain import Terrain
from Prison_Escape.environment.forest_coverage.generate_square_map import generate_square_map

def dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def produce_camera_distribution(dim_x, dim_y, camera_range, target_camera_density, terrain=None, unknown_hideout_locations=None):
    """
    Produce a camera distribution that a certain percentage of the map
    
    Args:
        dim_x: x dimension of the map
        dim_y: y dimension of the map
        camera_range: range of the camera
        target_camera_density: percentage of the map covered by cameras
    """

    unknown_hideouts = unknown_hideout_locations
    camera_density = 0
    camera_set = set()
    while camera_density < target_camera_density:
        x = np.random.randint(0, dim_x)
        y = np.random.randint(0, dim_y)

        camera_location = (x, y)
        if terrain:
            if terrain.location_in_mountain(camera_location):
                continue
        
        in_range_check = sum([dist(camera_location, camera) < camera_range for camera in camera_set])
        dist_hideouts = sum([dist(hideout, camera_location) < camera_range for hideout in unknown_hideouts])
        if camera_location not in camera_set and in_range_check == 0 and dist_hideouts == 0:
            camera_set.add(camera_location)

        camera_density = (len(camera_set) * np.pi * camera_range**2) / (dim_x * dim_y)
    print(f'camera_density: {camera_density}')

    return list(camera_set)

def write_cameras_to_file(camera_list, file_path):
    """
    Write camera locations to a file
    """
    with open(file_path, 'w') as f:
        for camera in camera_list:
            f.write(f'u,{camera[0]},{camera[1]}\n')

if __name__ == "__main__":
    raw_env_path = "/home/tsaisplus/MuRPE_base/Opponent-Modeling-Env/Prison_Escape/environment/configs/mytest.yaml"
    import yaml

    with open(raw_env_path, 'r') as stream:
        data = yaml.safe_load(stream)
    DIM_X = data['terrain_x']
    DIM_Y = data['terrain_y']

    forest_color_scale = 1

    percent_dense = data['percent_dense']
    size_of_dense_forest = int(DIM_X * percent_dense)

    percent_mountain = data['percent_mountain']

    mountain_locations = []

    forest_density_array = generate_square_map(size_of_dense_forest=size_of_dense_forest, dim_x=DIM_X,
                                               dim_y=DIM_Y)

    terrain = Terrain(
        dim_x=DIM_X, dim_y=DIM_Y, percent_mountain=percent_mountain, percent_dense=percent_dense,
        forest_color_scale=forest_color_scale,
        forest_density_array=forest_density_array,
        mountain_locations=mountain_locations)

    camera_range = data['camera_range']

    known_hideout_locations = [list(map(int, (x.split(',')))) for x in data['known_hideout_locations']]
    unknown_hideout_locations = [list(map(int, (x.split(',')))) for x in data['unknown_hideout_locations']]

    for percentage in range(10, 100, 10):
        print(percentage/100)
        file_path = f"camera_n_percentage/{percentage}_percent_cameras.txt"
        cameras = produce_camera_distribution(DIM_X, DIM_Y, camera_range, percentage/100, terrain, unknown_hideout_locations)
        write_cameras_to_file(cameras, file_path)

        known_camera_locations = [[DIM_X//4, DIM_X//4*3], [DIM_X//4, DIM_X//4],
                                  [DIM_X//2, DIM_X//2],
                                  [DIM_X//4*3, DIM_X//4*3], [DIM_X//4*3, DIM_X//4]]
        print(f'known_camera_locations: {known_camera_locations}')
        with open(file_path, 'a') as f:
            for camera in known_camera_locations:
                f.write(f'k,{camera[0]},{camera[1]}\n')

    # lines = open("simulator/camera_locations/original.txt", "r").readlines()
    # cameras = [list(map(int, i.strip().split(","))) for i in lines]
    # print(cameras)