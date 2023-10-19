import copy
import math
import os
import random
from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gym import spaces
import sys
sys.path.append('/home/tsaisplus/MuRPE_base/Heterogenous-MARL/harl/envs/ast/')
from Prison_Escape.environment.abstract_object import AbstractObject, DetectionObject
from Prison_Escape.environment.camera import Camera
from Prison_Escape.environment.forest_coverage.generate_square_map import generate_square_map
from Prison_Escape.environment.fugitive import Fugitive
from Prison_Escape.environment.helicopter import Helicopter
from Prison_Escape.environment.hideout import Hideout
from Prison_Escape.environment.observation_spaces import (\
    # create_observation_space_ground_truth, \
    create_observation_space_fugitive, \
    create_observation_space_blue, \
    create_observation_space_blue_team,\
    # create_observation_space_prediction, \
    create_action_space_blue_team, create_action_space_blue_team_v2, \
    )
from Prison_Escape.environment.search_party import SearchParty
from Prison_Escape.environment.terrain import Terrain
from Prison_Escape.environment.utils import create_camera_net


class ObservationType(Enum):
    Fugitive = auto()
    FugitiveGoal = auto()
    Blue = auto()


@dataclass
class RewardScheme:
    time: float = -5e-4
    known_detected: float = -10.
    known_undetected: float = 10.
    unknown_detected: float = -20.
    unknown_undetected: float = 20.
    timeout: float = -30.


presets = RewardScheme.presets = SimpleNamespace()
presets.default = RewardScheme()
presets.none = RewardScheme(0., 0., 0., 0., 0., 0.)
presets.any_hideout = RewardScheme(known_detected=1., known_undetected=1., unknown_detected=1., unknown_undetected=1.,
                                   timeout=-1.)
presets.time_only = copy.copy(presets.none)
presets.time_only.time = presets.default.time
presets.timeout_only = copy.copy(presets.none)
presets.timeout_only.timeout = -3.
del presets  # access as RewardScheme.presets


class PrisonerBothEnv(gym.Env):
    """
    PrisonerEnv simulates the prisoner behavior in a grid world.
    The considered factors include
        - prisoner
        - terrain (woods)
        - hideouts (targets)
        - max-time (72 hours)
        - cameras
        - helicopters
        - search parties

    *Detection is encoded by a three tuple [b, x, y] where b in binary.
    If b=1 (detected), [x, y] will have the detected location in world coordinates.
    If b=0 (not detected), [x, y] will be [-1, -1].

    State space
        - Terrain
        - Time
        - Locations of [cameras, helicopters, helicopter dropped cameras, hideouts, search parties, fugitive]
        - Detection of the fugitive from [cameras, helicopters, helicopter dropped cameras, search parties]
        - Fugitive's detection of [helicopters, helicopter dropped cameras, search parties]

    Observation space (evader)
        - Time
        - Locations of [known cameras, hideouts]
        - Self location, speed, heading
        - Detection of [helicopters, helicopter dropped cameras, search parties]

    Action space (evader)
        - 2 dimensional: speed [1,15] x direction [-pi, pi]
        - 2 dimensional: speed [1,15] x direction [-pi, pi]

    Observation space (pursuer team)
        - Time
        - Locations of [cameras, helicopters, helicopter dropped cameras, search parties, known hideouts]
        - Detection of the fugitive from [cameras, helicopters, helicopter dropped cameras, search parties]
        - Terrain

    Action space (pursuer team)
        - 2 dimensional discrete:
            [1, 20] x [1, 20] for  search party
            [1, 127] x [1, 127] for helicopter
            #TODO: same for now

    Coordinate system:
        - By default, cartesian Coordinate:
        y
        ^
        |
        |
        |
        |
        |----------->x

    Limitations:
        - Food and towns are not implemented yet
        - Sprint time maximum is not implemented yet. However, sprinting does still have drawbacks (easier to be detected)
        - No rain/fog
        - Fixed terrain, fixed hideout locations
        - Detection does not utilize an error ellipse. However, detection still has the range-based PoD.
        - Helicopter dropped cameras are not implemented yet.

    Details:
        - Good guys mean the side of search parties (opposite of the fugitive)
        - Each timestep represents 1 min, and the episode horizon is T=4320 (72 hours)
        - Each grid represents 21 meters
        - Our grid size is 2428x2428 as of now, representing a 50.988 kms x 50.988 kms field
        - Change to NxN grid
        - We have continuous speed profile from 1 grid/timestep to 15 grids/timestep (1.26km/h to 18.9km/h)
        - Right now we have by default:
            - 2 search parties
            - 1 helicopter
            - 5 known hideouts
            - 5 unknown hideouts
            - 5 known cameras, 5 unknown cameras,
            - with another 5 known cameras on known hideouts
            (to encode the fact that we can always detect the fugitive when evader goes to known hideouts)
    """

    def __init__(self,
                 # terrain
                 terrain=None,
                 terrain_x=None,
                 terrain_y=None,
                 terrain_map=None,
                 percent_dense=None,
                 percent_mountain=None,
                 mountain_locations=None,

                 # camera
                 random_cameras=False,
                 num_random_unknown_cameras=None,
                 num_random_known_cameras=None,
                 camera_file_path="/home/tsaisplus/MuRPE_base/Opponent-Modeling-Env/Prison_Escape/environment/camera_locations/camera_n_percentage/10_percent_cameras.txt",
                 camera_net_bool=False,
                 camera_net_path=None,

                 # pursuers team
                 num_search_parties=2,
                 num_helicopters=1,
                 search_party_speed=6.5,
                 helicopter_speed=127,
                 helicopter_battery_life=360,
                 helicopter_recharge_time=360,
                 helicopter_init_pos=None,
                 search_party_init_pos=None,
                 random_init_positions=False,

                 # evader team
                 spawn_mode='normal',
                 spawn_range=15.,
                 min_distance_from_hideout_to_start=1000,
                 prisoner_init_pos=None,

                 # hideout
                 num_known_hideouts=1,
                 num_unknown_hideouts=2,
                 hideout_radius=50.,
                 random_hideout_locations=False,
                 known_hideout_locations=None,
                 unknown_hideout_locations=None,

                 # observation
                 store_last_k_fugitive_detections=False,
                 observation_terrain_feature=True,
                 include_camera_at_start=False,
                 include_start_location_blue_obs=False,

                 # others
                 step_reset=True,  # TODO: this is important
                 detection_factor=None, #FIX: what is this for. this is read into DetectionObject

                 # 在yaml文件中没有定义的参数
                 max_timesteps=4320,  # 72 hours episode长度
                 reward_scheme=None,  # 奖励函数
                 debug=True,

                 # 关于detection的参数
                 camera_range_factor=1.0,  # camera在detect evader时的范围multiplier
                 search_party_range_factor=0.8,  # search party在detect evader时的范围multiplier
                 helicopter_range_factor=0.6,  # helicopter在detect evader时的范围multiplier


                 ):
        """
        PrisonerEnv simulates the prisoner behavior in a grid world.
        :param terrain: If given, the terrain is used from this object
        # :param terrain_map_file: This is the file that contains the terrain map, only used if terrain is None
        #     If none, the default map is used.
        #     Currently all the maps are stored in "/star-data/prisoner-maps/"
        #         We load in the map from .npy file, we use csv_generator.py to convert .nc to .npy
        #     If directory, cycle through all the files upon reset
        #     If single .npy file, use that file
        :param num_towns:
        :param num_search_parties:
        :param num_helicopters:
        :param random_hideout_locations: If True, hideouts are placed randomly with num_known_hideouts and num_unknown_hideouts
            If False, hideouts are selected from known_hideout_locations and unknown_hideout_locations based on the num_known_hideouts and num_unknown_hideouts
        :param num_known_hideouts: number of hideouts known to good guys
        :param num_unknown_hideouts: hideouts unknown to the good guys
        :param: known_hideout_locations: locations of known hideouts when random_hideout_locations=False
        :param: unknown_hideout_locations: locations of unknown hideouts when random_hideout_locations=False
        :param helicopter_battery_life: how many minutes the helicopter can last in the game
        :param helicopter_recharge_time: how many minutes the helicopter need to recharge itself
        :param spawn_mode: how the prisoner location is initialized on reset. Can be:
            'normal': the prisoner is spawned in the northeast corner
            'uniform': the prisoner spawns at a uniformly sampled random location
            'uniform_hideout_dist': spawn the prisoner at min_distance_from_hideout_to_start from the hideouts
                        This assumes the hideouts are chosen first
            'hideout': the prisoner spawns within `spawn_range` of the hideout
        :param spawn_range: how far from the edge of the hideout the prisoner spawns in 'hideout' mode, or
        how far from the corner the prisoner spawn in 'corner' mode
        :param max_timesteps: time horizon for each rollout. Default is 4320 (minutes = 72 hours)
        :param hideout_radius: minimum distance from a hideout to be considered "on" the hideout
        :param reward_scheme: a RewardScheme object definining reward scales for different events. If omitted, a default will be used.
        A custom one can be constructed. Several presets are available under RewardScheme.presets.
        :param known_hideout_locations: list of tuples of known hideout locations
        :param unknown_hideout_locations: list of tuples of unknown hideout locations
        :param random_cameras: boolean of whether to use random camera placements or fixed camera placements
        :param num_random_unknown_cameras: number of random unknown cameras
        :param num_random_known_cameras: number of random known cameras
        :param camera_file_path: path to the file containing the camera locations for the unknown cameras. This it for us to test.txt the Filtering algorithm
        :param camera_net_bool: boolean of whether to use the camera net around the fugitive or not
        :param camera_net_path: if None, place camera net by generating, if path, use the path
        :observation_step_type: What observation is returned in the "step" and "reset" functions
            'Fugitive': Returns fugitive observations
            'Blue': Returns observations from the BlueTeam (aka blue team's vision of the fugitive)
            'GroundTruth': Returns information of all agents in the environment
            'Prediction': Returns fugitive observations but without the unknown hideouts
        :observation_terrain_feature: boolean of whether to include the terrain feature in the observation
        :stopping_condition: boolean of whether to stop the game when the fugitive produces 0 speed
        #TODO: when will evader produce zero speed
        :step_reset: boolean of whether to reset the game after the episode is over or just wait at the final location no matter what action is given to it
            This is to make the multi-step prediction rollouts to work properly.
            Default is True 
        :param include_start_location_blue_obs: boolean of whether to include the start location of the prisoner in the blue team observation
            Default is True
        :param store_last_k_fugitive_detections: Whether or not to store the last k(=8) detections of the fugitive

        """

        # 定义Terrain相关
        self.terrain_list = []
        forest_color_scale = 1

        # 如果没有提供terrain，就从terrain_map中读取
        if terrain is None:
            if terrain_map is None:
                # use original map with size 2428x2428 (NxN)
                dim_x = terrain_x
                dim_y = terrain_y
                size_of_dense_forest = int(dim_x * percent_dense)
                forest_density_array = generate_square_map(size_of_dense_forest=size_of_dense_forest, dim_x=dim_x,
                                                           dim_y=dim_y)

                forest_density_list = [forest_density_array]
                # make terrain a list
                self.terrain_list = [Terrain(dim_x=dim_x, dim_y=dim_y,
                                             percent_mountain=0, percent_dense=percent_dense,
                                             forest_color_scale=forest_color_scale,
                                             forest_density_array=forest_density_array,
                                             mountain_locations=mountain_locations)]

            else:
                # if directory, cycle through all the files
                if os.path.isdir(terrain_map):
                    forest_density_list = []
                    for f in os.listdir(terrain_map):
                        if f.endswith(".npy"):
                            forest_density_array = np.load(os.path.join(terrain_map, f))
                            forest_density_list.append(forest_density_array)
                            dim_x, dim_y = forest_density_array.shape
                            self.terrain_list.append(
                                Terrain(dim_x=dim_x, dim_y=dim_y,
                                        percent_mountain=percent_mountain, percent_dense=percent_dense,
                                        forest_color_scale=forest_color_scale,
                                        forest_density_array=forest_density_array,
                                        mountain_locations=mountain_locations))
                    raise NotImplementedError("terrain_map should be null")

                else:
                    forest_density_array = np.load(terrain_map)
                    forest_density_list = [forest_density_array]
                    dim_x, dim_y = forest_density_array.shape
                    self.terrain_list = [Terrain(dim_x=dim_x, dim_y=dim_y, forest_color_scale=forest_color_scale,
                                                 forest_density_array=forest_density_array,
                                                 mountain_locations=mountain_locations)]
                    raise NotImplementedError("terrain_map should be null")

        else:
            # Getting terrain from terrain object
            # Assume we are just using a single terrain object
            forest_density_list = [terrain.forest_density_array]
            self.terrain_list = [terrain]
            raise NotImplementedError("Terrain should be null and terrain_map should be null")

        # 保存terrain底层图片
        self._cached_terrain_images = [terrain.visualize(just_matrix=True) for terrain in self.terrain_list]

        # 是否保存terrain的embedding
        if observation_terrain_feature:
            raise NotImplementedError("Terrain feature is not implemented yet")
            # # we save these to add to the observations
            # model = ConvAutoencoder()
            # model.load_state_dict(torch.load('Prison_Escape/environment/forest_coverage/autoencoder_state_dict.pt'))
            # self._cached_terrain_embeddings = [produce_terrain_embedding(model, terrain_np) for terrain_np in
            #                                    forest_density_list]
            # terrain_embedding_size = self._cached_terrain_embeddings[0].shape[0]
        else:
            # empty list
            self._cached_terrain_embeddings = [np.array([])] * len(forest_density_list)
            self.terrain_embedding_size = 0

        # 初始化terrain
        self.set_terrain_paramaters()

        # 保存terrain的维度
        self.dim_x = self.terrain.dim_x
        self.dim_y = self.terrain.dim_y


        # 定义camera相关
        self.random_cameras = random_cameras
        self.camera_file_path = camera_file_path
        self.camera_range_factor = camera_range_factor
        self.camera_net_bool = camera_net_bool
        self.camera_net_path = camera_net_path

        # 从yaml文件中读取camera位置
        if self.random_cameras:
            self.num_random_unknown_cameras = num_random_unknown_cameras
            self.num_random_known_cameras = num_random_known_cameras
            raise NotImplementedError("No random cameras yet")
        else:
            self.camera_file_path = camera_file_path
            self.known_camera_locations, self.unknown_camera_locations = self.read_camera_file(camera_file_path)

        # 是否在evader的初始位置放置camera
        self.include_camera_at_start = include_camera_at_start


        # 定义pursers team相关
        self.num_search_parties = num_search_parties
        self.num_helicopters = num_helicopters
        self.search_party_speed = search_party_speed
        self.helicopter_speed = helicopter_speed
        self.helicopter_battery_life = helicopter_battery_life
        self.helicopter_recharge_time = helicopter_recharge_time
        self.helicopter_range_factor = helicopter_range_factor
        self.random_init_positions = random_init_positions
        self.helicopter_init_pos = helicopter_init_pos
        self.search_party_init_pos = search_party_init_pos
        self.search_party_range_factor = search_party_range_factor



        # 定义evader team相关
        self.spawn_mode = spawn_mode
        self.spawn_range = spawn_range
        self.prisoner_init_pos = prisoner_init_pos
        self.min_distance_from_hideout_to_start = min_distance_from_hideout_to_start

        # 初始化evader速度, 出现在render里面计算detection范围，在step both函数里有实时更新
        self.current_prisoner_speed = 0


        # 定义observation相关
        self.include_start_location_blue_obs = include_start_location_blue_obs
        self.store_last_k_fugitive_detections = store_last_k_fugitive_detections
        # Store (t,x,y) for last k detections. Only updated if self.store_last_k_fugitive_detections is True
        self.last_k_fugitive_detections = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                           [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]


        # 定义hideout相关
        self.num_known_hideouts = num_known_hideouts
        self.num_unknown_hideouts = num_unknown_hideouts
        self.hideout_radius = hideout_radius
        self.random_hideout_locations = random_hideout_locations
        self.known_hideout_locations = known_hideout_locations
        self.unknown_hideout_locations = unknown_hideout_locations


        # 定义其他相关
        self.step_reset = step_reset  # TODO： 这是什么
        self.detection_factor = 400.0  #FIXME: testing a rational number

        DetectionObject.detection_factor = detection_factor


        # 在yaml文件中没有定义的参数
        self.max_timesteps = max_timesteps  # 72 hours = 4320 minutes
        # reward_scheme相关
        self.reward_scheme = reward_scheme or RewardScheme()  # accept a custom or use the default
        assert isinstance(self.reward_scheme, (type(None), str, RewardScheme))
        if isinstance(self.reward_scheme, str):
            self.reward_scheme = getattr(RewardScheme.presets, self.reward_scheme)
        self.DEBUG = debug


        # 初始化observation空间
        self._fugitive_observation = None
        self._blue_observation = None
        self._blue_team_state = None

        # 智能体数量
        self.num_agents = self.num_search_parties + self.num_helicopters

        # 初始化时间步
        self.timesteps = 0

        # episode done标志
        self.done = False
        self.is_detected = False
        self.last_detected_timestep = 0
        # TODO self.surrounded_by_pursuers_time = 5 - 创建一个first in first out的deque，长度为self.surrounded_by_pursuers_time

        # 初始化search parties, helicopters, cameras, fugitive等一系列objects
        self.set_up_world()

        # 创建状态空间&动作空间相关
        self.create_spaces()
        
        # 加载图片资源
        self.load_image_assets()

    def set_terrain_paramaters(self):
        """ Sets self.terrain_embedding, self.terrain, and self._cached_terrain_image"""
        # Choose a random terrain from the list
        terrain_index = random.randint(0, len(self.terrain_list) - 1)
        self.terrain = self.terrain_list[terrain_index]
        self._cached_terrain_image = self._cached_terrain_images[terrain_index]
        self._terrain_embedding = self._cached_terrain_embeddings[terrain_index]

    def set_up_world(self):
        """
        This function places all the objects,
        Right now,
            - cameras are initialized randomly
            - helicopter is initialized randomly
            - hideouts are initialized always at [20, 80], [100, 20]
            - search parties are initialized randomly or not
            - prisoner is initialized by different self.spawn_mode
        """
        self.camera_list = []
        self.helicopters_list = []
        self.hideout_list = []
        self.search_parties_list = []
        self.town_list = []
        self.hideout_list = []
        self.min_distance_between_hideouts = 300  # FIXED

        # randomized
        if not self.random_hideout_locations:
            self.place_fixed_hideouts()
        else:
            assert self.random_hideout_locations is False, "Random hideout locations have not been implemented"
            raise NotImplementedError
            # Random hideouts have not been implemented with spawn mode as uniform hideout dist
            # Random hideouts need to have prisoner location initialized first
            # self.place_random_hideouts()

        if self.spawn_mode == 'normal':
            prisoner_location = self.prisoner_init_pos
        elif self.spawn_mode == 'uniform':
            # in_mountain = True
            mountain_range = 150
            near_mountain = 0
            while near_mountain < mountain_range:
                # We do not want to place the fugitive on a mountain!
                prisoner_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                # We do not want to place the fugitive within a distance of the mountain!
                m_dists = np.array([np.linalg.norm(np.array(prisoner_location) - np.array([m[1], m[0]])) for m in
                                    self.terrain.mountain_locations])
                near_mountain = min(m_dists)
            raise NotImplementedError("Uniform spawn mode has not been tested")
        elif self.spawn_mode == 'uniform_hideout_dist':
            # Spawn uniformly on the map but with a distance of at least min_distance_from_hideout_to_start
            mountain_range = 150
            near_mountain = 0
            min_dist = 0
            while near_mountain < mountain_range or min_dist < self.min_distance_from_hideout_to_start:
                prisoner_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                s = [tuple(i.location) for i in self.hideout_list]
                dists = np.array(
                    [math.sqrt((prisoner_location[0] - s0) ** 2 + (prisoner_location[1] - s1) ** 2) for s0, s1 in s])
                min_dist = min(dists)

                # We do not want to place the fugitive within a distance of the mountain!
                m_dists = np.array([np.linalg.norm(np.array(prisoner_location) - np.array([m[1], m[0]])) for m in
                                    self.terrain.mountain_locations])
                near_mountain = min(m_dists)
            raise NotImplementedError("uniform_hideout_dist spawn mode has not been tested")

        elif self.spawn_mode == 'hideout':
            in_mountain = True
            in_map = False
            while in_mountain or not in_map:
                # We do not want to place the fugitive on a mountain or outside the map!
                hideout_id = np.random.choice(range(len(self.hideout_list)))
                hideout_loc = np.array(self.hideout_list[hideout_id].location)
                angle = np.random.random() * 2 * math.pi - math.pi
                radius = self.hideout_radius + np.random.random() * self.spawn_range
                vector = np.array([math.cos(angle), math.sin(angle)]) * radius
                prisoner_location = (hideout_loc + vector).astype(int).tolist()
                in_mountain = self.terrain.world_representation[0, prisoner_location[0], prisoner_location[1]] == 1
                in_map = prisoner_location[0] in range(0, self.dim_x) and \
                         prisoner_location[1] in range(0, self.dim_y)
            raise NotImplementedError("hideout spawn mode has not been tested")

        elif self.spawn_mode == 'corner':
            # generate the fugitive randomly near the top right corner
            prisoner_location = AbstractObject.generate_random_locations_with_range(
                [self.dim_x - self.spawn_range, self.dim_x], [self.dim_y - self.spawn_range, self.dim_y])
            raise NotImplementedError("corner spawn mode has not been tested")

        else:
            raise ValueError('Unknown spawn mode "%s"' % self.spawn_mode)

        # 初始化evader
        self.prisoner = Fugitive(self.terrain, prisoner_location)
        self.prisoner_start_location = prisoner_location

        # specify cameras' initial locations
        if (self.random_cameras):
            # randomized 
            known_camera_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y) for _ in
                                      range(self.num_random_known_cameras)]
            unknown_camera_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y) for _ in
                                        range(self.num_random_unknown_cameras)]
        else:
            known_camera_locations = self.known_camera_locations[:]
            unknown_camera_locations = copy.deepcopy(self.unknown_camera_locations)

        if self.camera_net_bool:
            if self.camera_net_path is None:
                cam_locs = create_camera_net(prisoner_location, dist_x=360, dist_y=360, spacing=30,
                                             include_camera_at_start=self.include_camera_at_start)
                unknown_camera_locations.extend(cam_locs.tolist())
            else:
                known_net, unknown_net = self.read_camera_file(self.camera_net_path)
                known_camera_locations.extend(known_net)
                unknown_camera_locations.extend(unknown_net)
        elif self.include_camera_at_start:
            known_camera_locations.append(prisoner_location)

        # append cameras at known hideouts
        for i in self.hideout_list:
            if i.known_to_good_guys:
                known_camera_locations.append(i.location)

        # 初始化camera
        # initialize these variables for observation spaces
        self.num_known_cameras = len(known_camera_locations)  # known cameras + known hideouts + camera_at_start
        self.num_unknown_cameras = len(unknown_camera_locations)
        for counter in range(self.num_known_cameras):
            camera_location = known_camera_locations[counter]
            self.camera_list.append(Camera(self.terrain, camera_location, known_to_fugitive=True,
                                           detection_object_type_coefficient=self.camera_range_factor))
        for counter in range(self.num_unknown_cameras):
            camera_location = unknown_camera_locations[counter]
            self.camera_list.append(Camera(self.terrain, camera_location, known_to_fugitive=False,
                                           detection_object_type_coefficient=self.camera_range_factor))

        # 初始化helicopter
        for _ in range(self.num_helicopters):
            if self.random_init_positions:
                # generate random helicopter locations
                helicopter_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
            else:
                # generate specified helicopter locations
                helicopter_location = self.helicopter_init_pos

            self.helicopters_list.append(
                Helicopter(self.terrain, helicopter_location, speed=self.helicopter_speed,
                           detection_object_type_coefficient=self.helicopter_range_factor))


        # 初始化search party
        if self.random_init_positions:
            search_party_initial_locations = []
            for _ in range(self.num_search_parties):
                search_party_initial_locations.append(AbstractObject.generate_random_locations(self.dim_x, self.dim_y))
        else:
            search_party_initial_locations = self.search_party_init_pos
        for counter in range(self.num_search_parties):
            search_party_location = search_party_initial_locations[
                counter]  # AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
            self.search_parties_list.append(
                SearchParty(self.terrain, search_party_location, speed=self.search_party_speed,
                            detection_object_type_coefficient=self.search_party_range_factor))


    def create_spaces(self):

        self.blue_observation_space, self.blue_obs_names = create_observation_space_blue(
            num_known_cameras=self.num_known_cameras,
            num_unknown_cameras=self.num_unknown_cameras,
            num_known_hideouts=self.num_known_hideouts,
            num_helicopters=self.num_helicopters,
            num_search_parties=self.num_search_parties,
            terrain_size=self.terrain_embedding_size,
            include_start_location_blue_obs=self.include_start_location_blue_obs)

        self.fugitive_observation_space, self.fugitive_obs_names = create_observation_space_fugitive(
            num_known_cameras=self.num_known_cameras,
            num_known_hideouts=self.num_known_hideouts,
            num_unknown_hideouts=self.num_unknown_hideouts,
            num_helicopters=self.num_helicopters,
            num_search_parties=self.num_search_parties,
            terrain_size=self.terrain_embedding_size)

        self.blue_team_observation_space = create_observation_space_blue_team(self.blue_observation_space, self.num_agents)
        
        # evader动作空间
        self.red_action_space = spaces.Box(low=np.array([0, -np.pi]), high=np.array([15, np.pi]))

        # pursuer团队动作空间
        self.blue_team_action_space = create_action_space_blue_team_v2(num_helicopters=self.num_helicopters,
                                                                  num_search_parties=self.num_search_parties)




    def load_image_assets(self):
        base_path = "/home/tsaisplus/MuRPE_base/Opponent-Modeling-Env/Prison_Escape"

        # Construct absolute paths using os.path.join
        self.known_camera_pic = Image.open(os.path.join(base_path, "environment/assets/camera_blue.png"))
        self.unknown_camera_pic = Image.open(os.path.join(base_path, "environment/assets/camera_red.png"))
        self.known_hideout_pic = Image.open(os.path.join(base_path, "environment/assets/star.png"))
        self.unknown_hideout_pic = Image.open(os.path.join(base_path, "environment/assets/star_blue.png"))
        self.town_pic = Image.open(os.path.join(base_path, "environment/assets/town.png"))
        self.search_party_pic = Image.open(os.path.join(base_path, "environment/assets/searching.png"))
        self.helicopter_pic = Image.open(os.path.join(base_path, "environment/assets/helicopter.png"))
        self.prisoner_pic = Image.open(os.path.join(base_path, "environment/assets/prisoner.png"))
        self.detected_prisoner_pic = Image.open(os.path.join(base_path, "environment/assets/detected_prisoner.png"))

        self.known_camera_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/camera_blue.png"),
                                              cv2.IMREAD_UNCHANGED)
        self.unknown_camera_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/camera_red.png"),
                                                cv2.IMREAD_UNCHANGED)
        self.known_hideout_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/star.png"),
                                               cv2.IMREAD_UNCHANGED)
        self.unknown_hideout_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/star_blue.png"),
                                                 cv2.IMREAD_UNCHANGED)
        self.town_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/town.png"), cv2.IMREAD_UNCHANGED)
        self.search_party_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/searching.png"),
                                              cv2.IMREAD_UNCHANGED)
        self.helicopter_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/helicopter.png"),
                                            cv2.IMREAD_UNCHANGED)
        self.helicopter_no_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/helicopter_no.png"),
                                               cv2.IMREAD_UNCHANGED)
        self.prisoner_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/prisoner.png"),
                                          cv2.IMREAD_UNCHANGED)
        self.detected_prisoner_pic_cv = cv2.imread(os.path.join(base_path, "environment/assets/detected_prisoner.png"),
                                                   cv2.IMREAD_UNCHANGED)

        # FIXED. How big is the assets image in render
        self.default_asset_size = 70


    def reset(self, seed=None):
        """
        Reset the environment. Should be called whenever done==True
        :return: 不返回任何东西，
        evader的observation不return，而是存在/更新self._fugitive_observation
        pursuer的observation不return，而是存在/更新self._blue_team_state和self._blue_observation里面
        """
        if seed is not None:
            self.set_seed(seed)

        # 初始化terrain
        self.set_terrain_paramaters()

        # 初始化evader
        self.prisoner = Fugitive(self.terrain, self.prisoner_init_pos)
        self.prisoner_location_history = [self.prisoner.location.copy()]

        # episode done标志
        self.timesteps = 0
        self.last_detected_timestep = 0
        self.done = False

        # 初始化search parties, helicopters, cameras, fugitive等一系列objects
        self.set_up_world()

        # 仅在reset里出现，双方detect对方
        fugitive_detection_of_parties, parties_detection_of_fugitive = self._determine_detection_reset()


        # 建立evader的observation. 当前evader的速度=0，0
        self._fugitive_observation = self._construct_fugitive_observation([0.0, 0.0], fugitive_detection_of_parties,
                                                                          self._terrain_embedding).astype(np.float32)

        # 建立pursuer的observation
        parties_detection_of_fugitive = self.transform_blue_detection_of_fugitive(parties_detection_of_fugitive)

        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive,
                                                                  self._terrain_embedding,
                                                                  self.include_start_location_blue_obs).astype(np.float32)
        self._blue_team_state = self._construct_blue_team_state().astype(np.float32)

        # 检查blue obs/fugitive obs/blue team的shape/dtype是否和创立空间的shape/dtype一致
        self.check_space()


    def set_seed(self, seed):
        """
        Set the seed for the random number generator.
        :param seed: the seed for the random number generator
        """
        np.random.seed(seed)
        random.seed(seed)


    def _determine_detection_reset(self):
        """
            仅在reset里出现，evader速度=0，双方detect对方
            输入：evader初始速度=0
            This function determines both sides.
            Evader detect pursuers && pursuers detect evader INITIALLY SPEED IS 0!

            输出： a list of [b, x, y] for each detection object.
                fugitive_detection_of_parties 的list长度是（num_helicopters + num_search_parties）* 3
                parties_detection_of_fugitive 的list长度是（num_helicopters + num_search_parties + num_camera）* 3
        """
        fugitive_detection_of_parties = []
        # SPRINT_SPEED_THRESHOLD = 8

        # Evader detect pursuers
        for helicopter in self.helicopters_list:
            # if speed > SPRINT_SPEED_THRESHOLD:
            #     # 当evader在sprint的时候，evader不能detect到任何东西
            #     # when sprinting the prisoner cannot detect anything
            #     fugitive_detection_of_parties.extend([0, -1, -1])
            #     raise NotImplementedError("Evader speed at reset should be 0!")
            # else:

            # detect_bool - [b, x, y]的格式
            # _ - 0% detection_range
            detect_bool, _ = self.prisoner.detect(helicopter.location, helicopter)
            fugitive_detection_of_parties.extend(detect_bool)
        if self.DEBUG:
            print("Evader detect Helicopter 100% detection_range: ", _/3)
            print("Evader detect Helicopter 0% detection_range: ", _)

        for search_party in self.search_parties_list:
            # if speed > SPRINT_SPEED_THRESHOLD:
            #     # when sprinting the prisoner cannot detect anything
            #     fugitive_detection_of_parties.extend([0, -1, -1])
            # else:
            detect_bool, _ = self.prisoner.detect(search_party.location, search_party)
            fugitive_detection_of_parties.extend(detect_bool)
        if self.DEBUG:
            print("Evader detect search_party 100% detection_range: ", _/3)
            print("Evader detect search_party 0% detection_range: ", _)

        # for camera in self.known_camera_locations: #TODO： evader为什么没有检测到camera的能力？
        #     detect_bool, _ = self.prisoner.detect(camera, self.camera_range_factor)
        #     fugitive_detection_of_parties.extend(detect_bool)

        # Pursuers detect evader
        parties_detection_of_fugitive = []
        for camera in self.camera_list:
            detect_bool, _ = camera.detect(self.prisoner.location)
            parties_detection_of_fugitive.extend(detect_bool)
        if self.DEBUG:
            print("Camera detect evader 100% detection_range: ", _/3)
            print("Camera detect evader 0% detection_range: ", _)
        for helicopter in self.helicopters_list:
            detect_bool, _ = helicopter.detect(self.prisoner.location)
            parties_detection_of_fugitive.extend(detect_bool)
        if self.DEBUG:
            print("Helicopter detect evader 100% detection_range: ", _/3)
            print("Helicopter detect evader 0% detection_range: ", _)
        for search_party in self.search_parties_list:
            detect_bool, _ = search_party.detect(self.prisoner.location)
            parties_detection_of_fugitive.extend(detect_bool)
        if self.DEBUG:
            print("Search_party detect evader 100% detection_range: ", _/3)
            print("Search_party detect evader 0% detection_range: ", _)

        if any(parties_detection_of_fugitive[::3]):
            self.last_detected_timestep = self.timesteps

        return fugitive_detection_of_parties, parties_detection_of_fugitive

    def _construct_fugitive_observation(self, action, fugitive_detection_of_parties, terrain):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param action: the action taken by the fugitive at this timestep
        :param fugitive_detection_of_parties: a list encoding fugitive's detection of all parties
        :param terrain: the terrain feature of the environment
        :return: the observation from the prisoner perspective
        """

        # NOTE: When editing, make sure this corresponds to names/orders in obs_names, in constructor
        # Future: Assign these using obs_names instead? may be slower...
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                observation.append(camera.location[0] / self.dim_x)
                observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            observation.append(float(hideout.known_to_good_guys))
            observation.append(hideout.location[0] / self.dim_x)
            observation.append(hideout.location[1] / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        observation.extend(fugitive_detection_of_parties)
        observation = np.array(observation)
        observation = np.concatenate((observation, terrain))

        return observation

    def transform_blue_detection_of_fugitive(self, parties_detection_of_fugitive):
        """ This is used to remove the repeated detections of the fugitive in the blue parties observation space.
        清理冗余的evader detection

            args: parties_detection_of_fugitive is a list of [b, x, y] where b is binary and [x, y] is the location of the detected fugitive for each agent
            returns: a list showing just if the prisoner was detected for each agent and the location of the agent if detected
        """
        one_hot = parties_detection_of_fugitive[::3]
        if not any(one_hot):
            # 所有的pursuer都没有detect到evader
            return one_hot + [-1, -1]
        else:
            # 有pursuer detect到evader
            index = one_hot.index(1)  # locate which agent detected the fugitive
            # 返回one hot和detect到的evader坐标
            return one_hot + [i for i in parties_detection_of_fugitive[index * 3 + 1:index * 3 + 3]]

    def _construct_blue_observation(self, parties_detection_of_fugitive, terrain,
                                    include_start_location_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective

        including the normalization
        """

        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            observation.append(camera.location[0] / self.dim_x)
            observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                observation.append(hideout.location[0] / self.dim_x)
                observation.append(hideout.location[1] / self.dim_y)
        for helicopter in self.helicopters_list:
            observation.append(helicopter.location[0] / self.dim_x)
            observation.append(helicopter.location[1] / self.dim_y)
        for search_party in self.search_parties_list:
            observation.append(search_party.location[0] / self.dim_x)
            observation.append(search_party.location[1] / self.dim_y)

        observation.extend(parties_detection_of_fugitive)

        if include_start_location_blue_obs:
            observation.append(self.prisoner_start_location[0] / self.dim_x)
            observation.append(self.prisoner_start_location[1] / self.dim_y)

        observation = np.array(observation)
        observation = np.concatenate((observation, terrain))

        return observation

    def _construct_blue_team_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.get_blue_observation() for i in range(self.num_agents)])

        return state

    def check_space(self):

        # 检查blue observation的shape是否和创立空间的shape一致
        assert self._blue_observation.shape == self.blue_observation_space.shape, "Wrong observation shape %s, %s" % (
            self._blue_observation.shape, self.blue_observation_space.shape)
        assert self._fugitive_observation.shape == self.fugitive_observation_space.shape, "Wrong observation shape %s, %s" % (
            self._fugitive_observation.shape, self.fugitive_observation_space.shape)
        assert self._blue_team_state.shape == self.blue_team_observation_space.shape, "Wrong observation shape %s, %s" % (
            self._blue_team_state.shape, self.blue_team_observation_space.shape)

        # 检查blue observation的dtype是否和创立空间的dtype一致
        assert self._blue_observation.dtype == self.blue_observation_space.dtype, "Wrong observation dtype %s, %s" % (
            self._blue_observation.dtype, self.blue_observation_space.dtype)
        assert self._fugitive_observation.dtype == self.fugitive_observation_space.dtype, "Wrong observation dtype %s, %s" % (
            self._fugitive_observation.dtype, self.fugitive_observation_space.dtype)
        assert self._blue_team_state.dtype == self.blue_team_observation_space.dtype, "Wrong observation dtype %s, %s" % (
            self._blue_team_state.dtype, self.blue_team_observation_space.dtype)


    def get_fugitive_observation(self):
        return self._fugitive_observation


    def step_both(self, red_action: np.ndarray, blue_action: np.ndarray):
        """
        The environment moves one timestep forward with the action chosen by the agent.
        :param red_action: an speed and direction vector for the red agent
        :param blue_action: waypoint


        :return: observation, reward, done (boolean), info (dict)
        """
        if self.done:
            if self.step_reset:
                raise RuntimeError("Episode is done")
            else:
                observation = np.zeros(self.blue_observation_space.shape)
                share_observation = np.zeros(self.blue_team_observation_space.shape)
                total_reward = 0

                return None, observation, share_observation, total_reward, self.done, {}

        assert self.red_action_space.contains(
            red_action), f"Red Actions should be in the action space, but got {red_action}"

        # 时间步+1
        self.timesteps += 1

        # evader之前的位置
        old_prisoner_location = self.prisoner.location.copy()

        # 移动evader
        direction = np.array([np.cos(red_action[1]), np.sin(red_action[1])])

        fugitive_speed = red_action[0]
        self.current_prisoner_speed = fugitive_speed

        prisoner_location = np.array(self.prisoner.location, dtype=np.float)
        new_location = np.round(prisoner_location + direction * fugitive_speed)
        new_location[0] = np.clip(new_location[0], 0, self.dim_x - 1)
        new_location[1] = np.clip(new_location[1], 0, self.dim_y - 1)
        new_location = new_location.astype(np.int)

        # bump back from mountain
        if self.terrain.world_representation[0, new_location[0], new_location[1]] == 1:
            new_location = np.array(old_prisoner_location)

        self.prisoner.location = new_location.tolist()
        self.prisoner_location_history.append(self.prisoner.location.copy())
        if self.DEBUG:
            print("-----------------")
            print("finish moving the prisoner")
            print("Prisoner detetion range: ", self.prisoner.detection_range)
            print("Prisoner location: ", self.prisoner.location)
            print("Prisoner speed: ", self.current_prisoner_speed)

        original_velocity_version = False  # FIXED

        if original_velocity_version:
            # 移动pursuers
            # UGVs. 提取速度和方向
            for i, search_party in enumerate(self.search_parties_list):
                # getattr(search_party, command)(*args, **kwargs)
                direction = blue_action[i][0:2]
                speed = blue_action[i][2]
                #
                search_party.path_v3(direction=direction, speed=speed)

            if self.is_helicopter_operating():
                for j, helicopter in enumerate(self.helicopters_list):
                    # getattr(helicopter, command)(*args, **kwargs)
                    direction = blue_action[i + j + 1][0:2]
                    speed = blue_action[i + j + 1][2]
                    helicopter.path_v3(direction=direction, speed=speed)
        else:
            # 移动pursuers
            # UGVs. 提取waypoint
            for i, search_party in enumerate(self.search_parties_list):
                search_party.path_v4(action=blue_action[i], speed=self.search_party_speed)

            if self.is_helicopter_operating():
                for j, helicopter in enumerate(self.helicopters_list):
                    helicopter.path_v4(action=blue_action[i + j + 1], speed=self.helicopter_speed)

        # 一系列判断是否结束的条件
        self.check_done(red_action)

        # 建立observation
        # 给evader的observation
        fugitive_detection_of_parties = self._determine_red_detection_of_blue(fugitive_speed)
        self._fugitive_observation = self._construct_fugitive_observation(red_action, fugitive_detection_of_parties,
                                                                          self._terrain_embedding).astype(np.float32)

        # 给pursuers的observation
        parties_detection_of_fugitive = self._determine_blue_detection_of_red(fugitive_speed)
        parties_detection_of_fugitive_one_hot = self.transform_blue_detection_of_fugitive(parties_detection_of_fugitive)
        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive_one_hot,
                                                                  self._terrain_embedding,
                                                                  self.include_start_location_blue_obs).astype(np.float32)
        self._blue_team_state = self._construct_blue_team_state().astype(np.float32)

        # 计算reward TODO
        self.is_detected = self.is_fugitive_detected(parties_detection_of_fugitive)

        if self.is_detected:
            if self.DEBUG:
                print("Evader is detected")
        if self.is_detected and self.store_last_k_fugitive_detections:
            self.last_k_fugitive_detections.pop(0)  # Remove old detection
            self.last_k_fugitive_detections.append([self.timesteps / self.max_timesteps,
                                                    self.prisoner.location[0] / self.dim_x,
                                                    self.prisoner.location[1] / self.dim_y])  # Append latest detection
        total_reward = self.get_reward()

        return (
            self._fugitive_observation,
            self._blue_observation,
            self._blue_team_state,
            total_reward,
            self.done,
            {'current_time_step': self.timesteps,
             'is_detected': self.is_detected,}
        )

    @property
    def hideout_locations(self):
        return [hideout.location for hideout in self.hideout_list]

    def is_helicopter_operating(self):
        """
        判断UAV是否还在工作
        Determines whether the helicopter is operating right now
        :return: Boolean indicating whether the helicopter is operating
        """
        # # comment - UAV会一直工作
        # timestep = self.timesteps % (self.helicopter_recharge_time + self.helicopter_battery_life)
        # if timestep < self.helicopter_battery_life:
        #     return True
        # else:
        #     return False
        return True

    @property
    def spawn_point(self):
        return self.prisoner_location_history[0].copy()

    @staticmethod
    def is_fugitive_detected(parties_detection_of_fugitive):
        for e, i in enumerate(parties_detection_of_fugitive):
            if e % 3 == 0:
                if i == 1:
                    return True
        return False

    def get_reward(self):
        # TODO recode this so combinations of scenarios are possible per timestep

        # 没完成并超时
        if self.timesteps == self.max_timesteps:
            return self.reward_scheme.timeout  # running out of time is bad!

        # 对hideout的detection reward
        hideout = self.near_hideout()
        if hideout is not None:
            if hideout.known_to_good_guys:
                if self.is_detected:
                    return self.reward_scheme.known_detected
                else:
                    return self.reward_scheme.known_undetected
            else:
                if self.is_detected:
                    return self.reward_scheme.unknown_detected
                else:
                    return self.reward_scheme.unknown_undetected

        # game is not done. Simple sparse timestep reward
        return self.reward_scheme.time

    def check_done(self, red_action):
        """一系列判断是否结束的条件"""

        # stop. evader到达终点
        if self.near_hideout():
            self.done = True
            if self.DEBUG:
                print("Stopping condition - evader is near hideout")

        # stop. evader被包围
        elif self.surrounded_by_pursuers():
            self.done = True
            if self.DEBUG:
                print("Stopping condition - evader is surrounded by pursuers")

        # stop. evader速度小于1
        elif (0 <= red_action[0] < 1):
            self.done = True
            if self.DEBUG:
                print("Stopping condition - evader speed is between 0 and 1")

        # stop. 时间步用完
        elif self.timesteps >= self.max_timesteps:
            self.done = True
            if self.DEBUG:
                print("Stopping condition - max timesteps reached")

        else:
            assert self.done == False, "Stopping condition not met, but done is True"

    def near_hideout(self):
        """If the prisoner is within range of a hideout, return it. Otherwise, return None."""
        for hideout in self.hideout_list:
            if ((np.asarray(hideout.location) - np.asarray(
                    self.prisoner.location)) ** 2).sum() ** .5 <= self.hideout_radius + 1e-6:
                # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
                return hideout
        return None

    def surrounded_by_pursuers(self):
        # TODO: not finished. 没完成并持续一段时间
        """
        判断evader是否被包围
        Determines whether the fugitive is surrounded by pursuers
        :return: Boolean indicating whether the fugitive is surrounded by pursuers
        """

        # 任意search_party和evader的距离和不超过search_party的detection_range，并且持续一段时间
        for search_party in self.search_parties_list:
            if np.linalg.norm(np.array(search_party.location) - np.array(self.prisoner.location)) <= search_party.detection_range:
                return True
        return False

    def _determine_red_detection_of_blue(self, speed):
        fugitive_detection_of_parties = []
        SPRINT_SPEED_THRESHOLD = 8
        for helicopter in self.helicopters_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(helicopter.location, helicopter))
        for search_party in self.search_parties_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(search_party.location, search_party))
        return fugitive_detection_of_parties

    def _determine_blue_detection_of_red(self, speed):
        parties_detection_of_fugitive = []
        for camera in self.camera_list:
            parties_detection_of_fugitive.extend(camera.detect(self.prisoner.location, speed))
        for helicopter in self.helicopters_list:
            parties_detection_of_fugitive.extend(helicopter.detect(self.prisoner.location, speed))
        for search_party in self.search_parties_list:
            parties_detection_of_fugitive.extend(search_party.detect(self.prisoner.location, speed))

        if any(parties_detection_of_fugitive[::3]):
            # print('Evader Detected!')
            self.last_detected_timestep = self.timesteps
        return parties_detection_of_fugitive












    def get_blue_observation(self):
        return self._blue_observation

    def get_blue_team_state(self):
        return self._blue_team_state

    def get_last_k_fugitive_detections(self):
        return self.last_k_fugitive_detections

    @property
    def cached_terrain_image(self):
        """
        cache terrain image to be more efficient when rendering
        :return:
        """
        return self._cached_terrain_image

    def read_camera_file(self, camera_file_path):
        """Generate a lists of camera objects from file

        Args:
            camera_file_path (str): path to camera file

        Raises:
            ValueError: If Camera file does not have a u or k at beginning of each line

        Returns:
            (list, list): Returns known camera locations and unknown camera locations
        """
        unknown_camera_locations = []
        known_camera_locations = []
        camera_file = open(camera_file_path, "r").readlines()
        for line in camera_file:
            line = line.strip().split(",")
            if line[0] == 'u':
                unknown_camera_locations.append([int(line[1]), int(line[2])])
            elif line[0] == 'k':
                known_camera_locations.append([int(line[1]), int(line[2])])
            else:
                raise ValueError(
                    "Camera file format is incorrect, each line must start with 'u' or 'k' to denote unknown or known")
        return known_camera_locations, unknown_camera_locations


    def place_fixed_hideouts(self):
        # specify hideouts' locations. These are passed in from the input args
        # We select a number of hideouts from num_known_hideouts and num_unknown_hideouts

        assert self.num_known_hideouts <= len(
            self.known_hideout_locations), f"Must provide a list of known_hideout_locations ({len(self.known_hideout_locations)}) greater than number of known hideouts {self.num_known_hideouts}"
        assert self.num_unknown_hideouts <= len(
            self.unknown_hideout_locations), f"Must provide a list of known_hideout_locations ({len(self.unknown_hideout_locations)}) greater than number of known hideouts {self.num_unknown_hideouts}"

        known_hideouts = random.sample(self.known_hideout_locations, self.num_known_hideouts)
        unknown_hideouts = random.sample(self.unknown_hideout_locations, self.num_unknown_hideouts)

        self.hideout_list = []
        for hideout_location in known_hideouts:
            self.hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=True))

        for hideout_location in unknown_hideouts:
            self.hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=False))

        pass



    def render(self, mode, show=True, fast=False, scale=3, show_delta=False, show_grid=False):
        """
        Render the environment.
        :param mode: required by `gym.Env` but we ignore it
        :param show: whether to show the rendered image
        :param fast: whether to use the fast version for render. The fast version takes less time to render but the render quality is lower.
        :param scale: scale for fast render
        :param show_delta: is a bool whether or not to display the square around the fugitive
        :return: opencv img object
        """
        if fast:
            return self.fast_render_canvas(show, scale, predicted_prisoner_location=None,
                                           show_delta=show_delta, show_grid=show_grid)
        else:
            # return self.slow_render_canvas(show)
            raise NotImplementedError("Slow render is not tested")

    def fast_render_canvas(self, show=True, scale=3, predicted_prisoner_location=None, show_delta=False,
                           show_grid=False):
        """
        We allow the predicted prisoner location to be passed in which renders a predicted prisoner location
        show_delta: is a bool whether or not to display the square around the fugitive
        """
        # Init the canvas
        self.canvas = self.cached_terrain_image
        self.canvas = cv2.flip(self.canvas, 0)

        def calculate_appropriate_image_extent_cv(loc, radius=0.4):
            y_new = -loc[1] + self.dim_y
            return list(map(int, [max(loc[0] - radius, 0), min(loc[0] + radius, self.dim_x),
                                  max(y_new - radius, 0), min(y_new + radius, self.dim_y)]))

        def draw_radius_of_detection(location, radius):
            radius = int(radius)
            color = (0, 0, 1)  # red detection circle
            location = (int(location[0]), self.dim_y - int(location[1]))
            cv2.circle(self.canvas, location, radius, color, 4)

        def draw_grid_on_canvas_cv(x, y, scale=3):
            # draw grid in black color
            for i in range(0, x, 100 // scale):
                cv2.line(self.canvas, (i, 0), (i, self.dim_y), (0, 0, 0), 1)
            for i in range(0, y, 100 // scale):
                cv2.line(self.canvas, (0, i), (self.dim_x, i), (0, 0, 0), 1)

        def draw_image_on_canvas_cv(image, location, asset_size):

            asset_size = int(asset_size)
            if asset_size % 2 != 0:
                asset_size = asset_size - 1

            x_min, x_max, y_min, y_max = calculate_appropriate_image_extent_cv(location, asset_size)

            img = cv2.resize(image, (x_max - x_min, y_max - y_min))

            # create mask based on alpha channel
            mask = img[:, :, 3]
            mask[mask > 50] = 255
            mask = cv2.bitwise_not(mask)

            # cut out portion of the background where we want to paste image
            cut_background = self.canvas[y_min:y_max, x_min:x_max, :]
            img_with_background = cv2.bitwise_and(cut_background, cut_background, mask=mask) + img[:, :, 0:3] / 255

            # insert new image into background/canvas
            self.canvas[y_min:y_max, x_min:x_max, :] = img_with_background

        # fugitive_speed = prisoner.
        if self.is_detected:
            draw_image_on_canvas_cv(self.detected_prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
        else:
            draw_image_on_canvas_cv(self.prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
        draw_radius_of_detection(self.prisoner.location, self.prisoner.detection_range)

        # draw predicted prisoner location
        if predicted_prisoner_location is not None:
            # flip for canvas
            predicted_prisoner_location[1] = self.dim_y - predicted_prisoner_location[1]
            cv2.circle(self.canvas, predicted_prisoner_location, 20, (0, 0, 1), -1)

        # search parties
        for search_party in self.search_parties_list:
            draw_image_on_canvas_cv(self.search_party_pic_cv, search_party.location, self.default_asset_size)
            draw_radius_of_detection(search_party.location,
                                     search_party.base_100_pod_distance())
            draw_radius_of_detection(search_party.location,
                                     search_party.base_100_pod_distance() * 3)
            # draw_radius_of_detection(search_party.location,
            #                          search_party.detection_range // 3)
            # draw_radius_of_detection(search_party.location,
            #                          search_party.detection_range)
            # assert search_party.base_100_pod_distance(self.current_prisoner_speed) * 3 == search_party.detection_range

        # helicopters
        if self.is_helicopter_operating():
            for helicopter in self.helicopters_list:
                draw_image_on_canvas_cv(self.helicopter_pic_cv, helicopter.location, self.default_asset_size)
                draw_radius_of_detection(helicopter.location,
                                         helicopter.base_100_pod_distance())
                draw_radius_of_detection(helicopter.location,
                                         helicopter.base_100_pod_distance() * 3)
                # draw_radius_of_detection(helicopter.location,
                #                          helicopter.detection_range // 3)
                # draw_radius_of_detection(helicopter.location,
                #                          helicopter.detection_range)
                # assert helicopter.base_100_pod_distance(self.current_prisoner_speed) * 3 == helicopter.detection_range
        else:
            for helicopter in self.helicopters_list:
                draw_image_on_canvas_cv(self.helicopter_no_pic_cv, helicopter.location, self.default_asset_size)

        if show_delta:
            # Added by Manisha (Check first before pushing changes) delta = 0.05 = 121.4 on the map
            x1, y1 = self.prisoner.location[0] - 121, self.dim_x - self.prisoner.location[1] + 121
            x2, y2 = self.prisoner.location[0] + 121, self.dim_y - self.prisoner.location[1] - 121
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 0, 1), 2)

        # hideouts
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                draw_image_on_canvas_cv(self.known_hideout_pic_cv, hideout.location, self.hideout_radius)
            else:
                draw_image_on_canvas_cv(self.unknown_hideout_pic_cv, hideout.location, self.hideout_radius)

        # cameras
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                draw_image_on_canvas_cv(self.known_camera_pic_cv, camera.location, camera.detection_range)
            else:
                draw_image_on_canvas_cv(self.unknown_camera_pic_cv, camera.location, camera.detection_range)

        # for mountains in self.terrain.mountain_locations:
        #     # mountains[1] = -mountains[1] + self.dim_y
        #     mountain_loc = (mountains[1], mountains[0])
        #     draw_image_on_canvas_cv(self.prisoner_pic_cv, mountain_loc, 20)

        x, y, _ = self.canvas.shape
        self.canvas = cv2.resize(self.canvas, (x // scale, y // scale))
        # print(np.max(self.canvas))

        x, y, _ = self.canvas.shape
        if show_grid:
            draw_grid_on_canvas_cv(x, y, scale)

        if show:
            cv2.imshow("test.txt", self.canvas)
            cv2.waitKey(1)
        return (self.canvas * 255).astype('uint8')

    def slow_render_canvas(self, show=True):
        """
        Provide a visualization of the current status of the environment.

        In rendering, imshow interprets the matrix as:
        [x, 0]
        ^
        |
        |
        |
        |
        |----------->[0, y]
        However, the extent of the figure is still:
        [0, y]
        ^
        |
        |
        |
        |
        |----------->[x, 0]
        Read https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html for more explanations.

        :param show: whether to show the visualization directly or just return
        :return: an opencv img object
        """

        def calculate_appropriate_image_extent(loc, radius=0.4):
            """
            :param loc: the center location to put a picture
            :param radius: the radius (size) of the figure
            :return: [left, right, bottom, top]
            """
            return [max(loc[0] - radius, 0), min(loc[0] + radius, self.dim_x),
                    max(loc[1] - radius, 0), min(loc[1] + radius, self.dim_y)]

        fig, ax = plt.subplots(figsize=(20, 20))
        # Show terrain
        im = ax.imshow(self.cached_terrain_image, origin='lower')
        # labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # prisoner_history
        prisoner_location_history = np.array(self.prisoner_location_history)
        ax.plot(prisoner_location_history[:, 0], prisoner_location_history[:, 1], "r")

        # prisoner
        if self.is_detected:
            ax.imshow(self.detected_prisoner_pic,
                      extent=calculate_appropriate_image_extent(self.prisoner.location, radius=50))
        else:
            ax.imshow(self.prisoner_pic, extent=calculate_appropriate_image_extent(self.prisoner.location, radius=50))

        # search parties
        for search_party in self.search_parties_list:
            ax.imshow(self.search_party_pic, extent=calculate_appropriate_image_extent(search_party.location,
                                                                                       radius=search_party.detection_range))
        # helicopters
        if self.is_helicopter_operating():
            for helicopter in self.helicopters_list:
                ax.imshow(self.helicopter_pic, extent=calculate_appropriate_image_extent(helicopter.location,
                                                                                         radius=helicopter.detection_range))
        # hideouts
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                ax.imshow(self.known_hideout_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))
            else:
                ax.imshow(self.unknown_hideout_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))

        # cameras
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                ax.imshow(self.known_camera_pic, extent=calculate_appropriate_image_extent(camera.location,
                                                                                           radius=camera.detection_range))
            else:
                ax.imshow(self.unknown_camera_pic, extent=calculate_appropriate_image_extent(camera.location,
                                                                                             radius=camera.detection_range))
        # finalize
        ax.axis('scaled')
        # plt.savefig("temp.png")
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if show:
            plt.show()
        plt.close()

        return img

    def get_prisoner_location(self):
        return self.prisoner.location


    @property
    def fugitive_observation(self):
        return self._fugitive_observation

    @property
    def blue_observation(self):
        #TODO: should be in the format of list of N agents, each agent (79,)
        return self._blue_observation

    @property
    def blue_team_state(self):
        # should be in the format of (316, )
        return self._blue_team_state


if __name__ == "__main__":
    np.random.seed(20)
    # p = PrisonerEnv()
