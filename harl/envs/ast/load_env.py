import os
import sys
sys.path.append(os.getcwd())
from .Prison_Escape.environment import PrisonerBothEnv

def load_environment(data):
    """
    input environment configuration instead of config_path
    """
    if data['mountain_locations'] is not None:
        mountain_locs = [list(map(int, (x.split(',')))) for x in data['mountain_locations']]
    else:
        mountain_locs = []
    known_hideout_locations = [list(map(int, (x.split(',')))) for x in data['known_hideout_locations']]
    unknown_hideout_locations = [list(map(int, (x.split(',')))) for x in data['unknown_hideout_locations']]

    env = PrisonerBothEnv(
        # terrain
        terrain=data['terrain'],
        terrain_x=data['terrain_x'],
        terrain_y=data['terrain_y'],
        terrain_map=data['terrain_map'],
        percent_dense=data['percent_dense'],
        percent_mountain=data['percent_mountain'],
        mountain_locations=mountain_locs,

        #   observation_step_type=data['observation_step_type'],

        # camera
        random_cameras=data['random_cameras'],
        num_random_known_cameras=data['num_random_known_cameras'],
        num_random_unknown_cameras=data['num_random_unknown_cameras'],
        camera_file_path=data['camera_file_path'],
        camera_range_factor=data['camera_range_factor'],
        camera_net_bool=data['camera_net_bool'],
        camera_net_path=data['camera_net_path'],

        # pursuers team
        num_search_parties=data['num_search_parties'],
        num_helicopters=data['num_helicopters'],
        search_party_speed=data['search_party_speed'],
        helicopter_speed=data['helicopter_speed'],
        helicopter_battery_life=data['helicopter_battery_life'],
        helicopter_recharge_time=data['helicopter_recharge_time'],
        random_init_positions=data['random_init_positions'],
        helicopter_init_pos=data['helicopter_init_pos'],
        search_party_init_pos=data['search_party_init_pos'],

        # evader team
        spawn_mode=data['spawn_mode'],
        spawn_range=data['spawn_range'],
        min_distance_from_hideout_to_start=data['min_distance_from_hideout_to_start'],
        prisoner_init_pos=data['prisoner_init_pos'],

        # Observation
        observation_terrain_feature=data['observation_terrain_feature'],
        include_camera_at_start=data['include_camera_at_start'],
        include_start_location_blue_obs=data['include_start_location_blue_obs'],
        store_last_k_fugitive_detections=data['store_last_k_fugitive_detections'],

        # others
        step_reset=data['step_reset'],
        detection_factor=data['detection_factor'],

        # hideout
        num_known_hideouts=data['num_known_hideouts'],
        num_unknown_hideouts=data['num_unknown_hideouts'],
        hideout_radius=data['hideout_radius'],
        random_hideout_locations=data['random_hideout_locations'],
        unknown_hideout_locations=unknown_hideout_locations,
        known_hideout_locations=known_hideout_locations,


    )

    assert data['num_unknown_hideouts'] != 0, "Must have at least one unknown hideout"
    return env