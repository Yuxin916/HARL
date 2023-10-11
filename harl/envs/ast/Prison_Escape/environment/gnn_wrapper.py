"""
Wrapper to support LSTM sequence.
"""

import gym
import numpy as np
import copy
import gc

from Prison_Escape.environment import PrisonerEnv


class PrisonerGNNEnv(gym.Wrapper):
    """ Batches the observations for an lstm """
    def __init__(self, env, seq_len = 1):
        """
        :param env: the PrisonerEnv instance to wrap.
        :param sequence_len: number of stacked observations to feed into the LSTM
        :param deterministic: whether the worker(s) should be run deterministically
        """
        super().__init__(env)
        # print("*"*60)
        # print(f"INIT GNN-Wrapper ENVIRONMENT")
        assert seq_len > 0
        self.env = env
        self.seq_len = seq_len
        self.obs_dict = self.env.obs_names._idx_dict
        
        self.total_agents_num = self.num_known_cameras + self.num_unknown_cameras + self.num_helicopters + self.num_search_parties
        self.observation_shape = (self.total_agents_num, 3)
        # print(f'Observation shape: {self.observation_shape}')

    def transform_obs(self, obs):
        """ This function creates three numpy arrays,
        the first representing all the agents,
        the second representing the hideouts,
        and the third the timestep"""

        obs_names = self.env.obs_names
        # This is a breakdown of the observation space composition (name as index)
        #  obs_named [.name and .array]
        obs_named = obs_names(obs)

        names = [[self.num_known_cameras, 'known_camera_', 'known_camera_loc_'], 
                [self.num_unknown_cameras, 'unknown_camera_', 'unknown_camera_loc_'],
                [self.num_helicopters, 'helicopter_', 'helicopter_location_'],
                [self.num_search_parties, 'search_party_', 'search_party_location_']]

        
        # (detect bool, x_loc, y_loc)
        gnn_obs = np.zeros((self.total_agents_num, 3))
        j = 0
        for num, detect_name, location_name in names:
            for i in range(num):
                detect_key = f'{detect_name}{i}'
                loc_key = f'{location_name}{i}'
                gnn_obs[j, 0] = obs_named[detect_key]
                gnn_obs[j, 1:] = obs_named[loc_key]
                j += 1
        # gnn_obs exclude below from obs
        #   -'time': (0, 1),
        #   - 'hideout_loc_0': (51, 53),
        #   - 'prisoner_detected': (90, 92),
        #   - 'prisoner_starting': (92, 94)}"

        timestep = obs_named['time']

        hideouts = np.zeros((self.num_known_hideouts, 2))
        for i in range(self.num_known_hideouts):
            key = f'hideout_loc_{i}'
            hideouts[i, :] = obs_named[key]
        hideouts = hideouts.flatten()

        num_agents = np.array(self.total_agents_num)

        return gnn_obs, hideouts, timestep, num_agents

    def reset(self, seed=None):
        # print("*"*60)
        # print(f"RESET GNN-Wrapper ENVIRONMENT")
        obs = self.env.reset(seed)
        gnn_obs = self.transform_obs(obs)
        return gnn_obs, obs

    def step(self, action):
        # print("*"*60)
        # print(f"STEP GNN-Wrapper ENVIRONMENT")
        obs, reward, done, i = self.env.step(action)
        gnn_obs = self.transform_obs(obs)
        return gnn_obs, obs, reward, done, i