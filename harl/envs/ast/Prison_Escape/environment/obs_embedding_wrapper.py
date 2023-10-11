"""
Wrapper to reconstruct obs (for LSTM?)
"""

import gym
import numpy as np
import copy
import gc
from gym.spaces import Dict as GymDict, Box


class PrisonerEmbedEnv(gym.Wrapper):
    """ Batches the observations for an lstm """

    def __init__(self, env):
        """
        :param env: the PrisonerEnv instance to wrap.
        :param sequence_len: number of stacked observations to feed into the LSTM
        :param deterministic: whether the worker(s) should be run deterministically
        """
        super().__init__(env)
        # print("*"*60)
        # print(f"INIT GNN-Wrapper ENVIRONMENT")
        self.env = env
        # This is a breakdown of the observation space composition (name as index)
        self.obs_dict = self.env.obs_names._idx_dict

        self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space
        self.num_agents = self.env.num_agents

        self.search_parties = ["search_parties_{}".format(i) for i in range(self.num_search_parties)]
        self.helicopters = ["helicopters_{}".format(i) for i in range(self.num_helicopters)]

        self.agents = self.search_parties + self.helicopters

        self.episode_limit = self.env.max_timesteps

        pass
        # self.observation_shape = (self.total_agents_num, 3)
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
        total_agents_num = (self.num_known_cameras + self.num_unknown_cameras +
                            self.num_helicopters + self.num_search_parties)
        gnn_obs = np.zeros((total_agents_num, 3))
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

        prisoner_detected = obs_named['prisoner_detected']
        prisoner_starting = obs_named['prisoner_starting']

        num_agents = np.array(total_agents_num)

        return gnn_obs, hideouts, timestep, num_agents

    def transform_obs_to_Dict(self, obs):

        temp = {f'search_parties_{i}': obs for i in range(self.num_search_parties)}
        temp.update({f'helicopters_{i}': obs for i in range(self.num_helicopters)})
        return temp

    def transform_action_to_list(self, action_dict):
        # Extract the keys with 'search_parties_' and 'helicopters_' prefixes
        search_parties_keys = [key for key in action_dict.keys() if key.startswith('search_parties_')]
        helicopters_keys = [key for key in action_dict.keys() if key.startswith('helicopters_')]

        # Initialize lists to store the reversed actions
        reversed_search_parties_actions = []
        reversed_helicopters_actions = []

        # Extract and store the reversed actions
        for key in search_parties_keys:
            reversed_search_parties_actions.append(action_dict[key])

        for key in helicopters_keys:
            reversed_helicopters_actions.append(action_dict[key])

        # Combine the reversed actions into a single list
        reversed_actions = reversed_search_parties_actions + reversed_helicopters_actions

        return reversed_actions

    def transform_rew_to_Dict(self, reward):

        temp = {f'search_parties_{i}': reward for i in range(self.num_search_parties)}
        temp.update({f'helicopters_{i}': reward for i in range(self.num_helicopters)})
        return temp

    def transform_done_to_Dict(self, done):

        temp = {f'search_parties_{i}': done for i in range(self.num_search_parties)}
        temp.update({f'helicopters_{i}': done for i in range(self.num_helicopters)})
        return temp

    def transform_info_to_Dict(self, info):
        temp = {f'search_parties_{i}': info for i in range(self.num_search_parties)}
        temp.update({f'helicopters_{i}': info for i in range(self.num_helicopters)})
        return temp


    def reset(self, seed=None):
        obs = self.env.reset(seed)
        obs = self.transform_obs_to_Dict(obs)

        # gnn_obs, hideouts, timestep, num_agents = self.transform_obs(obs)
        # return gnn_obs, obs
        return obs

    def step(self, action_dict):
        action = self.transform_action_to_list(action_dict=action_dict)

        obs, reward, dones, info = self.env.step(action)

        # transform obs, reward, done, info to Dict
        obs = self.transform_obs_to_Dict(obs)
        reward = self.transform_rew_to_Dict(reward)
        done = self.transform_done_to_Dict(dones)
        done["__all__"] = True if dones is True else False
        info = self.transform_info_to_Dict(info)

        return obs, reward, done, info
