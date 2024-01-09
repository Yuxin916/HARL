from gym import spaces, Env
from .scenarios.PredatorCapturePrey.PredatorCapturePrey import PredatorCapturePrey
from .scenarios.PredatorCapturePreyGNN.PredatorCapturePreyGNN import PredatorCapturePreyGNN
from .scenarios.Warehouse.warehouse import Warehouse
from .scenarios.MaterialTransport.MaterialTransport import MaterialTransport
from .scenarios.MaterialTransportGNN.MaterialTransportGNN import MaterialTransportGNN
from .scenarios.Simple.simple import simple
from .scenarios.ArcticTransport.ArcticTransport import ArcticTransport
from .scenarios.HeterogeneousSensorNetwork.HeterogeneousSensorNetwork import HeterogeneousSensorNetwork
#Add other scenario imports here
from robotarium_gym.utilities.misc import objectview
import os
import yaml

env_dict = {'PredatorCapturePrey': PredatorCapturePrey,
            "PredatorCapturePreyGNN": PredatorCapturePreyGNN,
            'Warehouse': Warehouse,
            'MaterialTransport': MaterialTransport,
            "MaterialTransportGNN": MaterialTransportGNN,
            'Simple': simple,
            'ArcticTransport': ArcticTransport,
            "HeterogeneousSensorNetwork": HeterogeneousSensorNetwork}


class Wrapper(Env):
    def __init__(self, env_name, config_path):
        """Creates the Gym Wrappers

        Args:
            env (PredatorCapturePrey): A PredatorCapturePrey object to wrap in a gym env
        """
        # 这里读取了HSN的config.yaml
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        args = objectview(config)

        #  env_dict是在robotarium_gym/__init__.py中register的环境名称
        # 把环境args传入HSN环境中，初始化环境
        self.env = env_dict[env_name](args)
        self.observation_space = self.get_observation_space()
        self.state_space = self.get_state_space()
        self.action_space = self.get_action_space()
        self.n_agents = self.env.num_robots

    def reset(self):
        # Reset the wrapped environment and return the initial observation
        observation = self.env.reset()
        return observation

    def step(self, action_n):
        # Execute the given action in the wrapped environment
        obs_n, reward_n, done_n, info_n = self.env.step(action_n)
        return tuple(obs_n), reward_n, done_n, info_n
    
    def get_action_space(self):
        return self.env.get_action_space()
    
    def get_observation_space(self):
        return self.env.get_observation_space()

    def get_state_space(self):
        return self.env.get_state_space()

    def get_adj_matrix(self):
        """Returns the adjacency matrix of the environment """
        return self.env.get_adj_matrix()
