# This file exports PrisonerEscape environment by registering in OpenAI Gym
# so that we could do `gym.make('PrisonerEscape-v0')`
import sys
sys.path.append('/home/tsaisplus/MuRPE_base/Heterogenous-MARL/harl/envs/ast/')

from gym.envs import register
from .prisoner_env import PrisonerBothEnv, RewardScheme
from .prisoner_perspective_envs import PrisonerBlueEnv
from .observation_spaces import ObservationNames
from .terrain import Terrain

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return

    print("Registering custom gym environments, could do `gym.make('PrisonerEscape-v0')`")
    register(id='PrisonerEscape-v0', entry_point='simulator.prisoner_env:PrisonerEnv', kwargs={})

    _REGISTERED = True


# register_custom_envs()