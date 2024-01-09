from absl import flags
from harl.envs.smac.smac_logger import SMACLogger
from harl.envs.smacv2.smacv2_logger import SMACv2Logger
from harl.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from harl.envs.gym.gym_logger import GYMLogger
from harl.envs.football.football_logger import FootballLogger
from harl.envs.dexhands.dexhands_logger import DexHandsLogger
from harl.envs.lag.lag_logger import LAGLogger
from harl.envs.ast.ast_logger import ASTLogger
from harl.envs.topo.topo_logger import TopoLogger
from harl.envs.robotarium.robotarium_logger import RobotariumLogger
from functools import partial
from harl.envs.robotarium.multiagentenv import MultiAgentEnv
from harl.envs.robotarium.gymmawrapper import _GymmaWrapper

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "lag": LAGLogger,
    "ast": ASTLogger,
    "topo": TopoLogger,
    "robotarium": RobotariumLogger,

}

def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_fn函数的输入是env和**kwargs，输出是MultiAgentEnv类的对象env(**kwargs)
    # 进入_GymmaWrapper类的__init__函数
    return env(**kwargs)
# 新环境的注册
REGISTRY = {}
REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)