import numpy as np
from gym import spaces
import copy
import yaml
import os
from copy import deepcopy
import math

# This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.HeterogeneousSensorNetwork.visualize import *
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_python_simulator.rps.utilities.graph import *

from robotarium_gym.utilities.misc import is_close
import numpy as np


class Agent:
    '''
    This is a helper class for PredatorCapturePrey
    Keeps track of information for each agent and creates functions needed by each agent
    This could optionally all be done in PredatorCapturePrey
    '''

    def __init__(self, index, id, radius, action_id_to_word, args):
        self.index = index
        self.id = id
        self.radius = radius
        self.action_id2w = action_id_to_word
        self.args = args

    def generate_goal(self, goal_pose, action, args):
        '''
        Sets the agent's goal to step_dist in the direction of choice
        Bounds the agent by args.LEFT, args.RIGHT, args.UP and args.DOWN
        '''

        if self.action_id2w[action] == 'left':
            goal_pose[0] = max(goal_pose[0] - args.step_dist, args.LEFT)
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
            goal_pose[0] = min(goal_pose[0] + args.step_dist, args.RIGHT)
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = max(goal_pose[1] - args.step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = min(goal_pose[1] + args.step_dist, args.DOWN)
        else:
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]

        return goal_pose


class HeterogeneousSensorNetwork(BaseEnv):
    """
    所有环境都是基于BaseEnv的
    """

    def __init__(self, args):
        # 环境相关的args - /mat/envs/robotarium/robotarium_gym/scenarios/HeterogeneousSensorNetwork/config.yaml
        self.args = args

        # 当前scenario的文件夹
        module_dir = os.path.dirname(__file__)
        # with open(f'{module_dir}/predefined_agents.yaml', 'r') as stream:
        #     self.predefined_agents = yaml.safe_load(stream)

        # 是否使用随机种子[-1不随机][任意其他数字 随机]
        if self.args.seed != -1:
            np.random.seed(self.args.seed)

        # if(args.hard_coded_coalition):  # KEEP FALSE
        #     self.args.resample = False
        #     with open(f'{module_dir}/grid_search_coalitions.yaml', 'r') as stream:
        #         self.predefined_coalition = yaml.safe_load(stream)
        #     raise NotImplementedError("Hard coded coalition not implemented yet")
        #
        # else:

        # 从预生成的coalition中加载，文件里包含所有robot的id和radius
        with open(f'{module_dir}/{args.coalition_file}', 'r') as stream:
            self.predefined_coalition = yaml.safe_load(stream)

        # n个agent
        self.num_robots = args.n_agents

        # declare
        self.agent_poses = None  # robotarium convention poses
        self.episode_number = 0
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'no_action'}

        # Initializes the agents
        # 初始化所有agent
        self.agents = []
        if self.args.load_from_predefined_coalitions:
            # 根据预定义的coalition初始化agent
            self.agents = self.load_agents_from_predefined_coalitions()
            # print("Loading from Predefined Coalitions")
        elif (self.args.load_from_predefined_agents):
            # 根据预定义的agent初始化agent
            self.agents = self.load_new_coalition_from_predefined_agents()
            print("Loading from Predefined Agents")
        else:
            # 根据分布初始化agent
            self.agents = self.load_agents_from_trait_distribution()
            print("Loading from Trait Distribution")

        # 单个agent的观测空间dim
        if self.args.capability_aware:
            self.agent_obs_dim = 2 + 1
        elif self.args.agent_id:  # agent ids are one hot encoded
            # self.agent_obs_dim = 2 + self.num_robots * self.args.n_coalitions
            # ID的长度是self.num_robots * self.args.n_coalitions
            self.agent_obs_dim = 2 + self.num_robots * self.args.n_coalitions  # TODO: FIX, THIS IS HACKY
        else:
            self.agent_obs_dim = 2

        # 单个agent的state空间dim = 单个agent的观测空间dim * agent的数量 (假设全局观测
        self.agent_state_dim = self.agent_obs_dim * self.args.n_agents

        ## 状态空间和动作空间
        actions = []
        observations = []
        states = []

        # 对于每个agent
        for i in range(self.num_robots):
            actions.append(spaces.Discrete(5))
            observations.append(spaces.Box(low=-1.5, high=1.5, shape=(self.agent_obs_dim,), dtype=np.float32))
            states.append(spaces.Box(low=-1.5, high=1.5, shape=(self.agent_state_dim,), dtype=np.float32))

        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        self.state_space = spaces.Tuple(tuple(states))

        # 初始化可视化
        self.visualizer = Visualize(self.args)

        # 初始化环境，基于roboEnv
        self.env = roboEnv(self, args)

        # 初始化通信网络 - 衡量哪些智能体是需要进行interaction
        self.adj_matrix = 1 - np.identity(self.num_robots, dtype=int)

    def _generate_step_goal_positions(self, actions):
        '''
        User implemented
        Calculates the goal locations for the current agent poses and actions
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:, i] = agent.generate_goal(goal[:, i], actions[i], self.args)
        return goal

    def load_agents_from_trait_distribution(self):
        """Loades agents/coaltions from by sampling each agent individually from trait distribution
        """
        agents = []
        func_args = copy.deepcopy(self.args.traits['radius'])
        del func_args['distribution']

        index = 0
        for idx in range(self.num_robots):
            radius_val = float(getattr(np.random, self.args.traits["radius"]['distribution'])(**func_args))
            default_id = ['0'] * (self.num_robots * self.args.n_coalitions)
            agents.append(Agent(index, default_id, radius_val, self.action_id2w, self.args))
            index += 1

        return agents

    def load_agents_from_predefined_coalitions(self):
        """
        Loades the pre-defined agents / coalitions
        """
        # 选取训练集还是测试集
        t = "train"
        if self.args.test:
            t = "test"

        # 初始化agent列表
        agents = []

        # 是否需要手动选择coalition
        if (self.args.manual_coalition_selection):
            coalition_idx = self.args.coalition_selection
        else:
            # 根据coalition的数量随机生成一个index
            coalition_idx = np.random.randint(self.args.n_coalitions)

        # 选取custom_5_coalitions_4_robots.yaml中的属于n agent的第coalition_idx个coalition
        s = str(self.num_robots) + "_agents"
        coalition = self.predefined_coalition[t]["coalitions"][s][coalition_idx]

        index = 0
        for idx, agent in coalition.items():
            # agents包含每一个agent的id和radius
            agents.append(Agent(index, agent["id"], agent["radius"], self.action_id2w, self.args))
            index += 1
        return agents

    def load_new_coalition_from_predefined_agents(self):
        '''
        Loades the pre-defined agents, and draws a random coalition from them
        '''
        # 选取训练集还是测试集
        t = "train"
        if self.args.test:
            t = "test"

        # 用来暂时存储所有的agents
        agent_pool = []
        agents = []

        s = str(self.num_robots) + "_agents"

        # 遍历所有的coalition -- custom_5_coalitions_4_robots.yaml里的所有coalition
        for coalition_idx in range(self.args.n_coalitions):
            coalition = self.predefined_coalition[t]["coalitions"][s][coalition_idx]

            index = 0
            for idx, agent in coalition.items():
                agent_pool.append(Agent(index, agent["id"], agent["radius"], self.action_id2w, self.args))
                index += 1

        # 根据需要n个robot来从agent pool随机sample
        index = 0
        for idx in range(self.num_robots):
            agent = random.choice(agent_pool)
            agent.index = index
            agents.append(agent)
            index += 1

        return agents

    def reset(self):
        '''
        Resets the simulation
        '''
        # episode数量+1
        self.episode_number += 1
        self.episode_steps = 0
        self.episode_return = 0

        # 这里的resample是重新选择agents
        if self.args.resample and (self.episode_number % self.args.resample_frequency == 0):
            if self.args.load_from_predefined_coalitions:
                self.agents = self.load_agents_from_predefined_coalitions()
            elif self.args.load_from_predefined_agents:
                self.agents = self.load_new_coalition_from_predefined_agents()
            else:
                self.agents = self.load_agents_from_trait_distribution()

        # shuffles the order of agents
        if self.args.shuffle_agent_order:
            self.agents = self.shuffle_agents(self.agents)

        # 场地长宽
        width = self.args.RIGHT - self.args.LEFT
        height = self.args.DOWN - self.args.UP

        # 根据robot数量，初始最小距离和场地长宽 生成agents的初始位置 --》3xN numpy array (of poses) [x,y,theta]
        self.agent_poses = generate_initial_conditions(self.num_robots, spacing=self.args.start_dist, width=width,
                                                       height=height)

        # Adjusts the poses based on the config
        self.agent_poses[0] += (1.5 + self.args.LEFT) / 2
        self.agent_poses[0] -= (1.5 - self.args.RIGHT) / 2
        self.agent_poses[1] -= (1 + self.args.UP) / 2
        self.agent_poses[1] += (1 - self.args.DOWN) / 2

        self.env.reset()

        return [[0] * self.agent_obs_dim] * self.num_robots

    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info
        '''
        terminated = False
        if self.episode_steps == 0:
            assert self.episode_return == 0, "Episode return is not 0 at the start of the episode"
        self.episode_steps += 1


        # call the environment step function and get the updated state
        return_message = self.env.step(actions_)

        # 当前step的obs
        obs = self.get_observations()

        # 获得这一步的reward和reward相关的信息
        rewards, rew_metric_info = self.get_rewards()

        # 不会提前终止，但是会给一个互相/边界碰撞的惩罚
        # penalize for collisions, record in info
        violation_occurred = 0
        if self.args.penalize_violations:
            if self.args.end_ep_on_violation and return_message != '':
                violation_occurred += 1
                rewards += self.args.violation_penalty
                # terminated=True
            elif not self.args.end_ep_on_violation:
                violation_occurred += 1
                # intent碰撞的次数
                violation_occurred = return_message
                rewards += np.log(return_message + 1) * self.args.violation_penalty
                # Taking the log because this can get out of control otherwise

        # 累计episodic reward
        self.episode_return += rewards

        # terminate if needed
        if self.episode_steps > self.args.max_episode_steps:
            terminated = True
            rew_metric_info["episode_return"] = self.episode_return

        info = {
            "violation_occurred": violation_occurred,  # not a true count, just binary for if ANY violation occurred
        }
        # 添加info
        info.update(rew_metric_info)

        return obs, [rewards] * self.num_robots, [terminated] * self.num_robots, [info] * self.num_robots

    def get_observations(self):
        # 获取当前step的obs

        observations = []  # Each agent's individual observation
        neighbors = []  # Stores the neighbors of each agent if delta > -1

        # def one_hot_encode(number, num_classes):
        #     encoded = [0] * num_classes
        #     encoded[number] = 1
        #     return encoded

        # 对于每个agent，获取其观测
        for a in self.agents:
            # 是否已知agent的radius
            if self.args.capability_aware:
                # 给位置后面append
                observations.append([*self.agent_poses[:, a.index][:2], a.radius])
            # 是否已知agent的id
            elif self.args.agent_id:
                agent_id = [int(bit) for bit in a.id]
                # 给位置后面append
                observations.append([*self.agent_poses[:, a.index][:2], *agent_id])
            # 否则只知道agent的位置
            else:
                observations.append([*self.agent_poses[:, a.index][:2]])

            if self.args.delta > -1:
                neighbors.append(delta_disk_neighbors(self.agent_poses, a.index, self.args.delta))

        return observations

    def get_rewards(self):
        """
        The agents goal is to get their radii to touch
        reward agent if they are more towards the center.

        """

        # Fully shared reward, this is a collaborative environment.
        reward = 0  # 总reward
        center_reward = []  # 每一个agent的距离中心的reward
        total_overlap = 0 if self.args.calculate_total_overlap else -1  # 总overlap
        edge_count = 0  # agent之间相交了几次
        edge_set = []  # 相交的agent set

        # 对于每两个agent来说
        for i, a1 in enumerate(self.agents):
            # agent a1的位置
            x1, y1 = self.agent_poses[:2, a1.index]
            # 计算a1和中心的距离
            center_reward.append(np.sqrt(np.sum(np.square(self.agent_poses[:2, a1.index] - np.array([0, 0])))))
            for j, a2 in enumerate(self.agents[i + 1:], i + 1):  # don't duplicate
                # agent a2的位置
                x2, y2 = self.agent_poses[:2, a2.index]
                # 计算a1和a2这两个agent之间的距离
                dist = np.sqrt(np.sum(np.square(self.agent_poses[:2, a1.index] - self.agent_poses[:2, a2.index])))
                # 计算a1和a2这两个agent之间的距离与它们半径之和的差值的变量 - 还差多少相接
                difference = dist - (a1.radius + a2.radius)
                # 两个agent相接
                if (difference < 0):
                    edge_count += 1
                    edge_set.append((i, j))
                # 是否计算覆盖面积
                if self.args.calculate_total_overlap:
                    # 读取两个agent的半径
                    r1 = a1.radius;
                    r2 = a2.radius
                    overlap = 0.0
                    # 当确实有相交时
                    if dist < r1 + r2:
                        # 两个圆中的一个完全包含在另一个圆内
                        if dist <= abs(r1 - r2):
                            # 重叠区域等于较小圆的整个面积
                            overlap = math.pi * min(r1, r2) ** 2
                        # 两个圆部分相交
                        else:
                            # 计算重叠面积
                            theta1 = math.acos((r1 ** 2 + dist ** 2 - r2 ** 2) / (2 * r1 * dist))
                            theta2 = math.acos((r2 ** 2 + dist ** 2 - r1 ** 2) / (2 * r2 * dist))
                            overlap = theta1 * r1 ** 2 + theta2 * r2 ** 2 - 0.5 * r1 ** 2 * math.sin(
                                2 * theta1) - 0.5 * r2 ** 2 * math.sin(2 * theta2)

                    total_overlap += overlap

                # 两个agent相接的reward
                if (difference < 0):
                    reward += -0.9 * abs(difference) + 0.05
                else:
                    reward += -1.1 * abs(difference) - 0.05

                    # 每个agent距离中心的reward
        reward += -1 * min(center_reward) * self.args.dist_reward_multiplier

        info = {'edge_count': edge_count, 'total_overlap': total_overlap, "edge_set": edge_set}

        return reward, info

    def shuffle_agents(self, agents):
        """
        Shuffle the order of agents
        """
        agents_ = deepcopy(agents)
        random.shuffle(agents_)
        return (agents_)

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_state_space(self):
        return self.state_space
