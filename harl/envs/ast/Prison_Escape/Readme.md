[//]: # (# [Learning Models of Adversarial Agent Behavior under Partial Observability]&#40;https://arxiv.org/pdf/2306.11168.pdf&#41;)

[//]: # ()
[//]: # (### Authors: Sean Ye, Manisha Natarajan, Zixuan Wu, Rohan Paleja, Letian Chen, and Matthew C. Gombolay)

[//]: # ()
[//]: # (### &#40;To Appear at IROS 2023&#41;)

[//]: # (---)

[//]: # (We present two large scale adversarial tracking environments: Prison Escape and Narco Traffic Interdiction as discussed )

[//]: # (in the paper. )
### Experiment Log 
This repository contains the code to run the Prison Escape environment and collect the datasets. 
The codebase for training models is located [here](https://github.com/CORE-Robotics-Lab/GrAMMI).
Dataset can be downloaded here for imitation learning. 

[//]: # (### Installation &#40;Done&#41;)

[//]: # (After cloning the repository, please use the provided conda environment &#40;`environment.yml`&#41; file to install dependencies:)

[//]: # (`conda env create -f environment.yml`)

[//]: # ()
[//]: # (This will create an environment named 'tracking_env'. Please edit the first line of `environment.yml` to name it something else.)

[//]: # ()
[//]: # (`conda activate tracking_env`)

### Prison Escape
#### About the Environment
A **heterogeneous** team of **fixed cameras, UGVs, and UAVs** (blue team) must coordinate to track an escaped **prisoner**
(red team). The game is played on a **2428 x 2428** grid map with varying terrains (woods, dense forest, high mountains) 
where each cell on the grid represents the (x,y) location. (state space is finite). 

REMARK: Assume **only woods for now** (no map encoder) -- remove the other type of terrains that may have different detection probabilities.

This domain is motivated by scenarios in military surveillance and border patrol, where there is a
need to track and intercept adversarial targets to ensure the safety of the general population. 

#### File structure and description
```bash
├── blue_policies
│   ├── heuristic.py
│   ├── __init__.py
├── environment
│   ├── assets
│   │   ├── xxx.png  # images for rendering
│   ├── basic_reward_scheme.json # a RewardScheme object define reward scales for different events
│   ├── camera_locations 
│   │   ├── camera_net.txt # camera surrounded the evader. FALSE. 
│   │   ├── camera_n_percentage # camera locations for different percentage of coverage required generated from generate_cameras.py
│   │   │   ├── 10_percent_cameras.txt
│   │   │   ├── 20_percent_cameras.txt
│   │   │   ├── 30_percent_cameras.txt
│   │   │   ├── xx_percent_cameras.txt
│   │   ├── generate_cameras.py
│   ├── configs
│   │   ├── balance_game.yaml # config file in original paper
│   │   └── mytest.yaml # config file for my experiment
│   ├── forest_coverage # All woods so all 1.0 and no embedding
│   │   ├── generate_square_map.py # generate a square map with different percentage of coverage.

│   ├── abstract_object.py # important!!! Define abstract classes for all objects: # AbstractObject, DetectionObject(AbstractObject), MovingObject(AbstractObject)
# AbstractObject(terrain, location)
# DetectionObject(terrain, location, detection_object_type_coefficient)
│   ├── camera.py # DetectionObject(terrain, location, known_to_evader)
│   ├── fugitive.py # DetectionObject
│   ├── helicopter.py # MovingObject, DetectionObject
│   ├── search_party.py # MovingObject, DetectionObject
│   ├── town.py # AbstractObject(terrain, location) [not used]
│   ├── hideout.py # AbstractObject(terrain, location, known_to_pursuers)

│   ├── generate_hideout_locations.py
│   ├── gnn_wrapper.py
│   ├── __init__.py
│   ├── load_environment_MARLlib.py
│   ├── load_environment.py
│   ├── obs_embedding_wrapper.py
│   ├── observation_spaces.py
│   ├── prisoner_env.py
│   ├── prisoner_env_variations.py
│   ├── prisoner_perspective_envs.py
│   ├── prisoner_sequence_wrapper.py
│   ├── terrain.py
│   └── utils.py
├── fugitive_policies
│   ├── a_star
│   │   ├── a_star.py
│   │   ├── gridmap.py
│   │   ├── occupancy_map_8n.py
│   │   └── utils.py
│   ├── a_star_avoid.py
│   ├── a_star_policy.py
│   ├── base_policy.py
│   ├── custom_queue.py
│   ├── heuristic.py
│   ├── __init__.py
│   ├── rrt.py
│   ├── rrt_star_adversarial_avoid.py
│   ├── rrt_star_adversarial_heuristic.py
│   └── utils.py  # create_camera_net function -- returns a list of camera locations in a square surrounding the evader location all the time
├── __init__.py
├── quick_start.py
└── test.txt.py
├── collect_demonstrations.py
```


#### Simulator
prisoner_env.py: This file contains all info include STEP for the Prison Escape environment.
[this file](./Prison_Escape/environment/prisoner_env.py).

prisoner_perspective_envs.py: This file contains info from the perspective of the blue team.
[this file](./Prison_Escape/environment/prisoner_perspective_envs.py.py).

**Observation Space Blue Team** 

1+2*(19+1+1+3+1+1+1)+1*(19+1+3+1) = 79

Pursuer Team Observation Space SHAPE:  (79,)
Pursuer Team Observation Space TYPE:  float32

Includes: 
time(1) + 

coordinates(2) x [No. of know/unknown cameras + 1 (hideout camera) + No. of known hideout + No. of UAVs + 
No. of UGVs + evader location + evader start location] +

detection(1) x [No. of know/unknown cameras + 1 (hideout camera) + No. of UAVs + No. of UGVs] 

**Action Space Blue Team** 
N is the velocity of the agent.

0: Do nothing
1: Move up N grids
2: Move down N grids
3: Move left N grids
4: Move right N grids
5: Move up left N grids
6: Move up right N grids
7: Move down left N grids
8: Move down right N grids

**State Space** 


**Rendering:** 

We have two modes for rendering the Prison Escape environment. 
We have a **fast** option that is less aesthetic, and a **slow** option that is more aesthetic.
- For training and debugging, please use the fast option.
- For visualizing results, please use the slow rendering option to get the best display.

### Collecting the Dataset
Run `Prison_Escape/collect_demonstrations.py` to collect train and test datasets. Please specify the 
parameters as mentioned in the main function. Each rollout is saved as a numpy file, and 
includes observations from both the blue
and the red team's perspective, the hideout locations, the current timestep, whether the prisoner was seen, 
and done to indicate
the end of the episode. All values are stored for every timestep of each rollout.

In original paper, they describe three datasets for Prison Escape. They obtain this by varying the detection factor
in the simulator config file: `Prison_Escape/environment/configs/balance_game.yaml`

In my experiment, the config file is `Prison_Escape/environment/configs/mytest.yaml`

### TODO
1. Reduce Dimension of terrain map. Make the 2428 parameter in the config file. [Not needed]
2. Remove mountain and forest. Only woods. [Done]
3. Is the position integer? [Yes! It is int 64. So we can change to discrete waypoint action space]
4. detection_range for each object. print them out [Done]
5. How is the velocity x time = position and how did agent arrive to waypoint. How to change to direct position (discrete then) [Done]
6. The camera range is not inserted in the environmet. How does the camera know the detection?
7. Integrate into the HARL framework. 
8. Rewrite the observation space file. Create new create_action_space_blue_team_v2 function [Done]
9. Create a new policy for blue team. （random generate policy）[Done]
10. Take note about the TODO (12 of them). Staged for now and switch to HRL. [Done]
11. How to change the blue heuristic policy to waypoint version? 
12. 不应该是multi discrete，在search任务中没必要。应该是single discrete。 [Done]
13. HARL reset share_obs and available action [Done]
14. Check the reward setting in env
15. construct blue observation. how to distinguish among different agents' obs. Not repeat 
16. the detection still need to be confirmed (see the printing in setupworld)
17. add stopping condition if the pursuers remain stationary for a long time./reach boundary
18. Change the detection range. Unrelated to speed. Too complicated. 


[//]: # (## 2. Narco Traffic Interdiction: )

[//]: # (### About the Environment)

[//]: # (This domain simulates illegal maritime drug trafficking on a $7884 \times 3538$ grid along the Central American Pacific )

[//]: # (Coastline. The adversary, a drug smuggler, is pursued by a team of heterogeneous tracker agents comprising airplanes and)

[//]: # (marine vessels. Airplanes have a larger search radius and speed than marine vessels, but only the vessels can capture )

[//]: # (the smuggler. Smugglers must first reach rendezvous points before heading to the hideouts, representing drug handoffs at)

[//]: # (sea. The locations of hideouts and rendezvous points are unknown to the tracking team. Episodes start after the team )

[//]: # (learns one location of the smuggler and end when the smuggler reaches a hideout or is captured by law enforcement.)

[//]: # ()
[//]: # (### Simulator)

[//]: # (The Narco Traffic domain is setup very similar to the Prison Escape environment, in that we have several classes to represent the terrain, )

[//]: # (different objects &#40;town, camera, etc.&#41;, and step all moving objects based )

[//]: # (on various agent policies/heuristics, which you can find under the `Smuggler/` folder. )

[//]: # (If you would like to know the details of our environment configuration)

[//]: # (&#40;state space, observation space, action space, etc.&#41;, please refer to [this file]&#40;./Smuggler/simulator/smuggler_env.py&#41;.)

[//]: # ()
[//]: # ()
[//]: # (### Collecting the Dataset)

[//]: # (Run `Smuggler/collect_dataset.py` to collect train and test datasets. Please specify the )

[//]: # (parameters as mentioned in the main function. Each rollout is saved as a numpy file, and includes observations from both the blue)

[//]: # (and the red team's perspective, the hideout locations, the current timestep, whether the smuggler was detected, and done to indicate)

[//]: # (the end of the episode. All values are stored for every timestep of each rollout.)

[//]: # (In our paper, we describe two datasets for Narco Traffic Interdiction. We obtain this by varying the parameters as specified)

[//]: # (in the simulator config file: `Prison_Escape/environment/configs/balance_game.yaml`)
[//]: # (---)

[//]: # (## Citation)

[//]: # ()
[//]: # (If you find our code or paper is useful, please consider citing:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{ye2023grammi,)

[//]: # (  title={Learning Models of Adversarial Agent Behavior under Partial)

[//]: # (Observability},)

[//]: # (  author={Ye, Sean and Natarajan, Manisha and Wu, Zixuan and Paleja, Rohan and Chen, Letian and Gombolay, Matthew},)

[//]: # (  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems &#40;IROS&#41;},)

[//]: # (  year={2023})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## License)

[//]: # ()
[//]: # (This code is distributed under an [MIT LICENSE]&#40;LICENSE&#41;.)