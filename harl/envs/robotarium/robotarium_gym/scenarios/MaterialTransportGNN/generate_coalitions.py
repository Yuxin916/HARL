import copy
import yaml
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle(('o', '+', 'x', '*', '.', 'X')) 

def generate_coalitions_from_agents(agents_, config):
    """Takes the generated agents, and forms coalitions
        from the sampled agents. There are n_coalitions formed
        for different coalitions sizes.
    """
    coalitions = {}

    # TRAINING COALITIONS
    coalitions["train"] = {"coalitions":{}}
    coalitions["test"] = {"coalitions":{}}
    num_robots_list = [2, 3, 4, 5, 6] # We are unlikely to use coalitions larger than 6
    num_coalitions = config["n_coalitions"]
    for t in ["train", "test"]:
        agents = agents_[t]
        for num_agents in num_robots_list:
            num_agents_str = str(num_agents) + "_agents"
            coalitions[t]["coalitions"][num_agents_str] = {}
            for k in range(num_coalitions):
                
                agent_idxs = np.random.randint(config["n_" + t + "_agents"], size=num_agents)
                coalitions[t]["coalitions"][num_agents_str][k] ={}
                coalitions[t]["coalitions"][num_agents_str][k]
                coalitions[t]["coalitions"][num_agents_str][k]

                for i, idx in enumerate(agent_idxs):
                    coalitions[t]["coalitions"][num_agents_str][k][int(i)] = deepcopy(agents[idx])
    return coalitions

            
def main():
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    agents={}
    agents['train'] = {}
    agents['test'] = {}
    num_candidates = config['n_train_agents'] + config['n_test_agents']
    idx_size = int(np.ceil(np.log2(num_candidates)))

    func_args = copy.deepcopy(config['traits']['radius'])
    del func_args['distribution']   

    candidate = 0
    for i in range(config['n_train_agents']):
        agents['train'][i] = {}
        agents['train'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['radius']['distribution'])(**func_args)
        agents['train'][i]['radius'] = float(val)
        candidate += 1
 
    for i in range(config['n_test_agents']):
        agents['test'][i] = {}
        agents['test'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['radius']['distribution'])(**func_args)
        agents['test'][i]['radius'] = float(val)
        candidate += 1

    coalitions = generate_coalitions_from_agents(agents, config)
    out = input("Would you like to save these as the new predefined coalitions?[y/N]\n")
    if(out == "y"):
        with open('predefined_coalitions.yaml', 'w') as outfile:
            yaml.dump(coalitions, outfile, default_flow_style=False, allow_unicode=True)

        with open('predefined_coalition_agents.yaml', 'w') as outfile:
            yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)

        with open('predefined_agents.yaml', 'w') as outfile:
            yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)
    else:
        print("Coalitions not saved.")

if __name__ == '__main__':
    main()