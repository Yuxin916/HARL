import copy
import yaml
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle(('o', '+', 'x', '*', '.', 'X')) 

def generate_coalitions_from_agents(agents, config):
    """Takes the generated agents, and forms coalitions
        from the sampled agents. There are n_coalitions formed
        for different coalitions sizes.
    """
    coalitions = {}

    # TRAINING COALITIONS
    coalitions["train"] = {"coalitions":{}}
    coalitions["test"] = {"coalitions":{}}
    capture_agents = agents["capture"]
    predator_agents = agents["predator"]
    num_robots_list = [2, 3, 4, 5, 6] # We are unlikely to use coalitions larger than 6
    num_coalitions = config["n_coalitions"]
    for t in ["train", "test"]:
        for num_agents in num_robots_list:
            num_agents_str = str(num_agents) + "_agents"
            coalitions[t]["coalitions"][num_agents_str] = {}
            out = input("Would you like to visualize the coalitions for %d_%s agents?\n" % (num_agents, t))
            
            if out == "y":
                plot_coalitions = True
            else:
                plot_coalitions = False
            
            for k in range(num_coalitions):
                x = []
                y = []
                # create a coalition
                val = np.random.uniform(0, 1.0)
                if(val < 0.5):
                    num_capture_agents = np.random.randint(num_agents - 1) + 1
                    num_predator_agents = num_agents - num_capture_agents
                else:
                    num_predator_agents = np.random.randint(num_agents - 1) + 1
                    num_capture_agents = num_agents - num_predator_agents

                predator_idxs = np.random.randint(config["n_predator_agents"], size=num_predator_agents)
                capture_idxs = np.random.randint(config["n_capture_agents"], size=num_capture_agents)

                coalitions[t]["coalitions"][num_agents_str][k] ={}
                coalitions[t]["coalitions"][num_agents_str][k]["predator"] = {}
                coalitions[t]["coalitions"][num_agents_str][k]["capture"] = {}

                for i, idx in enumerate(predator_idxs):
                    coalitions[t]["coalitions"][num_agents_str][k]["predator"][int(i)] = deepcopy(predator_agents[idx])
                    x.append(float(predator_agents[idx]["sensing_radius"])); y.append(0)
                for i, idx in enumerate(capture_idxs):
                    coalitions[t]["coalitions"][num_agents_str][k]["capture"][int(i)] = deepcopy(capture_agents[idx])
                    y.append(float(capture_agents[idx]["capture_radius"])); x.append(0)
            
                
                plt.plot(x, y, marker=next(marker), markersize=15, label=f"C{k}")

            plt.legend()
            plt.xlabel("Predator Sensing Radius")
            plt.ylabel("Capturer Capture Radius")
            plt.title("Coaltions: %d agents, %s set" % (num_agents, t))
            plt.savefig("coalition_figures/%s_%d_agents.png" % (t, num_agents))

            if(plot_coalitions):
                
                plt.show()


    return coalitions

            
def main():
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    agents={}
    agents['predator'] = {}
    agents['capture'] = {}
    agents['test_predator'] = {}
    agents['test_capture'] = {}
    num_candidates = config['n_capture_agents'] + config['n_test_capture_agents'] + config['n_predator_agents'] + config['n_test_predator_agents']
    idx_size = int(np.ceil(np.log2(num_candidates)))

    candidate = 0

    func_args = copy.deepcopy(config['traits']['capture'])
    del func_args['distribution']    
    for i in range(config['n_capture_agents']):
        agents['capture'][i] = {}
        agents['capture'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['capture']['distribution'])(**func_args)
        agents['capture'][i]['capture_radius'] = float(val)
        candidate += 1

    func_args = copy.deepcopy(config['traits']['predator'])
    del func_args['distribution']    
    for i in range(config['n_predator_agents']):
        agents['predator'][i] = {}
        agents['predator'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['predator']['distribution'])(**func_args)
        agents['predator'][i]['sensing_radius'] = float(val)
        candidate += 1

    func_args = copy.deepcopy(config['traits']['capture'])
    del func_args['distribution']
    for i in range(config['n_test_capture_agents']):
        agents['test_capture'][i] = {}
        agents['test_capture'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['capture']['distribution'])(**func_args)
        agents['test_capture'][i]['capture_radius'] = float(val)
        candidate += 1

    func_args = copy.deepcopy(config['traits']['predator'])
    del func_args['distribution']
    for i in range(config['n_test_predator_agents']):
        agents['test_predator'][i] = {}
        agents['test_predator'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['predator']['distribution'])(**func_args)
        agents['test_predator'][i]['sensing_radius'] = float(val)
        candidate += 1

    

    coalitions = generate_coalitions_from_agents(agents, config)
    out = input("Would you like to save these as the new predefined coalitions?[y/N]\n")
    if(out == "y"):
        with open('predefined_coalitions.yaml', 'w') as outfile:
            yaml.dump(coalitions, outfile, default_flow_style=False, allow_unicode=True)

        with open('predefined_coalition_agents.yaml', 'w') as outfile:
            yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)
    else:
        print("Coalitions not saved.")

if __name__ == '__main__':
    main()