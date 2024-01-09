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
    # 分别生成N个agents的coalitions (也可以只选2/3/4）
    num_robots_list = [2, 3, 4, 5, 6] # We are unlikely to use coalitions larger than 6
    num_coalitions = config["n_coalitions"]
    for t in ["train", "test"]:
        agents = agents_[t]
        for num_agents in num_robots_list:
            num_agents_str = str(num_agents) + "_agents"
            coalitions[t]["coalitions"][num_agents_str] = {}
            #out = input("Would you like to visualize the coalitions for %d_%s agents?\n" % (num_agents, t))
            out = "N"
            if out == "y":
                plot_coalitions = True
            else:
                plot_coalitions = False
            
            for k in range(num_coalitions):
                
                # num_coalition == num_candidates
                init_agent_idx = [k]
                agent_idxs = np.random.randint(config["n_" + t + "_agents"], size=num_agents-1)

                agent_idxs = list(agent_idxs) + init_agent_idx
                coalitions[t]["coalitions"][num_agents_str][k] ={}
                coalitions[t]["coalitions"][num_agents_str][k]
                coalitions[t]["coalitions"][num_agents_str][k]

                for i, idx in enumerate(agent_idxs):
                    coalitions[t]["coalitions"][num_agents_str][k][int(i)] = deepcopy(agents[idx])
    return coalitions
    # 分别生成训练和测试的集合，每个集合包含不同数量N的agents (2,3,4,5,6)，每个集合包含n个coalitions，每个coalition包含N个agents的id和radius

            
def main():
    # 读取环境配置文件
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    # 生成训练的agents和测试的agents
    agents={}
    agents['train'] = {}
    agents['test'] = {}
    num_candidates = config['n_train_agents'] + config['n_test_agents']
    idx_size = int(np.ceil(np.log2(num_candidates)))
    # radius的分布 - U(0.2, 0.6)
    func_args = copy.deepcopy(config['traits']['radius'])
    del func_args['distribution']   

    # 生成训练和测试agents的id和radius #TODO：为什么训练和测试的分布保持一致？
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

    # 分别生成训练和测试的集合，每个集合包含不同数量N的agents (2,3,4,5,6)，每个集合包含n个coalitions，每个coalition包含N个agents的id和radius
    coalitions = generate_coalitions_from_agents(agents, config)

    out = input("Would you like to save these as the new predefined coalitions?[y/N]\n")
    if(out == "y"):
        with open(config['coalition_file'], 'w') as outfile:
            print("Saving coalitions to %s" % outfile)
            yaml.dump(coalitions, outfile, default_flow_style=False, allow_unicode=True)

        # with open(config['coalition_file'].split(".y")[0] + '_agents.yaml', 'w') as outfile:
        #     print("Saving agents to %s" % outfile)
        #     yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)

    else:
        print("Coalitions not saved.")

if __name__ == '__main__':
    main()