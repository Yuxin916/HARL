import copy
import yaml
import numpy as np

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

    out = input("Would you like to save these as the new predefined agents?[y/N]\n")
    if(out == "y"):

        with open('test_predefined_agents.yaml', 'w') as outfile:
            yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)
    else:
        print("Agents not saved.")

if __name__ == '__main__':
    main()