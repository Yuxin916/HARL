import yaml
import os
from robotarium_gym.utilities.misc import run_env, objectview
import argparse

"""
这个文件是用来evaluate训练好的模型的

调用misc.py中的run_env函数来运行环境
"""

def main():
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "robotarium_gym":
        module_dir = ""

    # 读取命令行参数
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    parser.add_argument('--scenario', type=str, default='HeterogeneousSensorNetwork', help='scenario name')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the images. Show fig must be true. Set to None not to save')
    parser.add_argument('--config-dir', type=str, default=None, help='Directory to configuration file. Default (None) will be in the scenarios directory.')
    parser.add_argument('--config-file', type=str, default=None, help='Directory to configuration file. Default (None) will be in the scenarios directory.')
    args = parser.parse_args()

    # 读取环境配置文件
    config_path = ''
    if module_dir == "":
        config_file = "test_config.yaml"
    elif args.config_dir is not None:
        config_path = os.path.join(args.config_dir)
        config_file = os.path.join(config_path, "config.yaml")
    else:
        config_path = os.path.join(module_dir, "scenarios", args.scenario)
        config_file = os.path.join(config_path, "test_config.yaml")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    # 运行环境
    run_env(config, module_dir, args.save_dir, eval_dir=config_path, eval_file_name="eval.json")

if __name__ == '__main__':
    main()