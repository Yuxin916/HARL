import yaml
import os
from robotarium_gym.utilities.misc import run_env, objectview
import argparse
import glob
import time

def run_eval(config_files, config_dir, module_dir, gif_dir):

    for i, config_file in enumerate(config_files):
        with open(os.path.join(config_dir, config_file), 'r') as f:
            config = yaml.safe_load(f)
            config = objectview(config)

        run_env(config, module_dir, gif_dir, eval_dir=config_dir, eval_file_name=config_file.split(".y")[0])
        time.sleep(5)


def main():
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "robotarium_gym":
        module_dir = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    parser.add_argument('--gif-dir', type=str, default=None, help='Directory to save the images. Show fig must be true. Set to None not to save')
    parser.add_argument('--config-dir', type=str, help='Directory to configuration file. Default (None) will be in the scenarios directory.')
    args = parser.parse_args()

    
    # config_dir = os.path.join(module_dir, "scenarios", args.scenario, args.config_dir)
    config_dir = args.config_dir

    file_pattern = os.path.join(config_dir, '*.yaml')
    config_files = glob.glob(file_pattern)
    
    run_eval(config_files, config_dir, module_dir, config_dir)
    # with open(os.path.join(config_dir, args.config_file), 'r') as f:
    #     config = yaml.safe_load(f)
    #     config = objectview(config)

    
    # run_env(config, module_dir, args.gif_dir, eval_dir=config_dir, eval_file_name=args.config_file.split(".y")[0])

if __name__ == '__main__':
    main()