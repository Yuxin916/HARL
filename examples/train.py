"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

"""
python train.py --algo <ALGO> --env <ENV> --exp_name <EXPERIMENT NAME> or 
python train.py --load_config <CONFIG FILE PATH>

for example: 
python train.py --algo mappo --env pettingzoo_mpe --exp_name mpe_mappo
python train.py --load_config /home/tsaisplus/MuRPE_base/Heterogenous-MARL/tuned_configs/pettingzoo_mpe/simple_spread_v2-discrete/mappo/config.json

eg:
share_observation_space:  [Box(-inf, inf, (54,), float32), Box(-inf, inf, (54,), float32), Box(-inf, inf, (54,), float32)]
observation_space:  [Box(-inf, inf, (18,), float32), Box(-inf, inf, (18,), float32), Box(-inf, inf, (18,), float32)]
action_space:  [Box(0.0, 1.0, (5,), float32), Box(0.0, 1.0, (5,), float32), Box(0.0, 1.0, (5,), float32)]

"""

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 使用什么算法
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    # 使用什么环境
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "ast",
            "robotarium"
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag, ast.",
    )
    # 实验名称
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    # 是否使用config file
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    # 读取命令行参数
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # env-specific的参数
    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    if args["env"] == "robotarium":
        algo_args["train"]["episode_length"] = 80  # FIXED

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
