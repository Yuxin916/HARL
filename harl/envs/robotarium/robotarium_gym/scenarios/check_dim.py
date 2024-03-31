import numpy as np
def check_obs_dimensions(variable, n_agent, obs_dim):
    # Check if the variable is a list and has length equal to n_agent
    if not isinstance(variable, list) or len(variable) != n_agent:
        return False

    # Check each element in the variable
    for element in variable:
        # Each element must be a list and have a length equal to obs_dim
        if not ((isinstance(element, list) or isinstance(element, np.ndarray)) and len(element) == obs_dim):
            return False

    return True


def check_reward_dimensions(reward, n_agent):
    # Check if reward is a list and has length equal to n_agent
    if not isinstance(reward, list) or len(reward) != n_agent:
        return False

    # Check each element in the reward list
    for element in reward:
        # Each element must be a float
        if not isinstance(element, float):
            return False

    return True


def check_terminated_dimensions(terminated, n_agent):
    # Check if terminated is a list and has length equal to n_agent
    if not isinstance(terminated, list) or len(terminated) != n_agent:
        return False

    # Check each element in the terminated list
    for element in terminated:
        # Each element must be a boolean
        if not isinstance(element, bool):
            return False

    return True


def check_info_dimensions(info, n_agent):
    # Check if info is a list and has length equal to n_agent
    if not isinstance(info, list) or len(info) != n_agent:
        return False

    # Check each element in the info list
    for element in info:
        # Each element must be a dictionary
        if not isinstance(element, dict):
            return False

    return True