import numpy as np

def create_random_policy(num_agents, action_dim, min_action, max_action):
    """
    Create a random policy for multi-agent RL.

    Args:
    - num_agents: Number of agents.
    - action_dim: Dimension of each agent's action space.
    - min_action: Minimum action value.
    - max_action: Maximum action value.

    Returns:
    - List of NumPy arrays representing the policy for each agent.
    """
    policies = []
    for _ in range(num_agents):
        policy = np.random.randint(min_action, max_action + 1, size=action_dim)
        policies.append(policy)

    return policies


if __name__ == "__main__":
    # Example usage:
    num_agents = 4
    action_dim = 2
    min_action = -20
    max_action = 20

    random_policy = create_random_policy(num_agents, action_dim, min_action, max_action)
    # print(random_policy)