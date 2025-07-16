import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000):
    """
    Perform Value Iteration for a given environment.

    Args:
        env: The environment object (must have reset(), step(), action_space, etc).
        gamma (float): Discount factor.
        theta (float): Convergence threshold.
        max_iterations (int): Maximum number of iterations.

    Returns:
        V (np.ndarray): Optimal value function.
        policy (np.ndarray): Optimal policy.
    """
    n_states = env.size
    n_actions = len(env.action_space)
    V = np.zeros(n_states)

    for i in range(max_iterations):
        delta = 0
        for s in range(n_states):
            if env.is_terminal(s):
                continue
            v = V[s]
            # For each action, calculate expected value
            action_values = []
            for a in env.action_space:
                # Simulate the action
                if a == 0:
                    next_state = max(s - 1, 0)
                else:
                    next_state = min(s + 1, n_states - 1)
                reward = env.get_reward(next_state)
                action_values.append(reward + gamma * V[next_state])
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            print(f"Value iteration converged after {i+1} iterations.")
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if env.is_terminal(s):
            policy[s] = -1  # No action
            continue
        action_values = []
        for a in env.action_space:
            if a == 0:
                next_state = max(s - 1, 0)
            else:
                next_state = min(s + 1, n_states - 1)
            reward = env.get_reward(next_state)
            action_values.append(reward + gamma * V[next_state])
        policy[s] = np.argmax(action_values)
    return V, policy