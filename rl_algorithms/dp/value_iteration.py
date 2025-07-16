import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000):
    """
    Perform Value Iteration for a generic environment with state-index mapping.
    
    Args:
        env: The environment object with methods:
            - n_states (int)
            - action_space (list)
            - index_to_state(idx)
            - state_to_index(state)
            - is_terminal(state)
            - simulate_step(state, action)
        gamma (float): Discount factor.
        theta (float): Convergence threshold.
        max_iterations (int): Maximum number of iterations.
    
    Returns:
        V (np.ndarray): Optimal value function (array of shape [n_states]).
        policy (np.ndarray): Optimal policy (best action for each state index, -1 for terminal).
    """
    V = np.zeros(env.n_states)
    
    for i in range(max_iterations):
        delta = 0
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
            v = V[idx]
            action_values = []
            for a in env.action_space:
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                action_values.append(reward + gamma * V[next_idx])
            V[idx] = max(action_values)
            delta = max(delta, abs(v - V[idx]))
        if delta < theta:
            print(f"Value iteration converged after {i+1} iterations.")
            break

    # Extract policy
    policy = np.zeros(env.n_states, dtype=int)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if env.is_terminal(state):
            policy[idx] = -1  # No action
            continue
        action_values = []
        for a in env.action_space:
            next_state, reward, done = env.simulate_step(state, a)
            next_idx = env.state_to_index(next_state)
            action_values.append(reward + gamma * V[next_idx])
        policy[idx] = np.argmax(action_values)
    return V, policy