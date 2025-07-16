import numpy as np

def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    """
    Perform Policy Iteration for a generic environment with state-index mapping.

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
        max_iterations (int): Maximum number of policy improvement steps.

    Returns:
        policy (np.ndarray): Optimal policy (best action for each state index, -1 for terminal).
        V (np.ndarray): Optimal value function (array of shape [n_states]).
    """
    policy = np.random.choice(env.action_space, size=env.n_states)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if env.is_terminal(state):
            policy[idx] = -1  # No action for terminal states

    V = np.zeros(env.n_states)

    for it in range(max_iterations):
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for idx in range(env.n_states):
                state = env.index_to_state(idx)
                if env.is_terminal(state):
                    continue
                v = V[idx]
                a = policy[idx]
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                V[idx] = reward + gamma * V[next_idx]
                delta = max(delta, abs(v - V[idx]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
            old_action = policy[idx]
            action_values = []
            for a in env.action_space:
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                action_values.append(reward + gamma * V[next_idx])
            best_action = np.argmax(action_values)
            policy[idx] = env.action_space[best_action]
            if old_action != policy[idx]:
                policy_stable = False
        if policy_stable:
            print(f"Policy iteration converged after {it+1} iterations.")
            break

    return policy, V