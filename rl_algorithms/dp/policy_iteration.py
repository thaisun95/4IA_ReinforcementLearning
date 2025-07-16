import numpy as np

def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    """
    Perform Policy Iteration for a given environment.

    Args:
        env: The environment object (must have reset(), step(), action_space, etc).
        gamma (float): Discount factor.
        theta (float): Convergence threshold.
        max_iterations (int): Maximum number of policy improvement steps.

    Returns:
        policy (np.ndarray): Optimal policy.
        V (np.ndarray): Optimal value function.
    """
    n_states = env.size
    n_actions = len(env.action_space)
    # Random initial policy: always move right (can be random)
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if not env.is_terminal(s):
            policy[s] = np.random.choice(env.action_space)
        else:
            policy[s] = -1

    V = np.zeros(n_states)

    for it in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(n_states):
                if env.is_terminal(s):
                    continue
                v = V[s]
                a = policy[s]
                if a == 0:
                    next_state = max(s - 1, 0)
                else:
                    next_state = min(s + 1, n_states - 1)
                reward = env.get_reward(next_state)
                V[s] = reward + gamma * V[next_state]
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            if env.is_terminal(s):
                continue
            old_action = policy[s]
            action_values = []
            for a in env.action_space:
                if a == 0:
                    next_state = max(s - 1, 0)
                else:
                    next_state = min(s + 1, n_states - 1)
                reward = env.get_reward(next_state)
                action_values.append(reward + gamma * V[next_state])
            best_action = np.argmax(action_values)
            policy[s] = env.action_space[best_action]
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            print(f"Policy iteration converged after {it+1} iterations.")
            break
    return policy, V