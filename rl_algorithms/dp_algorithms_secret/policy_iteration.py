import numpy as np
import random

def policy_iteration_blackbox(env, gamma=0.99, num_episodes=5000, max_iterations=20, theta=1e-6, verbose=True):
    """
    Approximate Policy Iteration for blackbox envs (like SecretEnv).
    - Policy evaluation/improvement by sampling episodes.
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    policy = np.random.randint(n_actions, size=n_states)
    V = np.zeros(n_states)
    
    for it in range(max_iterations):
        # Policy Evaluation: Monte Carlo estimate of value function under current policy
        returns_sum = np.zeros(n_states)
        returns_count = np.zeros(n_states)
        for ep in range(num_episodes):
            env.reset()
            episode = []
            done = False
            while not done:
                state = env.state_id()
                actions = env.available_actions()
                if len(actions) == 0:
                    break
                a = policy[state]
                if a not in actions:
                    a = random.choice(actions)
                env.step(a)
                reward = env.score() if env.is_game_over() else 0.0
                episode.append((state, a, reward))
                done = env.is_game_over()
            G = 0
            for t in reversed(range(len(episode))):
                state, a, r = episode[t]
                G = gamma * G + r
                returns_sum[state] += G
                returns_count[state] += 1
        # Average returns for value function
        for s in range(n_states):
            if returns_count[s] > 0:
                V[s] = returns_sum[s] / returns_count[s]

        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            env.reset()
            # Go to state s (approximate, not always possible)
            # For blackbox: just use value estimates and choose best among available actions
            env.reset()
            actions = env.available_actions()
            if len(actions) == 0:
                continue
            action_values = []
            for a in actions:
                env.reset()
                env.step(a)
                s_prime = env.state_id()
                reward = env.score() if env.is_game_over() else 0.0
                val = reward + gamma * V[s_prime]
                action_values.append(val)
            best_action = actions[np.argmax(action_values)]
            if policy[s] != best_action:
                policy_stable = False
            policy[s] = best_action
        if verbose:
            print(f"Policy iteration round {it+1} done")
        if policy_stable:
            print(f"Policy converged after {it+1} rounds.")
            break
    return policy, V