import numpy as np
import random

def mc_off_policy(env, num_episodes=10000, gamma=0.99, max_steps_per_episode=None, verbose=False):
    """
    Off-policy MC control using weighted importance sampling (state-dependent actions).
    Returns:
        Q (np.ndarray): Action-value function.
        policy (np.ndarray): Deterministic optimal policy.
    """
    n_states = env.n_states
    all_actions = set()
    for idx in range(n_states):
        state = env.index_to_state(idx)
        all_actions.update(env.get_valid_actions(state))
    n_actions = max(all_actions) + 1 if all_actions else 1

    Q = np.zeros((n_states, n_actions))
    C = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    if max_steps_per_episode is None:
        max_steps_per_episode = n_states * 2

    valid_episodes = 0
    for episode in range(num_episodes):
        s = env.reset()
        episode_list = []
        steps = 0
        done = False
        # Behavior policy: uniform random among valid actions
        while not done and steps < max_steps_per_episode:
            s_idx = env.state_to_index(s)
            valid_actions = env.get_valid_actions(s)
            if not valid_actions:
                break
            a = random.choice(valid_actions)
            next_s, reward, done = env.simulate_step(s, a)
            episode_list.append((s_idx, a, reward))
            s = next_s
            steps += 1

        if env.is_terminal(s):
            valid_episodes += 1
            G = 0
            W = 1
            for t in reversed(range(len(episode_list))):
                s_idx, a, r = episode_list[t]
                G = gamma * G + r
                C[s_idx, a] += W
                Q[s_idx, a] += (W / C[s_idx, a]) * (G - Q[s_idx, a])
                # Improve target policy greedily over valid actions
                valid_acts = env.get_valid_actions(env.index_to_state(s_idx))
                if valid_acts:
                    best_a = valid_acts[np.argmax([Q[s_idx, x] for x in valid_acts])]
                    policy[s_idx] = best_a
                # If action taken isn't the greedy action, stop (importance weight = 0)
                if a != policy[s_idx]:
                    break
                W = W / (1.0 / len(valid_acts))  # because behavior policy is uniform among valid
    if verbose:
        print(f"Valid episodes (terminated): {valid_episodes}/{num_episodes}")
    return Q, policy