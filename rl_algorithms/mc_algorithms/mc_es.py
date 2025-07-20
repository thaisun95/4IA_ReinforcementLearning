import numpy as np
import random

def mc_es(env, num_episodes=10000, gamma=0.99, max_steps_per_episode=None, verbose=False):
    """
    Monte Carlo Exploring Starts (ES) Control.
    Compatible with state-dependent action spaces.
    """
    n_states = env.n_states
    # 1. Compute all possible action indices for array dimensions
    all_actions = set()
    for idx in range(n_states):
        state = env.index_to_state(idx)
        all_actions.update(env.get_valid_actions(state))
    n_actions = max(all_actions) + 1 if all_actions else 1

    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    if max_steps_per_episode is None:
        max_steps_per_episode = n_states * 2

    valid_episodes = 0
    for episode in range(num_episodes):
        # Exploring starts: pick a non-terminal state and valid action
        non_terminal_states = [i for i in range(n_states) if not env.is_terminal(env.index_to_state(i))]
        state_idx = random.choice(non_terminal_states)
        state = env.index_to_state(state_idx)
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            continue  # skip states with no valid actions (should not happen)
        action = random.choice(valid_actions)
        episode_list = []

        if hasattr(env, "state"):
            env.state = state

        # Start the episode from (state, action)
        next_state, reward, done = env.simulate_step(state, action)
        s_idx = env.state_to_index(state)
        episode_list.append((s_idx, action, reward))
        s_curr = next_state
        steps = 0

        while not env.is_terminal(s_curr) and steps < max_steps_per_episode:
            s_idx = env.state_to_index(s_curr)
            valid_actions = env.get_valid_actions(s_curr)
            if not valid_actions:
                break
            a = policy[s_idx]
            if a not in valid_actions:
                a = random.choice(valid_actions)
            next_state, reward, done = env.simulate_step(s_curr, a)
            episode_list.append((s_idx, a, reward))
            s_curr = next_state
            steps += 1

        if env.is_terminal(s_curr):
            valid_episodes += 1
            G = 0
            visited = set()
            for t in reversed(range(len(episode_list))):
                s_idx, a, r = episode_list[t]
                G = gamma * G + r
                if (s_idx, a) not in visited:
                    N[s_idx, a] += 1
                    Q[s_idx, a] += (G - Q[s_idx, a]) / N[s_idx, a]
                    # Update greedy policy among valid actions
                    valid_acts = env.get_valid_actions(env.index_to_state(s_idx))
                    if valid_acts:
                        best_a = valid_acts[np.argmax([Q[s_idx, x] for x in valid_acts])]
                        policy[s_idx] = best_a
                    visited.add((s_idx, a))
    if verbose:
        print(f"Valid episodes (terminated): {valid_episodes}/{num_episodes}")
    return Q, policy