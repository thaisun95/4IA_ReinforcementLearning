import numpy as np
import random

def mc_on_policy(env, num_episodes=10000, gamma=0.99, epsilon=0.1, max_steps_per_episode=None, verbose=False):
    """
    On-policy first-visit MC control for ε-soft policies (handles state-dependent actions).
    Returns Q-table and final greedy policy (array of best action per state).
    """
    n_states = env.n_states
    all_actions = set()
    for idx in range(n_states):
        state = env.index_to_state(idx)
        all_actions.update(env.get_valid_actions(state))
    n_actions = max(all_actions) + 1 if all_actions else 1

    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    # For each state, initialize uniform ε-soft policy
    policy_soft = np.ones((n_states, n_actions)) / n_actions

    if max_steps_per_episode is None:
        max_steps_per_episode = n_states * 2

    valid_episodes = 0
    for episode in range(num_episodes):
        s = env.reset()
        episode_list = []
        steps = 0
        done = False
        while not done and steps < max_steps_per_episode:
            s_idx = env.state_to_index(s)
            valid_actions = env.get_valid_actions(s)
            if not valid_actions:
                break
            # ε-soft policy sampling
            if random.random() < epsilon:
                a = random.choice(valid_actions)
            else:
                q_valid = [Q[s_idx, act] for act in valid_actions]
                a = valid_actions[np.argmax(q_valid)]
            next_s, reward, done = env.simulate_step(s, a)
            episode_list.append((s_idx, a, reward))
            s = next_s
            steps += 1

        if env.is_terminal(s):
            valid_episodes += 1
            G = 0
            visited = set()
            for t in reversed(range(len(episode_list))):
                s_idx, a, r = episode_list[t]
                G = gamma * G + r
                if (s_idx, a) not in visited:
                    N[s_idx, a] += 1
                    Q[s_idx, a] += (G - Q[s_idx, a]) / N[s_idx, a]
                    # Update ε-soft policy only among valid actions
                    valid_acts = env.get_valid_actions(env.index_to_state(s_idx))
                    if valid_acts:
                        best_a = valid_acts[np.argmax([Q[s_idx, x] for x in valid_acts])]
                        for act in range(n_actions):
                            if act in valid_acts:
                                if act == best_a:
                                    policy_soft[s_idx, act] = 1 - epsilon + epsilon / len(valid_acts)
                                else:
                                    policy_soft[s_idx, act] = epsilon / len(valid_acts)
                            else:
                                policy_soft[s_idx, act] = 0
                    visited.add((s_idx, a))
    if verbose:
        print(f"Valid episodes (terminated): {valid_episodes}/{num_episodes}")
    # Extract final greedy policy for visualization
    policy_greedy = extract_greedy_policy(Q, env)
    return Q, policy_greedy

def extract_greedy_policy(Q, env):
    """Extract greedy policy from Q-table for given env (handles variable action sets)."""
    n_states, n_actions = Q.shape
    policy = np.zeros(n_states, dtype=int)
    for idx in range(n_states):
        state = env.index_to_state(idx)
        valid_acts = env.get_valid_actions(state)
        if valid_acts:
            q_valid = [Q[idx, act] for act in valid_acts]
            best_a = valid_acts[np.argmax(q_valid)]
            policy[idx] = best_a
        else:
            policy[idx] = -1
    return policy