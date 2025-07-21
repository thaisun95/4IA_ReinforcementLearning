import numpy as np
import random

def sarsa(
    env, episodes=1000, alpha=0.1, gamma=0.99,
    epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
    verbose=False
):
    """
    SARSA (on-policy TD control) algorithm. Harmonized version.
    - Compatible with both tabular and blackbox environments.
    - Handles both "env.action_space/env.get_valid_actions/state_to_index" (tabular)
      and "env.available_actions()/env.state_id()" (blackbox) APIs.
    Returns: Q-table, rewards list, episode lengths.
    """
    # Detect API style
    is_blackbox = hasattr(env, "available_actions") and hasattr(env, "state_id")
    if is_blackbox:
        n_states = env.num_states()
        n_actions = env.num_actions()
        get_state = lambda: env.state_id()
        get_valid_actions = lambda s=None: env.available_actions()
    else:
        n_states = env.n_states
        n_actions = len(env.action_space)
        get_state = lambda: env.state_to_index(env.state) if hasattr(env, "state") else None
        get_valid_actions = lambda s: env.get_valid_actions(s)

    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        # Init episode
        if is_blackbox:
            env.reset()
            state = get_state()
            valid_actions = get_valid_actions()
        else:
            state_obj = env.reset()
            state = env.state_to_index(state_obj)
            valid_actions = get_valid_actions(state_obj)

        total_reward, steps = 0, 0

        # Initial action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            q_valid = [Q[state, a] for a in valid_actions]
            max_q = max(q_valid)
            best = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
            action = random.choice(best)

        done = False
        while not done:
            # Step (note: reward=env.score() on last step in blackbox)
            if is_blackbox:
                env.step(action)
                next_state = get_state()
                done = env.is_game_over()
                reward = env.score() if done else 0.0
                next_valid_actions = get_valid_actions() if not done else []
            else:
                next_state_obj, reward, done = env.step(action)
                next_state = env.state_to_index(next_state_obj)
                next_valid_actions = get_valid_actions(next_state_obj) if not done else []

            # Next action (on-policy, epsilon-greedy)
            if not done and next_valid_actions:
                if random.random() < epsilon:
                    next_action = random.choice(next_valid_actions)
                else:
                    q_valid_next = [Q[next_state, a] for a in next_valid_actions]
                    max_q_next = max(q_valid_next)
                    best_next = [a for a, q in zip(next_valid_actions, q_valid_next) if q == max_q_next]
                    next_action = random.choice(best_next)
            else:
                next_action = None

            # SARSA update
            if next_action is not None:
                Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            else:
                Q[state, action] += alpha * (reward - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (episode + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_r:.3f}, Epsilon: {epsilon:.3f}")

    return Q, episode_rewards, episode_lengths

def get_policy_from_q(Q, env):
    """
    Extract greedy policy from Q-table.
    - For tabular envs: env.action_space
    - For blackbox envs: just argmax over actions (actions indices!)
    Returns: np.ndarray of shape (n_states,)
    """
    is_blackbox = hasattr(env, "available_actions") and hasattr(env, "state_id")
    n_states = Q.shape[0]
    policy = np.zeros(n_states, dtype=int)
    for state_idx in range(n_states):
        if is_blackbox:
            # On ne connaît pas la vraie validité des actions par état, on suppose toutes valides
            policy[state_idx] = int(np.argmax(Q[state_idx]))
        else:
            state_obj = env.index_to_state(state_idx)
            if hasattr(env, "is_terminal") and env.is_terminal(state_obj):
                policy[state_idx] = -1  # No action
            else:
                policy[state_idx] = env.action_space[int(np.argmax(Q[state_idx]))]
    return policy

def evaluate_policy(env, policy, episodes=100):
    """
    Evaluate a policy by running multiple episodes.
    """
    is_blackbox = hasattr(env, "available_actions") and hasattr(env, "state_id")
    total_rewards = []
    total_lengths = []
    for _ in range(episodes):
        if is_blackbox:
            env.reset()
            state = env.state_id()
            done = env.is_game_over()
        else:
            state_obj = env.reset()
            state = env.state_to_index(state_obj)
            done = env.is_terminal(state_obj)
        total_reward, steps = 0, 0
        while not done:
            action = policy[state]
            if is_blackbox:
                env.step(action)
                done = env.is_game_over()
                reward = env.score() if done else 0.0
                state = env.state_id()
            else:
                next_state_obj, reward, done = env.step(action)
                state = env.state_to_index(next_state_obj)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        total_lengths.append(steps)
    return np.mean(total_rewards), np.mean(total_lengths)