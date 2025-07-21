import numpy as np
import random

def q_learning(
    env, episodes=1000, alpha=0.1, gamma=0.99,
    epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
    verbose=False
):
    """
    Q-Learning algorithm (harmonized for variable action spaces).
    Compatible with envs where the set of valid actions depends on the state (e.g., MontyHall).
    """
    n_states = env.n_states
    n_actions = len(env.action_space)
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state_obj = env.reset()
        state = env.state_to_index(state_obj)
        total_reward = 0
        steps = 0

        while not env.is_terminal(state_obj):
            valid_actions = env.get_valid_actions(state_obj)
            # Epsilon-greedy on valid actions only!
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                q_valid = [Q[state, env.action_space.index(a)] for a in valid_actions]
                max_q = max(q_valid)
                best = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
                action = random.choice(best)

            # Interact with env
            next_state_obj, reward, done = env.step(action)
            next_state = env.state_to_index(next_state_obj)
            action_idx = env.action_space.index(action)

            # Q-Learning update: only maximize over valid actions of next_state!
            if not done:
                next_valid_actions = env.get_valid_actions(next_state_obj)
                q_next_valid = [Q[next_state, env.action_space.index(a)] for a in next_valid_actions]
                max_next_q = max(q_next_valid)
                target = reward + gamma * max_next_q
            else:
                target = reward

            Q[state, action_idx] += alpha * (target - Q[state, action_idx])

            state_obj = next_state_obj
            state = next_state
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
    """Extract greedy policy from Q-table (compatible with variable actions)."""
    policy = np.zeros(env.n_states, dtype=int)
    for state_idx in range(env.n_states):
        state_obj = env.index_to_state(state_idx)
        if env.is_terminal(state_obj):
            policy[state_idx] = -1
        else:
            valid_actions = env.get_valid_actions(state_obj)
            q_valid = [Q[state_idx, env.action_space.index(a)] for a in valid_actions]
            max_q = max(q_valid)
            best = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
            policy[state_idx] = random.choice(best)
    return policy

def evaluate_policy(env, policy, episodes=100):
    total_rewards = []
    total_lengths = []
    for _ in range(episodes):
        state_obj = env.reset()
        total_reward = 0
        steps = 0
        while not env.is_terminal(state_obj):
            state_idx = env.state_to_index(state_obj)
            action = policy[state_idx]
            if action == -1:
                break
            state_obj, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        total_lengths.append(steps)
    return np.mean(total_rewards), np.mean(total_lengths)