import numpy as np
import random

def q_learning_blackbox(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, verbose=False):
    """
    Q-Learning algorithm for blackbox/opaque environments (e.g. SecretEnvX).
    Assumes env exposes:
        - reset()
        - step(action)
        - is_game_over()
        - state_id()
        - available_actions()
        - score() (returns reward, only valid if is_game_over else 0)
        - num_states(), num_actions()
    Returns:
        Q (np.ndarray): Q-table of shape [num_states, num_actions]
        episode_rewards (list): List of total rewards per episode
        episode_lengths (list): List of episode lengths
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    Q = np.zeros((n_states, n_actions))

    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        env.reset()
        total_reward = 0
        steps = 0
        state = env.state_id()
        done = env.is_game_over()

        while not done:
            valid_actions = env.available_actions()
            if len(valid_actions) == 0:
                break
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                # Only consider valid actions for selection
                q_valid = [Q[state, a] for a in valid_actions]
                max_q = max(q_valid)
                best_actions = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
                action = random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()
            done = env.is_game_over()
            reward = env.score() if done else 0.0

            # Q-learning update
            if done:
                target = reward
            else:
                next_valid_actions = env.available_actions()
                if len(next_valid_actions) == 0:
                    max_next_q = 0.0
                else:
                    max_next_q = max([Q[next_state, a] for a in next_valid_actions])
                target = reward + gamma * max_next_q

            Q[state, action] += alpha * (target - Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.3f}, Epsilon: {epsilon:.3f}")

    return Q, episode_rewards, episode_lengths