import numpy as np
import random

def sarsa_blackbox(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, verbose=False):
    """
    SARSA algorithm for blackbox environments (e.g., SecretEnvX).
    Assumes:
        - env.reset(), env.step(action), env.is_game_over(), env.state_id(), env.available_actions(), env.score()
        - env.num_states(), env.num_actions()
    Returns:
        Q (np.ndarray): Q-table
        episode_rewards (list)
        episode_lengths (list)
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    Q = np.zeros((n_states, n_actions))

    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        env.reset()
        state = env.state_id()
        done = env.is_game_over()
        total_reward = 0
        steps = 0

        # Choose first action epsilon-greedy among valid actions
        valid_actions = env.available_actions()
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            q_valid = [Q[state, a] for a in valid_actions]
            max_q = max(q_valid)
            best_actions = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
            action = random.choice(best_actions)

        while not done:
            env.step(action)
            next_state = env.state_id()
            done = env.is_game_over()
            reward = env.score() if done else 0.0

            next_valid_actions = env.available_actions() if not done else []
            # Select next action
            if not done and next_valid_actions:
                if random.random() < epsilon:
                    next_action = random.choice(next_valid_actions)
                else:
                    q_valid_next = [Q[next_state, a] for a in next_valid_actions]
                    max_q_next = max(q_valid_next)
                    best_next_actions = [a for a, q in zip(next_valid_actions, q_valid_next) if q == max_q_next]
                    next_action = random.choice(best_next_actions)
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

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.3f}, Epsilon: {epsilon:.3f}")

    return Q, episode_rewards, episode_lengths