import numpy as np
import random

def expected_sarsa_blackbox(
    env,
    episodes=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    verbose=True,
):
    """
    Expected SARSA for blackbox environments (SecretEnv).
    - The agent only relies on env.state_id(), env.available_actions(), env.step(), env.is_game_over().
    Returns: Q, episode_rewards, episode_lengths
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

        while not done:
            # Epsilon-greedy on available actions
            actions = env.available_actions()
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_vals = [Q[state, a] for a in actions]
                max_q = max(q_vals)
                best_actions = [a for a in actions if Q[state, a] == max_q]
                action = random.choice(best_actions)

            env.step(action)
            next_state = env.state_id()
            done = env.is_game_over()
            reward = env.score() if done else 0.0
            total_reward += reward

            # Compute expected value for next state using current epsilon-greedy
            if not done:
                next_actions = env.available_actions()
                n_valid = len(next_actions)
                q_next = [Q[next_state, a] for a in next_actions]
                max_next_q = max(q_next)
                n_max = sum([q == max_next_q for q in q_next])
                expected_value = 0
                for i, a in enumerate(next_actions):
                    if q_next[i] == max_next_q:
                        prob = (1 - epsilon) / n_max + epsilon / n_valid
                    else:
                        prob = epsilon / n_valid
                    expected_value += prob * Q[next_state, a]
                target = reward + gamma * expected_value
            else:
                target = reward

            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.3f}, Epsilon: {epsilon:.3f}")

    return Q, episode_rewards, episode_lengths