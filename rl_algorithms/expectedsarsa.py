import numpy as np
import random

def expected_sarsa(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
    """
    Expected SARSA algorithm implementation.
    """
    n_states = env.n_states
    n_actions = len(env.action_space)
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    episode_lengths = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        while not env.is_terminal(state):
            valid_actions = env.get_valid_actions(state)
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_idx = env.state_to_index(state)
                q_valid = [Q[state_idx, env.action_space.index(a)] for a in valid_actions]
                max_q = np.max(q_valid)
                best_actions = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
                action = random.choice(best_actions)
            # Contrôle de validité de l'action
            if action not in env.get_valid_actions(state):
                raise ValueError(f"Action {action} is not valid for state {state} (valid: {env.get_valid_actions(state)})")
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            state_idx = env.state_to_index(state)
            action_idx = env.action_space.index(action)
            if done:
                target = reward
            else:
                # Calculate expected value of next state using valid actions
                valid_next_actions = env.get_valid_actions(next_state)
                next_q_values = Q[next_state_idx]
                max_q = np.max([next_q_values[env.action_space.index(a)] for a in valid_next_actions])
                n_max_actions = np.sum([next_q_values[env.action_space.index(a)] == max_q for a in valid_next_actions])
                expected_value = 0
                for a in valid_next_actions:
                    a_idx = env.action_space.index(a)
                    if next_q_values[a_idx] == max_q:
                        prob = (1 - epsilon) / n_max_actions + epsilon / len(valid_next_actions)
                    else:
                        prob = epsilon / len(valid_next_actions)
                    expected_value += prob * next_q_values[a_idx]
                target = reward + gamma * expected_value
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])
            state = next_state
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.3f}, Epsilon: {epsilon:.3f}")
    return Q, episode_rewards, episode_lengths

def expected_sarsa_softmax(env, episodes=1000, alpha=0.1, gamma=0.99, temperature=1.0, temperature_decay=0.995, min_temperature=0.1):
    """
    Expected SARSA with softmax action selection instead of epsilon-greedy.
    """
    n_states = env.n_states
    n_actions = len(env.action_space)
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    episode_lengths = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        while not env.is_terminal(state):
            valid_actions = env.get_valid_actions(state)
            state_idx = env.state_to_index(state)
            q_valid = [Q[state_idx, env.action_space.index(a)] for a in valid_actions]
            action = softmax_policy(q_valid, temperature, valid_actions)
            # Contrôle de validité de l'action
            if action not in env.get_valid_actions(state):
                raise ValueError(f"Action {action} is not valid for state {state} (valid: {env.get_valid_actions(state)})")
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            action_idx = env.action_space.index(action)
            if done:
                target = reward
            else:
                # Calculate expected value using softmax policy
                valid_next_actions = env.get_valid_actions(next_state)
                next_q_values = Q[next_state_idx]
                expected_value = 0
                for a in valid_next_actions:
                    a_idx = env.action_space.index(a)
                    prob = softmax_probability(next_q_values, a_idx, temperature, valid_next_actions)
                    expected_value += prob * next_q_values[a_idx]
                target = reward + gamma * expected_value
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])
            state = next_state
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        temperature = max(min_temperature, temperature * temperature_decay)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.3f}, Temperature: {temperature:.3f}")
    return Q, episode_rewards, episode_lengths

def softmax_policy(q_values, temperature, valid_actions):
    """
    Softmax action selection for valid actions.
    """
    if temperature == 0:
        return valid_actions[np.argmax(q_values)]
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(valid_actions, p=probs)

def softmax_probability(q_values, action_idx, temperature, valid_actions):
    """
    Calculate softmax probability for a specific action.
    """
    if temperature == 0:
        return 1.0 if action_idx == np.argmax([q_values[env.action_space.index(a)] for a in valid_actions]) else 0.0
    valid_q = [q_values[env.action_space.index(a)] for a in valid_actions]
    exp_q = np.exp(valid_q / temperature)
    probs = exp_q / np.sum(exp_q)
    action_pos = valid_actions.index(env.action_space[action_idx])
    return probs[action_pos] if action_idx in [env.action_space.index(a) for a in valid_actions] else 0.0

def get_policy_from_q(Q, env):
    """
    Extract greedy policy from Q-table.
    """
    policy = np.zeros(env.n_states, dtype=int)
    for state_idx in range(env.n_states):
        if env.is_terminal(env.index_to_state(state_idx)):
            policy[state_idx] = -1  # No action for terminal states
        else:
            policy[state_idx] = env.action_space[np.argmax(Q[state_idx])]
    return policy

def get_softmax_policy_from_q(Q, env, temperature=1.0):
    """
    Extract softmax policy from Q-table.
    """
    policy = np.zeros(env.n_states, dtype=int)
    for state_idx in range(env.n_states):
        if env.is_terminal(env.index_to_state(state_idx)):
            policy[state_idx] = -1  # No action for terminal states
        else:
            valid_actions = env.get_valid_actions(env.index_to_state(state_idx))
            q_valid = [Q[state_idx, env.action_space.index(a)] for a in valid_actions]
            policy[state_idx] = softmax_policy(q_valid, temperature, valid_actions)
    return policy

def evaluate_policy(env, policy, episodes=100, max_steps=100):
    """
    Evaluate a policy by running multiple episodes.
    
    Args:
        env: Environment object
        policy (np.ndarray): Policy to evaluate
        episodes (int): Number of evaluation episodes
        max_steps (int): Maximum number of steps per episode to avoid infinite loops
    
    Returns:
        avg_reward (float): Average reward per episode
        avg_length (float): Average episode length
    """
    total_rewards = []
    total_lengths = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        while not env.is_terminal(state) and steps < max_steps:
            state_idx = env.state_to_index(state)
            action = policy[state_idx]
            if action == -1:  # Terminal state
                break
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        total_lengths.append(steps)
    return np.mean(total_rewards), np.mean(total_lengths)

