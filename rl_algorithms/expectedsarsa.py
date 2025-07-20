import numpy as np
import random

def expected_sarsa(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
    """
    Expected SARSA algorithm implementation.
    
    Expected SARSA uses the expected value of the next state-action pair instead of a single sample,
    which can reduce variance and improve learning stability.
    
    Args:
        env: Environment object with methods:
            - reset()
            - step(action)
            - n_states (property)
            - action_space (list)
            - is_terminal(state)
        episodes (int): Number of episodes to train
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate
        epsilon_decay (float): Decay rate for epsilon
        min_epsilon (float): Minimum epsilon value
    
    Returns:
        Q (np.ndarray): Q-table of shape [n_states, n_actions]
        episode_rewards (list): List of total rewards per episode
        episode_lengths (list): List of episode lengths
    """
    n_states = env.n_states
    n_actions = len(env.action_space)
    
    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.is_terminal(state):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                state_idx = env.state_to_index(state)
                action = env.action_space[np.argmax(Q[state_idx])]
            
            # Take action and observe result
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            state_idx = env.state_to_index(state)
            action_idx = env.action_space.index(action)
            
            # Expected SARSA update rule
            if done:
                target = reward
            else:
                # Calculate expected value of next state
                next_q_values = Q[next_state_idx]
                max_q = np.max(next_q_values)
                
                # Count number of actions with maximum Q-value
                n_max_actions = np.sum(next_q_values == max_q)
                
                # Calculate expected value using epsilon-greedy policy
                expected_value = 0
                for a_idx, a in enumerate(env.action_space):
                    if next_q_values[a_idx] == max_q:
                        # Greedy action
                        prob = (1 - epsilon) / n_max_actions + epsilon / n_actions
                    else:
                        # Non-greedy action
                        prob = epsilon / n_actions
                    expected_value += prob * next_q_values[a_idx]
                
                target = reward + gamma * expected_value
            
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.3f}, Epsilon: {epsilon:.3f}")
    
    return Q, episode_rewards, episode_lengths

def expected_sarsa_softmax(env, episodes=1000, alpha=0.1, gamma=0.99, temperature=1.0, temperature_decay=0.995, min_temperature=0.1):
    """
    Expected SARSA with softmax action selection instead of epsilon-greedy.
    
    Args:
        env: Environment object with methods:
            - reset()
            - step(action)
            - n_states (property)
            - action_space (list)
            - is_terminal(state)
        episodes (int): Number of episodes to train
        alpha (float): Learning rate
        gamma (float): Discount factor
        temperature (float): Temperature parameter for softmax
        temperature_decay (float): Decay rate for temperature
        min_temperature (float): Minimum temperature value
    
    Returns:
        Q (np.ndarray): Q-table of shape [n_states, n_actions]
        episode_rewards (list): List of total rewards per episode
        episode_lengths (list): List of episode lengths
    """
    n_states = env.n_states
    n_actions = len(env.action_space)
    
    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))
    
    episode_rewards = []
    episode_lengths = []
    
    def softmax_policy(q_values, temp):
        """Compute softmax probabilities for action selection."""
        exp_q = np.exp(q_values / temp)
        return exp_q / np.sum(exp_q)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.is_terminal(state):
            # Softmax action selection
            state_idx = env.state_to_index(state)
            action_probs = softmax_policy(Q[state_idx], temperature)
            action = np.random.choice(env.action_space, p=action_probs)
            
            # Take action and observe result
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            action_idx = env.action_space.index(action)
            
            # Expected SARSA update rule with softmax
            if done:
                target = reward
            else:
                # Calculate expected value using softmax policy
                next_action_probs = softmax_policy(Q[next_state_idx], temperature)
                expected_value = np.sum(next_action_probs * Q[next_state_idx])
                target = reward + gamma * expected_value
            
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Decay temperature
        temperature = max(min_temperature, temperature * temperature_decay)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.3f}, Temperature: {temperature:.3f}")
    
    return Q, episode_rewards, episode_lengths

def get_policy_from_q(Q, env):
    """
    Extract greedy policy from Q-table.
    
    Args:
        Q (np.ndarray): Q-table
        env: Environment object
    
    Returns:
        policy (np.ndarray): Greedy policy (best action for each state)
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
    
    Args:
        Q (np.ndarray): Q-table
        env: Environment object
        temperature (float): Temperature parameter
    
    Returns:
        policy (np.ndarray): Softmax policy probabilities for each state
    """
    policy = np.zeros((env.n_states, len(env.action_space)))
    for state_idx in range(env.n_states):
        if env.is_terminal(env.index_to_state(state_idx)):
            policy[state_idx] = 0  # No action for terminal states
        else:
            exp_q = np.exp(Q[state_idx] / temperature)
            policy[state_idx] = exp_q / np.sum(exp_q)
    return policy

def evaluate_policy(env, policy, episodes=100):
    """
    Evaluate a policy by running multiple episodes.
    
    Args:
        env: Environment object
        policy (np.ndarray): Policy to evaluate (can be greedy or softmax)
        episodes (int): Number of evaluation episodes
    
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
        
        while not env.is_terminal(state):
            state_idx = env.state_to_index(state)
            
            if len(policy.shape) == 1:
                # Greedy policy
                action = policy[state_idx]
                if action == -1:  # Terminal state
                    break
            else:
                # Softmax policy
                action_probs = policy[state_idx]
                action = np.random.choice(env.action_space, p=action_probs)
                
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
        total_lengths.append(steps)
    
    return np.mean(total_rewards), np.mean(total_lengths)

