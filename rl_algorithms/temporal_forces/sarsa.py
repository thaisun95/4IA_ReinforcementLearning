import numpy as np
import random

def sarsa(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
    """
    SARSA (State-Action-Reward-State-Action) algorithm implementation.
    
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
        
        # Choose initial action using epsilon-greedy
        valid_actions = env.get_valid_actions(state)
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            state_idx = env.state_to_index(state)
            q_valid = [Q[state_idx, env.action_space.index(a)] for a in valid_actions]
            max_q = np.max(q_valid)
            best_actions = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
            action = random.choice(best_actions)
        
        while not env.is_terminal(state):
            # Contrôle de validité de l'action
            if action not in env.get_valid_actions(state):
                raise ValueError(f"Action {action} is not valid for state {state} (valid: {env.get_valid_actions(state)})")
            # Take action and observe result
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            state_idx = env.state_to_index(state)
            action_idx = env.action_space.index(action)
            
            # Choose next action using epsilon-greedy
            if done:
                next_action = None
                target = reward
            else:
                valid_next_actions = env.get_valid_actions(next_state)
                if random.random() < epsilon:
                    next_action = random.choice(valid_next_actions)
                else:
                    q_valid_next = [Q[next_state_idx, env.action_space.index(a)] for a in valid_next_actions]
                    max_q_next = np.max(q_valid_next)
                    best_next_actions = [a for a, q in zip(valid_next_actions, q_valid_next) if q == max_q_next]
                    next_action = random.choice(best_next_actions)
                next_action_idx = env.action_space.index(next_action)
                target = reward + gamma * Q[next_state_idx, next_action_idx]
            
            # SARSA update rule
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])
            
            state = next_state
            action = next_action
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

def sarsa_lambda(env, episodes=1000, alpha=0.1, gamma=0.99, lambda_param=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
    """
    SARSA(λ) algorithm implementation with eligibility traces.
    
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
        lambda_param (float): Eligibility trace decay parameter
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
        
        # Initialize eligibility traces
        E = np.zeros((n_states, n_actions))
        
        # Choose initial action using epsilon-greedy
        valid_actions = env.get_valid_actions(state)
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            state_idx = env.state_to_index(state)
            q_valid = [Q[state_idx, env.action_space.index(a)] for a in valid_actions]
            max_q = np.max(q_valid)
            best_actions = [a for a, q in zip(valid_actions, q_valid) if q == max_q]
            action = random.choice(best_actions)
        
        while not env.is_terminal(state):
            # Contrôle de validité de l'action
            if action not in env.get_valid_actions(state):
                raise ValueError(f"Action {action} is not valid for state {state} (valid: {env.get_valid_actions(state)})")
            # Take action and observe result
            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            state_idx = env.state_to_index(state)
            action_idx = env.action_space.index(action)
            
            # Choose next action using epsilon-greedy
            if done:
                next_action = None
                target = reward
            else:
                valid_next_actions = env.get_valid_actions(next_state)
                if random.random() < epsilon:
                    next_action = random.choice(valid_next_actions)
                else:
                    q_valid_next = [Q[next_state_idx, env.action_space.index(a)] for a in valid_next_actions]
                    max_q_next = np.max(q_valid_next)
                    best_next_actions = [a for a, q in zip(valid_next_actions, q_valid_next) if q == max_q_next]
                    next_action = random.choice(best_next_actions)
                next_action_idx = env.action_space.index(next_action)
                target = reward + gamma * Q[next_state_idx, next_action_idx]
            
            # Calculate TD error
            td_error = target - Q[state_idx, action_idx]
            
            # Update eligibility traces
            E[state_idx, action_idx] += 1
            
            # Update all state-action pairs
            Q += alpha * td_error * E
            E *= gamma * lambda_param
            
            state = next_state
            action = next_action
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

def evaluate_policy(env, policy, episodes=100):
    """
    Evaluate a policy by running multiple episodes.
    
    Args:
        env: Environment object
        policy (np.ndarray): Policy to evaluate
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
            action = policy[state_idx]
            
            if action == -1:  # Terminal state
                break
                
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
        total_lengths.append(steps)
    
    return np.mean(total_rewards), np.mean(total_lengths)
