import random
from collections import defaultdict
import numpy as np

class DynaQBlackboxAgent:
    """
    Dyna-Q algorithm adapted for blackbox/gym-like environments.
    Assumes environment exposes: reset(), step(a), available_actions(), is_game_over(), state_id(), score().
    """
    def __init__(self, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=50):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        
        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        
        # Model: Model[state][action] = (reward, next_state)
        self.Model = defaultdict(lambda: dict())
        
        # Set of visited (state, action) pairs for planning
        self.visited_state_actions = set()
    
    def select_action(self, state, valid_actions=None):
        """Epsilon-greedy policy"""
        if valid_actions is None or len(valid_actions) == 0:
            valid_actions = list(range(self.n_actions))
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = [self.Q[state][a] for a in valid_actions]
        max_q = max(q_vals)
        best_acts = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return random.choice(best_acts)
    
    def update_q(self, state, action, reward, next_state, valid_next_actions=None):
        """Q-learning update step"""
        if valid_next_actions is None or len(valid_next_actions) == 0:
            max_next_q = 0.0
        else:
            max_next_q = max([self.Q[next_state][a] for a in valid_next_actions])
        td_target = reward + self.gamma * max_next_q
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])
    
    def update_model(self, state, action, reward, next_state):
        """Model update for planning"""
        self.Model[state][action] = (reward, next_state)
        self.visited_state_actions.add((state, action))
    
    def planning_step(self, get_valid_actions_fn=None):
        """One planning step using the model"""
        if not self.visited_state_actions:
            return
        state, action = random.choice(list(self.visited_state_actions))
        if action in self.Model[state]:
            reward, next_state = self.Model[state][action]
            if get_valid_actions_fn:
                valid_next_actions = get_valid_actions_fn(next_state)
            else:
                valid_next_actions = list(range(self.n_actions))
            self.update_q(state, action, reward, next_state, valid_next_actions)
    
    def learn_step(self, state, action, reward, next_state, valid_next_actions=None, get_valid_actions_fn=None):
        """Full Dyna-Q update: direct RL, model update, and planning"""
        self.update_q(state, action, reward, next_state, valid_next_actions)
        self.update_model(state, action, reward, next_state)
        for _ in range(self.n_planning):
            self.planning_step(get_valid_actions_fn)