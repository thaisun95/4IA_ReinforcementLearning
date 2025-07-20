import random
from collections import defaultdict

class DynaQAgent:
    """Algorithme Dyna-Q selon Sutton & Barto"""
    
    def __init__(self, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=50):
        self.n_actions = n_actions
        self.alpha = alpha          # Taux d'apprentissage
        self.gamma = gamma          # Facteur de discount
        self.epsilon = epsilon      # Exploration ε-greedy
        self.n_planning = n_planning # Étapes de planification
        
        # Table Q : Q[état][action] = valeur
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Modèle : Model[état][action] = (récompense, nouvel_état)
        self.Model = defaultdict(lambda: defaultdict(lambda: None))
        
        # Paires (état, action) visitées pour planification
        self.visited_state_actions = set()
    
    def select_action(self, state, valid_actions=None):
        """Politique ε-greedy"""
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.Q[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q(self, state, action, reward, next_state, valid_next_actions=None):
        """Mise à jour Q-learning"""
        if valid_next_actions is None:
            valid_next_actions = list(range(self.n_actions))
            
        if valid_next_actions:
            max_next_q = max([self.Q[next_state][a] for a in valid_next_actions])
        else:
            max_next_q = 0.0
            
        # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        current_q = self.Q[state][action]
        td_target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (td_target - current_q)
    
    def update_model(self, state, action, reward, next_state):
        """Mise à jour du modèle"""
        self.Model[state][action] = (reward, next_state)
        self.visited_state_actions.add((state, action))
    
    def planning_step(self, get_valid_actions_fn=None):
        """Une étape de planification"""
        if not self.visited_state_actions:
            return
            
        # Échantillonner (état, action) aléatoirement
        state, action = random.choice(list(self.visited_state_actions))
        
        if self.Model[state][action] is not None:
            reward, next_state = self.Model[state][action]
            
            # Actions valides dans next_state
            if get_valid_actions_fn:
                valid_next_actions = get_valid_actions_fn(next_state)
            else:
                valid_next_actions = list(range(self.n_actions))
            
            # Mise à jour Q avec expérience simulée
            self.update_q(state, action, reward, next_state, valid_next_actions)
    
    def learn_step(self, state, action, reward, next_state, 
                   valid_next_actions=None, get_valid_actions_fn=None):
        """Étape complète Dyna-Q"""
        
        # 1. APPRENTISSAGE DIRECT
        self.update_q(state, action, reward, next_state, valid_next_actions)
        
        # 2. MISE À JOUR MODÈLE  
        self.update_model(state, action, reward, next_state)
        
        # 3. PLANIFICATION (n étapes)
        for _ in range(self.n_planning):
            self.planning_step(get_valid_actions_fn)