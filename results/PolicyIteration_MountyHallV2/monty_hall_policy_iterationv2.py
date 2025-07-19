import numpy as np
import random
import pygame
import sys
import pickle
from itertools import combinations
import time

# Policy Iteration Algorithm
def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    """
    Perform Policy Iteration for a generic environment with state-index mapping.
    """
    print(f"Starting Policy Iteration with {env.n_states} states...")
    
    # Set a fixed winning door for deterministic training
    env._winning_door = 0  # Always door 0 for training consistency
    
    # Ensure all states are pre-generated
    env.generate_all_states()
    
    # Initialize policy and value function with correct size
    policy = np.full(env.n_states, -1, dtype=int)  # Default to -1 (no action)
    V = np.zeros(env.n_states)
    
    # Initialize valid policies for non-terminal states
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if not env.is_terminal(state):
            step, doors_remaining, last_chosen = state
            if doors_remaining:  # If there are available doors
                policy[idx] = doors_remaining[0]  # Choose first available door

    for it in range(max_iterations):
        print(f"Policy Iteration - Iteration {it+1}")
        
        # --- Policy Evaluation ---
        eval_iterations = 0
        while True:
            eval_iterations += 1
            delta = 0
            for idx in range(env.n_states):
                state = env.index_to_state(idx)
                if env.is_terminal(state):
                    continue
                    
                v = V[idx]
                a = int(policy[idx])
                
                # Skip if no valid action
                if a == -1:
                    continue
                
                try:
                    next_state, reward, done = env.simulate_step(state, a)
                    next_idx = env.state_to_index(next_state)
                    V[idx] = reward + gamma * V[next_idx]
                except (ValueError, KeyError) as e:
                    # Handle invalid actions by setting a penalty
                    V[idx] = -1.0
                    
                delta = max(delta, abs(v - V[idx]))
            if delta < theta:
                break
        print(f"  Policy Evaluation converged after {eval_iterations} iterations")

        # --- Policy Improvement ---
        policy_stable = True
        changes = 0
            
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
                
            old_action = policy[idx]
            action_values = []
            valid_actions = []
            
            step, doors_remaining, last_chosen = state
            
            # Try all doors that are still available
            for a in doors_remaining:
                try:
                    next_state, reward, done = env.simulate_step(state, a)
                    next_idx = env.state_to_index(next_state)
                    action_values.append(reward + gamma * V[next_idx])
                    valid_actions.append(a)
                except (ValueError, KeyError):
                    # Invalid action, skip
                    continue
            
            if valid_actions:
                best_action_idx = np.argmax(action_values)
                policy[idx] = valid_actions[best_action_idx]
            else:
                # No valid actions, keep -1
                policy[idx] = -1
                
            if old_action != policy[idx]:
                policy_stable = False
                changes += 1
        
        print(f"  Policy changes: {changes}")
        
        if policy_stable:
            print(f"Policy iteration converged after {it+1} iterations.")
            break

    return policy, V

# Monty Hall Level 2 Environment
class MontyHallV2:
    def __init__(self, n_doors=5):
        """
        Monty Hall Level 2 environment (generalized for n doors, default 5).
        """
        self.n_doors = n_doors
        self.action_space = list(range(n_doors))
        
        # Initialize empty states list
        self.states = []
        self.state_to_idx = {}  # Dictionary for faster lookups
        
        # Generate all states
        self.generate_all_states()
        
        self.n_states = len(self.states)
        print(f"Generated {self.n_states} states for {n_doors} doors")
        
        # Initialize game state
        self.state = (0, tuple(range(n_doors)), -1)
        self._winning_door = None
        self._revealed = []

    def generate_all_states(self):
        """Generate ALL possible states systematically"""
        self.states = []
        self.state_to_idx = {}
        
        print("Generating all possible states...")
        
        # Initial state: step 0, all doors available, no choice made
        initial_state = (0, tuple(range(self.n_doors)), -1)
        self._add_state(initial_state)
        
        # Generate states for each step
        for step in range(self.n_doors):
            print(f"  Generating states for step {step+1}/{self.n_doors}")
            
            if step == 0:
                # Step 1: player chooses from all doors
                doors_remaining = tuple(range(self.n_doors))
                for chosen_door in range(self.n_doors):
                    state = (1, doors_remaining, chosen_door)
                    self._add_state(state)
            else:
                # Subsequent steps: Monty removes doors
                doors_count = self.n_doors - step + 1
                
                # Generate all possible combinations of remaining doors
                for doors_remaining in combinations(range(self.n_doors), doors_count):
                    doors_remaining = tuple(sorted(doors_remaining))
                    
                    # For each combination, any door in it could be the chosen one
                    for chosen_door in doors_remaining:
                        state = (step + 1, doors_remaining, chosen_door)
                        self._add_state(state)
        
        # Terminal states (final step)
        for chosen_door in range(self.n_doors):
            for final_door in range(self.n_doors):
                state = (self.n_doors, (final_door,), chosen_door)
                self._add_state(state)
        
        print(f"Total states generated: {len(self.states)}")

    def _add_state(self, state):
        """Add a state if it doesn't already exist"""
        if state not in self.state_to_idx:
            idx = len(self.states)
            self.states.append(state)
            self.state_to_idx[state] = idx

    def state_to_index(self, state):
        """Convert state to index with error handling"""
        if state not in self.state_to_idx:
            raise KeyError(f"State {state} not found in state space")
        return self.state_to_idx[state]

    def index_to_state(self, idx):
        """Convert index to state"""
        if idx >= len(self.states):
            raise IndexError(f"Index {idx} out of bounds for {len(self.states)} states")
        return self.states[idx]

    def reset(self):
        """Reset the environment"""
        self._winning_door = random.choice(range(self.n_doors))
        self.state = (0, tuple(range(self.n_doors)), -1)
        self._revealed = []
        return self.state

    def is_terminal(self, state):
        """Check if state is terminal"""
        step, doors_remaining, chosen = state
        return step == self.n_doors

    def simulate_step(self, state, action):
        """Simulate a step without changing the environment state"""
        step, doors_remaining, last_chosen = state
        
        if self.is_terminal(state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return state, reward, True

        chosen = int(action)
        
        # Validate that the chosen door is available
        if chosen not in doors_remaining:
            raise ValueError(f"Door {chosen} is not available. Available doors: {doors_remaining}")
        
        if step < self.n_doors - 1:
            # Monty removes a door (not chosen, not winning)
            removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
            if len(removable) == 0:
                raise ValueError("No door for Monty to remove!")
            
            # For deterministic training, always remove the first removable door
            door_to_remove = removable[0]  # Deterministic choice
            next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
            next_state = (step + 1, next_doors, chosen)
            
            reward = 0.0
            done = False
        else:
            # Final step - game ends
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
            
        return next_state, reward, done

    def step(self, action):
        """Take a step in the environment"""
        step, doors_remaining, last_chosen = self.state
        
        if self.is_terminal(self.state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return self.state, reward, True

        chosen = int(action)
        
        if step < self.n_doors - 1:
            # Monty removes a door
            removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
            if len(removable) == 0:
                raise ValueError("No door for Monty to remove!")
            
            door_to_remove = random.choice(removable)  # Random for actual gameplay
            self._revealed.append(door_to_remove)
            next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
            next_state = (step + 1, next_doors, chosen)
            reward = 0.0
            done = False
        else:
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
            
        self.state = next_state
        return next_state, reward, done

# Pygame Visualization
class MontyHallVisualizer:
    def __init__(self, env, policy=None, value_function=None):
        pygame.init()
        self.env = env
        self.policy = policy
        self.value_function = value_function
        
        # Screen settings
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Monty Hall Level 2 - Policy Iteration")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (64, 64, 64)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Game state
        self.mode = "manual"  # "manual" or "auto"
        self.game_over = False
        self.message = ""
        
    def draw_door(self, x, y, width, height, door_num, is_chosen=False, is_revealed=False, is_winning=False):
        """Draw a single door"""
        # Door color
        if is_revealed:
            color = self.GRAY
        elif is_chosen:
            color = self.GREEN
        elif is_winning and self.game_over:
            color = self.YELLOW
        else:
            color = self.LIGHT_GRAY
            
        # Draw door
        pygame.draw.rect(self.screen, color, (x, y, width, height))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, width, height), 3)
        
        # Draw door number
        text = self.font_medium.render(str(door_num), True, self.BLACK)
        text_rect = text.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text, text_rect)
        
        # Draw status
        if is_revealed:
            status_text = self.font_small.render("REVEALED", True, self.RED)
            self.screen.blit(status_text, (x, y - 20))
        elif is_chosen:
            status_text = self.font_small.render("CHOSEN", True, self.GREEN)
            self.screen.blit(status_text, (x, y - 20))
    
    def draw_game_state(self):
        """Draw the current game state"""
        step, doors_remaining, last_chosen = self.env.state
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Title
        title = self.font_large.render(f"Monty Hall Level 2 ({self.env.n_doors} doors)", True, self.BLACK)
        self.screen.blit(title, (10, 10))
        
        # Mode indicator
        mode_text = self.font_medium.render(f"Mode: {self.mode.upper()}", True, self.BLUE)
        self.screen.blit(mode_text, (10, 50))
        
        # Step info
        step_text = self.font_medium.render(f"Step: {step + 1}/{self.env.n_doors}", True, self.BLACK)
        self.screen.blit(step_text, (10, 80))
        
        # Draw doors
        door_width = 80
        door_height = 120
        start_x = 50
        start_y = 150
        
        for i, door in enumerate(doors_remaining):
            x = start_x + i * (door_width + 20)
            is_chosen = (door == last_chosen)
            is_revealed = (door in self.env._revealed)
            is_winning = (door == self.env._winning_door)
            
            self.draw_door(x, start_y, door_width, door_height, door, is_chosen, is_revealed, is_winning)
        
        # Revealed doors info
        if self.env._revealed:
            revealed_text = self.font_medium.render(f"Revealed doors: {self.env._revealed}", True, self.RED)
            self.screen.blit(revealed_text, (10, 300))
        
        # Winning door (only show if game over)
        if self.game_over:
            winning_text = self.font_medium.render(f"Winning door: {self.env._winning_door}", True, self.YELLOW)
            self.screen.blit(winning_text, (10, 330))
        
        # Policy recommendation
        if self.policy is not None and not self.env.is_terminal(self.env.state):
            try:
                state_idx = self.env.state_to_index(self.env.state)
                if state_idx < len(self.policy):
                    recommended_action = self.policy[state_idx]
                    if recommended_action != -1:
                        policy_text = self.font_medium.render(f"Policy recommends: Door {recommended_action}", True, self.BLUE)
                        self.screen.blit(policy_text, (10, 360))
                        
                        # Value function
                        if self.value_function is not None and state_idx < len(self.value_function):
                            value = self.value_function[state_idx]
                            value_text = self.font_medium.render(f"State value: {value:.3f}", True, self.BLUE)
                            self.screen.blit(value_text, (10, 390))
            except (KeyError, IndexError):
                pass
        
        # Instructions
        instructions = [
            "Instructions:",
            "- Click on a door to choose it (Manual mode)",
            "- Press SPACE to follow policy recommendation",
            "- Press 'M' to toggle Manual/Auto mode",
            "- Press 'R' to reset game",
            "- Press 'T' to train new policy",
            "- Press 'S' to save policy",
            "- Press 'L' to load policy"
        ]
        
        for i, instruction in enumerate(instructions):
            color = self.BLACK if i == 0 else self.DARK_GRAY
            font = self.font_medium if i == 0 else self.font_small
            text = font.render(instruction, True, color)
            self.screen.blit(text, (10, 450 + i * 20))
        
        # Message
        if self.message:
            message_text = self.font_medium.render(self.message, True, self.RED)
            self.screen.blit(message_text, (10, 650))
    
    def get_clicked_door(self, pos):
        """Get which door was clicked"""
        x, y = pos
        door_width = 80
        door_height = 120
        start_x = 50
        start_y = 150
        
        step, doors_remaining, last_chosen = self.env.state
        
        for i, door in enumerate(doors_remaining):
            door_x = start_x + i * (door_width + 20)
            if door_x <= x <= door_x + door_width and start_y <= y <= start_y + door_height:
                return door
        return None
    
    def reset_game(self):
        """Reset the game"""
        self.env.reset()
        self.game_over = False
        self.message = ""
    
    def make_move(self, action):
        """Make a move in the game"""
        if self.env.is_terminal(self.env.state):
            return
            
        try:
            state, reward, done = self.env.step(action)
            
            if done:
                self.game_over = True
                if reward > 0:
                    self.message = "You WIN! 🎉"
                else:
                    self.message = "You lose... 😞"
        except ValueError as e:
            self.message = str(e)
    
    def follow_policy(self):
        """Follow the policy recommendation"""
        if self.policy is None:
            self.message = "No policy loaded!"
            return
            
        if self.env.is_terminal(self.env.state):
            return
            
        try:
            state_idx = self.env.state_to_index(self.env.state)
            if state_idx < len(self.policy):
                action = self.policy[state_idx]
                if action != -1:
                    self.make_move(action)
                else:
                    self.message = "No action available for this state"
            else:
                self.message = "State not found in policy"
        except (KeyError, IndexError):
            self.message = "Error following policy"
    
    def train_policy(self):
        """Train a new policy"""
        self.message = "Training policy... Please wait"
        pygame.display.flip()
        
        try:
            policy, value_function = policy_iteration(self.env)
            self.policy = policy
            self.value_function = value_function
            self.message = "Policy trained successfully!"
        except Exception as e:
            self.message = f"Training error: {str(e)}"
    
    def save_policy(self, filename="monty_hall_policy.pkl"):
        """Save the current policy"""
        if self.policy is None:
            self.message = "No policy to save!"
            return
            
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'policy': self.policy,
                    'value_function': self.value_function,
                    'n_doors': self.env.n_doors
                }, f)
            self.message = f"Policy saved to {filename}"
        except Exception as e:
            self.message = f"Save error: {str(e)}"
    
    def load_policy(self, filename="monty_hall_policy.pkl"):
        """Load a policy"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                if data['n_doors'] != self.env.n_doors:
                    self.message = f"Policy is for {data['n_doors']} doors, but environment has {self.env.n_doors}"
                    return
                self.policy = data['policy']
                self.value_function = data['value_function']
                self.message = f"Policy loaded from {filename}"
        except FileNotFoundError:
            self.message = f"File {filename} not found"
        except Exception as e:
            self.message = f"Load error: {str(e)}"
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Try to load existing policy
        self.load_policy()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_m:
                        self.mode = "auto" if self.mode == "manual" else "manual"
                        self.message = f"Switched to {self.mode} mode"
                    elif event.key == pygame.K_SPACE:
                        self.follow_policy()
                    elif event.key == pygame.K_t:
                        self.train_policy()
                    elif event.key == pygame.K_s:
                        self.save_policy()
                    elif event.key == pygame.K_l:
                        self.load_policy()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.mode == "manual" and event.button == 1:  # Left click
                        clicked_door = self.get_clicked_door(event.pos)
                        if clicked_door is not None:
                            self.make_move(clicked_door)
            
            # Auto mode
            if self.mode == "auto" and not self.game_over and not self.env.is_terminal(self.env.state):
                time.sleep(1)  # Delay for visualization
                self.follow_policy()
            
            # Draw everything
            self.draw_game_state()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

# Performance evaluation
def evaluate_policy(env, policy, n_episodes=1000):
    """Evaluate the performance of a policy"""
    wins = 0
    
    for _ in range(n_episodes):
        env.reset()
        
        while not env.is_terminal(env.state):
            try:
                state_idx = env.state_to_index(env.state)
                if state_idx >= len(policy):
                    break
                action = policy[state_idx]
                if action == -1:
                    break
                env.step(action)
            except (ValueError, KeyError, IndexError):
                break
        
        # Check if won
        step, doors_remaining, chosen = env.state
        if chosen == env._winning_door:
            wins += 1
    
    return wins / n_episodes

# Main function
def main():
    print("Monty Hall Level 2 - Policy Iteration")
    print("=" * 40)
    
    # Create environment
    n_doors = 5  # You can change this
    env = MontyHallV2(n_doors=n_doors)
    print(f"Environment created with {n_doors} doors")
    
    # Train policy
    print("\nTraining policy...")
    policy, value_function = policy_iteration(env, gamma=0.99, theta=1e-6)
    
    # Evaluate policy
    print("\nEvaluating policy...")
    win_rate = evaluate_policy(env, policy, n_episodes=1000)
    print(f"Policy win rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
    
    # Save policy
    with open("monty_hall_policy.pkl", 'wb') as f:
        pickle.dump({
            'policy': policy,
            'value_function': value_function,
            'n_doors': n_doors
        }, f)
    print("Policy saved to monty_hall_policy.pkl")
    
    # Start visualization
    print("\nStarting visualization...")
    visualizer = MontyHallVisualizer(env, policy, value_function)
    visualizer.run()

if __name__ == "__main__":
    main()