import numpy as np
import pygame
import sys
import json
import os

# Policy Iteration Algorithm
def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    """
    Perform Policy Iteration for a generic environment with state-index mapping.
    """
    policy = np.random.choice(env.action_space, size=env.n_states)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if env.is_terminal(state):
            policy[idx] = -1

    V = np.zeros(env.n_states)

    for it in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            for idx in range(env.n_states):
                state = env.index_to_state(idx)
                if env.is_terminal(state):
                    continue
                v = V[idx]
                a = policy[idx]
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                V[idx] = reward + gamma * V[next_idx]
                delta = max(delta, abs(v - V[idx]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
            old_action = policy[idx]
            action_values = []
            for a in env.action_space:
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                action_values.append(reward + gamma * V[next_idx])
            best_action = np.argmax(action_values)
            policy[idx] = env.action_space[best_action]
            if old_action != policy[idx]:
                policy_stable = False
        if policy_stable:
            print(f"Policy iteration converged after {it+1} iterations.")
            break

    return policy, V

# GridWorld Environment
class GridWorld:
    def __init__(self, n_rows=4, n_cols=4, start_state=(0,0), terminal_states=None, reward_matrix=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.state = start_state
        self.start_state = start_state
        if terminal_states is None:
            self.terminal_states = [(n_rows-1, n_cols-1)]
        else:
            self.terminal_states = terminal_states
        self.action_space = [0, 1, 2, 3]
        if reward_matrix is None:
            self.reward_matrix = -np.ones((n_rows, n_cols))
            for t in self.terminal_states:
                self.reward_matrix[t] = 0
        else:
            self.reward_matrix = reward_matrix

    def state_to_index(self, state):
        row, col = state
        return row * self.n_cols + col

    def index_to_state(self, index):
        row = index // self.n_cols
        col = index % self.n_cols
        return (row, col)

    @property
    def n_states(self):
        return self.n_rows * self.n_cols

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_reward(self, state):
        return self.reward_matrix[state]

    def step(self, action):
        if self.is_terminal(self.state):
            return self.state, self.get_reward(self.state), True

        row, col = self.state
        if action == 0:   # up
            next_row = max(row - 1, 0)
            next_col = col
        elif action == 1: # down
            next_row = min(row + 1, self.n_rows - 1)
            next_col = col
        elif action == 2: # left
            next_row = row
            next_col = max(col - 1, 0)
        elif action == 3: # right
            next_row = row
            next_col = min(col + 1, self.n_cols - 1)

        next_state = (next_row, next_col)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
        row, col = state
        if self.is_terminal(state):
            return state, self.get_reward(state), True
        if action == 0:
            next_row, next_col = max(row - 1, 0), col
        elif action == 1:
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 2:
            next_row, next_col = row, max(col - 1, 0)
        elif action == 3:
            next_row, next_col = row, min(col + 1, self.n_cols - 1)
        
        next_state = (next_row, next_col)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

# Pygame Visualization
class GridWorldVisualizer:
    def __init__(self, env, cell_size=100):
        self.env = env
        self.cell_size = cell_size
        self.width = env.n_cols * cell_size
        self.height = env.n_rows * cell_size + 100  # Extra space for controls
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World - Policy Iteration")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        
        # Game state
        self.policy = None
        self.value_function = None
        self.show_policy = True
        self.show_values = True
        self.manual_mode = False
        self.auto_play = False
        self.step_by_step = False
        self.episode_history = []

    def train_policy(self, gamma=0.99, theta=1e-6, max_iterations=100):
        """Train the policy using Policy Iteration"""
        print("Training policy...")
        self.policy, self.value_function = policy_iteration(self.env, gamma, theta, max_iterations)
        print("Training completed!")
        self.save_results()

    def save_results(self):
        """Save trained policy and value function"""
        results = {
            'policy': self.policy.tolist(),
            'value_function': self.value_function.tolist(),
            'env_config': {
                'n_rows': self.env.n_rows,
                'n_cols': self.env.n_cols,
                'start_state': self.env.start_state,
                'terminal_states': self.env.terminal_states,
                'reward_matrix': self.env.reward_matrix.tolist()
            }
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/policy_iteration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to 'results/policy_iteration_results.json'")

    def load_results(self):
        """Load saved policy and value function"""
        try:
            with open('results/policy_iteration_results.json', 'r') as f:
                results = json.load(f)
            self.policy = np.array(results['policy'])
            self.value_function = np.array(results['value_function'])
            print("Results loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved results found. Please train first.")
            return False

    def draw_grid(self):
        """Draw the grid and current state"""
        self.screen.fill(self.WHITE)
        
        # Draw grid lines
        for i in range(self.env.n_rows + 1):
            pygame.draw.line(self.screen, self.BLACK, 
                           (0, i * self.cell_size), 
                           (self.width, i * self.cell_size))
        
        for j in range(self.env.n_cols + 1):
            pygame.draw.line(self.screen, self.BLACK, 
                           (j * self.cell_size, 0), 
                           (j * self.cell_size, self.env.n_rows * self.cell_size))

        # Draw cells
        for row in range(self.env.n_rows):
            for col in range(self.env.n_cols):
                x = col * self.cell_size
                y = row * self.cell_size
                
                # Color terminal states
                if (row, col) in self.env.terminal_states:
                    pygame.draw.rect(self.screen, self.GREEN, 
                                   (x+2, y+2, self.cell_size-4, self.cell_size-4))

                # Show values if available
                if self.show_values and self.value_function is not None:
                    idx = self.env.state_to_index((row, col))
                    value_text = f"{self.value_function[idx]:.2f}"
                    text_surface = self.small_font.render(value_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(center=(x + self.cell_size//4, y + self.cell_size//4))
                    self.screen.blit(text_surface, text_rect)

                # Show policy arrows if available
                if self.show_policy and self.policy is not None:
                    idx = self.env.state_to_index((row, col))
                    if not self.env.is_terminal((row, col)):
                        action = self.policy[idx]
                        self.draw_arrow(x, y, action)

        # Draw agent
        agent_x = self.env.state[1] * self.cell_size + self.cell_size // 2
        agent_y = self.env.state[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, self.RED, (agent_x, agent_y), self.cell_size // 4)

    def draw_arrow(self, x, y, action):
        """Draw policy arrow for given action"""
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        arrow_size = self.cell_size // 4
        
        if action == 0:  # up
            points = [(center_x, center_y - arrow_size),
                     (center_x - arrow_size//2, center_y),
                     (center_x + arrow_size//2, center_y)]
        elif action == 1:  # down
            points = [(center_x, center_y + arrow_size),
                     (center_x - arrow_size//2, center_y),
                     (center_x + arrow_size//2, center_y)]
        elif action == 2:  # left
            points = [(center_x - arrow_size, center_y),
                     (center_x, center_y - arrow_size//2),
                     (center_x, center_y + arrow_size//2)]
        elif action == 3:  # right
            points = [(center_x + arrow_size, center_y),
                     (center_x, center_y - arrow_size//2),
                     (center_x, center_y + arrow_size//2)]
        
        if action != -1:
            pygame.draw.polygon(self.screen, self.BLUE, points)

    def draw_controls(self):
        """Draw control panel"""
        control_y = self.env.n_rows * self.cell_size + 10
        
        # Mode indicators
        mode_text = "MANUAL MODE" if self.manual_mode else "AUTO MODE"
        color = self.RED if self.manual_mode else self.GREEN
        text_surface = self.font.render(mode_text, True, color)
        self.screen.blit(text_surface, (10, control_y))
        
        # Instructions
        instructions = [
            "M: Toggle Manual/Auto | R: Reset | T: Train Policy | L: Load Results",
            "P: Toggle Policy | V: Toggle Values | SPACE: Step (Auto) | Arrow Keys: Move (Manual)"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(text_surface, (10, control_y + 30 + i * 20))

    def handle_manual_input(self, key):
        """Handle manual movement"""
        action_map = {
            pygame.K_UP: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_RIGHT: 3
        }
        
        if key in action_map:
            action = action_map[key]
            next_state, reward, done = self.env.step(action)
            self.episode_history.append((self.env.state, action, reward, done))
            print(f"Manual step: Action={action}, State={next_state}, Reward={reward}, Done={done}")

    def auto_step(self):
        """Take one step following the learned policy"""
        if self.policy is None:
            return
        
        current_state = self.env.state
        if self.env.is_terminal(current_state):
            return
        
        idx = self.env.state_to_index(current_state)
        action = self.policy[idx]
        next_state, reward, done = self.env.step(action)
        self.episode_history.append((current_state, action, reward, done))
        print(f"Auto step: Action={action}, State={next_state}, Reward={reward}, Done={done}")

    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.manual_mode = not self.manual_mode
                        print(f"Mode changed to: {'Manual' if self.manual_mode else 'Auto'}")
                    
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        self.episode_history = []
                        print("Environment reset")
                    
                    elif event.key == pygame.K_t:
                        self.train_policy()
                    
                    elif event.key == pygame.K_l:
                        self.load_results()
                    
                    elif event.key == pygame.K_p:
                        self.show_policy = not self.show_policy
                        print(f"Policy display: {'ON' if self.show_policy else 'OFF'}")
                    
                    elif event.key == pygame.K_v:
                        self.show_values = not self.show_values
                        print(f"Value display: {'ON' if self.show_values else 'OFF'}")
                    
                    elif event.key == pygame.K_SPACE:
                        if not self.manual_mode:
                            self.auto_step()
                    
                    elif self.manual_mode:
                        self.handle_manual_input(event.key)

            self.draw_grid()
            self.draw_controls()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

# Main execution
def main():
    # Create environment (you can modify these parameters)
    env = GridWorld(
        n_rows=4, 
        n_cols=4, 
        start_state=(0, 0), 
        terminal_states=[(3, 3)],
        reward_matrix=None  # Will use default: -1 everywhere except terminals (0)
    )
    
    # Create visualizer
    visualizer = GridWorldVisualizer(env)
    
    # Try to load existing results
    if not visualizer.load_results():
        print("No saved results found. Press 'T' to train a new policy.")
    
    print("\n=== CONTROLS ===")
    print("M: Toggle between Manual and Auto mode")
    print("R: Reset environment to start state")
    print("T: Train new policy using Policy Iteration")
    print("L: Load saved policy and value function")
    print("P: Toggle policy arrows display")
    print("V: Toggle value function display")
    print("SPACE: Take one step (Auto mode only)")
    print("Arrow Keys: Move agent (Manual mode only)")
    print("ESC/Close: Exit")
    print("\n=== Starting Visualization ===")
    
    # Run the visualization
    visualizer.run()

if __name__ == "__main__":
    main()