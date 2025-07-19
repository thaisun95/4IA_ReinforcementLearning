import numpy as np
import random
import pygame
import sys
import pickle
import json

class TwoRoundRPS:
    def __init__(self):
        """
        Two-round Rock-Paper-Scissors environment.
        State: (round, my_first_move)
            - round: 0 (first round), 1 (second round), 2 (terminal)
            - my_first_move: -1 (not played yet), 0=Rock, 1=Paper, 2=Scissors
        """
        self.action_space = [0, 1, 2]  # 0=Rock, 1=Paper, 2=Scissors
        # All possible states: (round, my_first_move)
        self.states = [(0, -1)]
        self.states += [(1, a) for a in self.action_space]
        self.states += [(2, a) for a in self.action_space]
        self.n_states = len(self.states)
        self.state = (0, -1)  # initial state

        # For manual play
        self.last_opp_move = None

    def state_to_index(self, state):
        return self.states.index(state)

    def index_to_state(self, index):
        return self.states[index]

    def reset(self):
        self.state = (0, -1)
        self.last_opp_move = None
        return self.state

    def is_terminal(self, state):
        return state[0] == 2

    def get_reward(self, my_move, opp_move):
        """
        Standard RPS reward: 1 if win, 0 if draw, -1 if lose.
        """
        if my_move == opp_move:
            return 0
        elif (my_move == 0 and opp_move == 2) or (my_move == 1 and opp_move == 0) or (my_move == 2 and opp_move == 1):
            return 1
        else:
            return -1

    def simulate_step(self, state, action):
        """
        Simulate the next step given a state and action.
        Does not modify self.state!
        Returns: next_state, reward, done
        """
        round_id, my_first_move = state
        if round_id == 0:
            opp_move = random.choice(self.action_space)  # opponent plays random in first round
            reward = self.get_reward(action, opp_move)
            next_state = (1, action)
            done = False
        elif round_id == 1:
            opp_move = my_first_move  # opponent copies agent's first move
            reward = self.get_reward(action, opp_move)
            next_state = (2, my_first_move)
            done = True
        else:
            next_state = state
            reward = 0
            done = True
        return next_state, reward, done

    def step(self, action):
        """
        Take an action in the environment, update self.state, return (next_state, reward, done).
        """
        round_id, my_first_move = self.state
        if round_id == 0:
            opp_move = random.choice(self.action_space)
            self.last_opp_move = opp_move
            reward = self.get_reward(action, opp_move)
            next_state = (1, action)
            done = False
        elif round_id == 1:
            opp_move = my_first_move
            self.last_opp_move = opp_move
            reward = self.get_reward(action, opp_move)
            next_state = (2, my_first_move)
            done = True
        else:
            next_state = self.state
            reward = 0
            done = True
        self.state = next_state
        return next_state, reward, done


def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    """
    Perform Policy Iteration for a generic environment with state-index mapping.
    """
    policy = np.random.choice(env.action_space, size=env.n_states)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if env.is_terminal(state):
            policy[idx] = -1  # No action for terminal states

    V = np.zeros(env.n_states)

    for it in range(max_iterations):
        # --- Policy Evaluation ---
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

        # --- Policy Improvement ---
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


class RPSVisualizer:
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Two Round RPS - Policy Iteration")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 200)
        self.GREEN = (0, 150, 0)
        self.RED = (200, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 18)
        
        # Game state
        self.env = TwoRoundRPS()
        self.policy = None
        self.V = None
        self.mode = "training"  # "training", "ai_play", "manual_play"
        self.training_done = False
        self.current_episode = []
        self.episode_step = 0
        self.game_history = []
        
        self.move_names = ["Rock", "Paper", "Scissors"]
        self.move_emojis = ["🪨", "📄", "✂️"]
        
    def train_policy(self):
        """Train the policy using Policy Iteration"""
        print("Training Policy Iteration...")
        self.policy, self.V = policy_iteration(self.env, gamma=0.99, theta=1e-6, max_iterations=100)
        self.training_done = True
        
        # Save results
        self.save_results()
        print("Training completed!")
        
    def save_results(self):
        """Save policy, value function and results"""
        results = {
            'policy': self.policy.tolist(),
            'V': self.V.tolist(),
            'states': self.env.states
        }
        
        with open('rps_policy_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save as pickle for easy loading
        with open('rps_policy_results.pkl', 'wb') as f:
            pickle.dump({'policy': self.policy, 'V': self.V, 'env': self.env}, f)
            
        print("Results saved to rps_policy_results.json and .pkl")
        
    def load_results(self):
        """Load saved results"""
        try:
            with open('rps_policy_results.pkl', 'rb') as f:
                data = pickle.load(f)
                self.policy = data['policy']
                self.V = data['V']
                self.training_done = True
                print("Results loaded successfully!")
                return True
        except FileNotFoundError:
            print("No saved results found.")
            return False
            
    def get_ai_action(self, state):
        """Get action from trained policy"""
        if self.policy is None:
            return random.choice(self.env.action_space)
        
        state_idx = self.env.state_to_index(state)
        return int(self.policy[state_idx])
        
    def start_ai_episode(self):
        """Start a new AI episode for step-by-step visualization"""
        self.env.reset()
        self.current_episode = []
        self.episode_step = 0
        
        # Generate complete episode
        state = self.env.state
        done = False
        
        while not done:
            action = self.get_ai_action(state)
            next_state, reward, done = self.env.step(action)
            
            self.current_episode.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'done': done,
                'opp_move': self.env.last_opp_move
            })
            
            state = next_state
            
    def draw_policy_table(self, x, y):
        """Draw the policy table"""
        if not self.training_done or self.policy is None:
            return
            
        title_text = self.big_font.render("Optimal Policy", True, self.BLACK)
        self.screen.blit(title_text, (x, y))
        y += 40
        
        # Headers
        headers = ["State", "Round", "First Move", "Action", "Value"]
        col_widths = [50, 60, 100, 80, 80]
        x_pos = x
        
        for i, header in enumerate(headers):
            text = self.font.render(header, True, self.BLACK)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, (x_pos, y, col_widths[i], 25))
            pygame.draw.rect(self.screen, self.BLACK, (x_pos, y, col_widths[i], 25), 1)
            self.screen.blit(text, (x_pos + 5, y + 3))
            x_pos += col_widths[i]
            
        y += 25
        
        # Data rows
        for idx in range(self.env.n_states):
            state = self.env.index_to_state(idx)
            round_id, first_move = state
            
            x_pos = x
            
            # State index
            text = self.font.render(str(idx), True, self.BLACK)
            pygame.draw.rect(self.screen, self.WHITE, (x_pos, y, col_widths[0], 25))
            pygame.draw.rect(self.screen, self.BLACK, (x_pos, y, col_widths[0], 25), 1)
            self.screen.blit(text, (x_pos + 5, y + 3))
            x_pos += col_widths[0]
            
            # Round
            text = self.font.render(str(round_id + 1) if round_id < 2 else "Term", True, self.BLACK)
            pygame.draw.rect(self.screen, self.WHITE, (x_pos, y, col_widths[1], 25))
            pygame.draw.rect(self.screen, self.BLACK, (x_pos, y, col_widths[1], 25), 1)
            self.screen.blit(text, (x_pos + 5, y + 3))
            x_pos += col_widths[1]
            
            # First move
            first_move_text = "--" if first_move == -1 else self.move_names[first_move]
            text = self.font.render(first_move_text, True, self.BLACK)
            pygame.draw.rect(self.screen, self.WHITE, (x_pos, y, col_widths[2], 25))
            pygame.draw.rect(self.screen, self.BLACK, (x_pos, y, col_widths[2], 25), 1)
            self.screen.blit(text, (x_pos + 5, y + 3))
            x_pos += col_widths[2]
            
            # Action
            if self.env.is_terminal(state):
                action_text = "--"
                color = self.GRAY
            else:
                action_text = self.move_names[int(self.policy[idx])]
                color = self.BLACK
            text = self.font.render(action_text, True, color)
            pygame.draw.rect(self.screen, self.WHITE, (x_pos, y, col_widths[3], 25))
            pygame.draw.rect(self.screen, self.BLACK, (x_pos, y, col_widths[3], 25), 1)
            self.screen.blit(text, (x_pos + 5, y + 3))
            x_pos += col_widths[3]
            
            # Value
            value_text = f"{self.V[idx]:.3f}"
            text = self.font.render(value_text, True, self.BLACK)
            pygame.draw.rect(self.screen, self.WHITE, (x_pos, y, col_widths[4], 25))
            pygame.draw.rect(self.screen, self.BLACK, (x_pos, y, col_widths[4], 25), 1)
            self.screen.blit(text, (x_pos + 5, y + 3))
            
            y += 25
            
    def draw_game_state(self, x, y):
        """Draw current game state"""
        title_text = self.big_font.render("Current Game State", True, self.BLACK)
        self.screen.blit(title_text, (x, y))
        y += 40
        
        state = self.env.state
        round_id, first_move = state
        
        # Current round
        round_text = f"Round: {round_id + 1 if round_id < 2 else 'Terminal'}"
        text = self.font.render(round_text, True, self.BLACK)
        self.screen.blit(text, (x, y))
        y += 30
        
        # First move info
        if first_move != -1:
            first_move_text = f"Agent's first move: {self.move_names[first_move]}"
            text = self.font.render(first_move_text, True, self.BLACK)
            self.screen.blit(text, (x, y))
            y += 25
            
        # Last opponent move
        if self.env.last_opp_move is not None:
            opp_text = f"Opponent's last move: {self.move_names[self.env.last_opp_move]}"
            text = self.font.render(opp_text, True, self.BLACK)
            self.screen.blit(text, (x, y))
            y += 25
            
        return y
        
    def draw_episode_playback(self, x, y):
        """Draw step-by-step episode playback"""
        if not self.current_episode:
            return y
            
        title_text = self.big_font.render("Episode Playback", True, self.BLACK)
        self.screen.blit(title_text, (x, y))
        y += 40
        
        step_text = f"Step: {self.episode_step + 1}/{len(self.current_episode)}"
        text = self.font.render(step_text, True, self.BLACK)
        self.screen.blit(text, (x, y))
        y += 30
        
        if self.episode_step < len(self.current_episode):
            step_data = self.current_episode[self.episode_step]
            
            # State info
            state = step_data['state']
            round_id, first_move = state
            
            state_text = f"State: Round {round_id + 1}, First move: {self.move_names[first_move] if first_move != -1 else 'None'}"
            text = self.font.render(state_text, True, self.BLACK)
            self.screen.blit(text, (x, y))
            y += 25
            
            # Action
            action_text = f"Agent plays: {self.move_names[step_data['action']]}"
            text = self.font.render(action_text, True, self.GREEN)
            self.screen.blit(text, (x, y))
            y += 25
            
            # Opponent move
            opp_text = f"Opponent plays: {self.move_names[step_data['opp_move']]}"
            text = self.font.render(opp_text, True, self.RED)
            self.screen.blit(text, (x, y))
            y += 25
            
            # Reward
            reward_text = f"Reward: {step_data['reward']}"
            text = self.font.render(reward_text, True, self.BLUE)
            self.screen.blit(text, (x, y))
            y += 25
            
        return y
        
    def draw_controls(self, x, y):
        """Draw control instructions"""
        title_text = self.big_font.render("Controls", True, self.BLACK)
        self.screen.blit(title_text, (x, y))
        y += 40
        
        controls = [
            "T - Train Policy (if not done)",
            "L - Load saved results",
            "A - Start AI episode playback",
            "M - Manual play mode",
            "SPACE - Next step (in AI playback)",
            "R - Reset environment",
            "1,2,3 - Rock, Paper, Scissors (manual mode)",
            "ESC - Quit"
        ]
        
        for control in controls:
            text = self.font.render(control, True, self.BLACK)
            self.screen.blit(text, (x, y))
            y += 22
            
        return y
        
    def draw_mode_info(self, x, y):
        """Draw current mode information"""
        mode_colors = {
            "training": self.BLUE,
            "ai_play": self.GREEN,
            "manual_play": self.RED
        }
        
        mode_text = f"Mode: {self.mode.upper()}"
        color = mode_colors.get(self.mode, self.BLACK)
        text = self.big_font.render(mode_text, True, color)
        self.screen.blit(text, (x, y))
        
        if not self.training_done:
            status_text = "Policy not trained yet - Press T to train"
            text = self.font.render(status_text, True, self.RED)
            self.screen.blit(text, (x, y + 35))
        else:
            status_text = "Policy trained and ready"
            text = self.font.render(status_text, True, self.GREEN)
            self.screen.blit(text, (x, y + 35))
            
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                    
                elif event.key == pygame.K_t and not self.training_done:
                    self.train_policy()
                    
                elif event.key == pygame.K_l:
                    self.load_results()
                    
                elif event.key == pygame.K_a and self.training_done:
                    self.mode = "ai_play"
                    self.start_ai_episode()
                    
                elif event.key == pygame.K_m:
                    self.mode = "manual_play"
                    self.env.reset()
                    
                elif event.key == pygame.K_r:
                    self.env.reset()
                    self.current_episode = []
                    self.episode_step = 0
                    
                elif event.key == pygame.K_SPACE and self.mode == "ai_play":
                    if self.episode_step < len(self.current_episode) - 1:
                        self.episode_step += 1
                    else:
                        self.start_ai_episode()  # Start new episode
                        
                elif self.mode == "manual_play" and not self.env.is_terminal(self.env.state):
                    action = None
                    if event.key == pygame.K_1:
                        action = 0  # Rock
                    elif event.key == pygame.K_2:
                        action = 1  # Paper
                    elif event.key == pygame.K_3:
                        action = 2  # Scissors
                        
                    if action is not None:
                        next_state, reward, done = self.env.step(action)
                        print(f"Manual play: {self.move_names[action]} -> Reward: {reward}")
                        
        return True
        
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Try to load existing results
        self.load_results()
        
        while running:
            running = self.handle_events()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw UI elements
            self.draw_mode_info(20, 20)
            
            y_pos = self.draw_game_state(20, 80)
            
            if self.mode == "ai_play":
                y_pos = self.draw_episode_playback(20, y_pos + 20)
                
            self.draw_controls(20, y_pos + 20)
            
            # Draw policy table on the right side
            if self.training_done:
                self.draw_policy_table(450, 20)
            
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()
        sys.exit()


def main():
    """Main function to run the visualization"""
    visualizer = RPSVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()