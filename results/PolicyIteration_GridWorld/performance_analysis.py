import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Policy Iteration Algorithm (copied from main script)
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

# GridWorld Environment (copied from main script)
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

class PerformanceAnalyzer:
    def __init__(self):
        self.results = {}
    
    def test_different_grid_sizes(self):
        """Test Policy Iteration on different grid sizes"""
        print("Testing different grid sizes...")
        sizes = [(3, 3), (4, 4), (5, 5), (6, 6), (8, 8)]
        
        for n_rows, n_cols in sizes:
            print(f"\nTesting {n_rows}x{n_cols} grid...")
            env = GridWorld(
                n_rows=n_rows, 
                n_cols=n_cols, 
                start_state=(0, 0), 
                terminal_states=[(n_rows-1, n_cols-1)]
            )
            
            start_time = time.time()
            policy, V = policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100)
            end_time = time.time()
            
            # Calculate metrics
            computation_time = end_time - start_time
            max_value = np.max(V)
            min_value = np.min(V)
            avg_value = np.mean(V)
            
            self.results[f"{n_rows}x{n_cols}"] = {
                'computation_time': computation_time,
                'max_value': max_value,
                'min_value': min_value,
                'avg_value': avg_value,
                'policy': policy.tolist(),
                'value_function': V.tolist()
            }
            
            print(f"Computation time: {computation_time:.4f}s")
            print(f"Value range: [{min_value:.3f}, {max_value:.3f}]")
    
    def test_different_rewards(self):
        """Test with different reward structures"""
        print("\nTesting different reward structures...")
        
        # Standard setup
        env = GridWorld(4, 4, (0, 0), [(3, 3)])
        
        reward_configs = {
            'standard': -np.ones((4, 4)),
            'sparse': np.zeros((4, 4)) - 0.1,
            'dense_negative': -np.ones((4, 4)) * 2,
            'mixed': np.array([[-1, -1, -1, -1],
                              [-1, -5, -1, -1], 
                              [-1, -1, -1, -1],
                              [-1, -1, -1, +10]])
        }
        
        # Set terminal reward to 0 for all configs
        for name, reward_matrix in reward_configs.items():
            reward_matrix[3, 3] = 0
        
        for config_name, reward_matrix in reward_configs.items():
            print(f"\nTesting {config_name} rewards...")
            env.reward_matrix = reward_matrix
            
            start_time = time.time()
            policy, V = policy_iteration(env, gamma=0.99)
            end_time = time.time()
            
            self.results[f"reward_{config_name}"] = {
                'computation_time': end_time - start_time,
                'max_value': np.max(V),
                'min_value': np.min(V),
                'avg_value': np.mean(V),
                'policy': policy.tolist(),
                'value_function': V.tolist(),
                'reward_matrix': reward_matrix.tolist()
            }
    
    def test_different_gamma(self):
        """Test with different discount factors"""
        print("\nTesting different discount factors...")
        
        env = GridWorld(4, 4, (0, 0), [(3, 3)])
        gamma_values = [0.5, 0.7, 0.9, 0.95, 0.99]
        
        for gamma in gamma_values:
            print(f"\nTesting gamma = {gamma}...")
            
            start_time = time.time()
            policy, V = policy_iteration(env, gamma=gamma)
            end_time = time.time()
            
            self.results[f"gamma_{gamma}"] = {
                'gamma': gamma,
                'computation_time': end_time - start_time,
                'max_value': np.max(V),
                'min_value': np.min(V),
                'avg_value': np.mean(V),
                'policy': policy.tolist(),
                'value_function': V.tolist()
            }
    
    def visualize_results(self):
        """Create visualizations of the results"""
        # Grid size performance
        grid_sizes = []
        computation_times = []
        
        for key, data in self.results.items():
            if 'x' in key and key.count('x') == 1:
                grid_sizes.append(key)
                computation_times.append(data['computation_time'])
        
        if grid_sizes:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.bar(grid_sizes, computation_times)
            plt.title('Computation Time vs Grid Size')
            plt.xlabel('Grid Size')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45)
        
        # Gamma analysis
        gamma_values = []
        gamma_times = []
        gamma_max_values = []
        
        for key, data in self.results.items():
            if key.startswith('gamma_'):
                gamma_values.append(data['gamma'])
                gamma_times.append(data['computation_time'])
                gamma_max_values.append(data['max_value'])
        
        if gamma_values:
            plt.subplot(1, 3, 2)
            plt.plot(gamma_values, gamma_times, 'o-')
            plt.title('Computation Time vs Discount Factor')
            plt.xlabel('Gamma')
            plt.ylabel('Time (seconds)')
            
            plt.subplot(1, 3, 3)
            plt.plot(gamma_values, gamma_max_values, 'o-')
            plt.title('Max Value vs Discount Factor')
            plt.xlabel('Gamma')
            plt.ylabel('Max Value')
        
        plt.tight_layout()
        plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis(self):
        """Save analysis results"""
        import os
        os.makedirs('results', exist_ok=True)
        
        with open('results/performance_analysis.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary report
        with open('results/performance_report.txt', 'w') as f:
            f.write("POLICY ITERATION PERFORMANCE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Grid size analysis
            f.write("1. GRID SIZE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for key, data in self.results.items():
                if 'x' in key and key.count('x') == 1:
                    f.write(f"Grid {key}:\n")
                    f.write(f"  Computation time: {data['computation_time']:.4f}s\n")
                    f.write(f"  Value range: [{data['min_value']:.3f}, {data['max_value']:.3f}]\n")
                    f.write(f"  Average value: {data['avg_value']:.3f}\n\n")
            
            # Gamma analysis
            f.write("\n2. DISCOUNT FACTOR ANALYSIS\n")
            f.write("-" * 25 + "\n")
            for key, data in self.results.items():
                if key.startswith('gamma_'):
                    f.write(f"Gamma {data['gamma']}:\n")
                    f.write(f"  Computation time: {data['computation_time']:.4f}s\n")
                    f.write(f"  Max value: {data['max_value']:.3f}\n")
                    f.write(f"  Average value: {data['avg_value']:.3f}\n\n")
            
            # Reward structure analysis
            f.write("\n3. REWARD STRUCTURE ANALYSIS\n")
            f.write("-" * 27 + "\n")
            for key, data in self.results.items():
                if key.startswith('reward_'):
                    reward_type = key.replace('reward_', '')
                    f.write(f"Reward type: {reward_type}\n")
                    f.write(f"  Computation time: {data['computation_time']:.4f}s\n")
                    f.write(f"  Value range: [{data['min_value']:.3f}, {data['max_value']:.3f}]\n")
                    f.write(f"  Average value: {data['avg_value']:.3f}\n\n")
        
        print("Analysis saved to 'results/' directory")

def run_complete_analysis():
    """Run complete performance analysis"""
    analyzer = PerformanceAnalyzer()
    
    # Run all tests
    analyzer.test_different_grid_sizes()
    analyzer.test_different_rewards()
    analyzer.test_different_gamma()
    
    # Create visualizations
    analyzer.visualize_results()
    
    # Save results
    analyzer.save_analysis()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("Check 'results/' directory for:")
    print("- performance_analysis.json (raw data)")
    print("- performance_report.txt (summary)")
    print("- performance_analysis.png (plots)")
    print("="*50)

if __name__ == "__main__":
    run_complete_analysis()