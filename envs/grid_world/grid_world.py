import numpy as np

class GridWorld:
    def __init__(self, n_rows=4, n_cols=4, start_state=(0,0), terminal_states=None, reward_matrix=None):
        """
        Initialize the GridWorld environment.

        Args:
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            start_state (tuple): Starting position (row, col).
            terminal_states (list of tuples): Terminal states positions.
            reward_matrix (np.ndarray): Matrix of rewards (optional).
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.state = start_state
        self.start_state = start_state
        # Terminal states: list of (row, col). Default: bottom-right corner.
        if terminal_states is None:
            self.terminal_states = [(n_rows-1, n_cols-1)]
        else:
            self.terminal_states = terminal_states
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = [0, 1, 2, 3]
        # Set rewards: default -1 everywhere, +1 in terminal state
        if reward_matrix is None:
            self.reward_matrix = -np.ones((n_rows, n_cols))
            for t in self.terminal_states:
                self.reward_matrix[t] = 0
        else:
            self.reward_matrix = reward_matrix

    def state_to_index(self, state):
        """
        Convert a (row, col) tuple to a single state index.
        Args:
            state (tuple): (row, col)
        Returns:
            int: state index
        """
        row, col = state
        return row * self.n_cols + col

    def index_to_state(self, index):
        """
        Convert a single state index to a (row, col) tuple.
        Args:
            index (int): state index
        Returns:
            tuple: (row, col)
        """
        row = index // self.n_cols
        col = index % self.n_cols
        return (row, col)

    @property
    def n_states(self):
        """Total number of states."""
        return self.n_rows * self.n_cols

    def reset(self):
        """
        Reset the environment to the initial state.
        Returns:
            tuple: Initial state (row, col).
        """
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        """
        Check if a state is terminal.
        Args:
            state (tuple): State as (row, col).
        Returns:
            bool: True if state is terminal.
        """
        return state in self.terminal_states

    def get_reward(self, state):
        """
        Get the reward for entering a given state.
        Args:
            state (tuple): State as (row, col).
        Returns:
            float: Reward.
        """
        return self.reward_matrix[state]

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): 0=up, 1=down, 2=left, 3=right

        Returns:
            next_state (tuple): Next state (row, col).
            reward (float): Reward for the transition.
            done (bool): Whether the next state is terminal.
        """
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
        else:
            raise ValueError("Invalid action (0=up, 1=down, 2=left, 3=right)")

        next_state = (next_row, next_col)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def render(self):
        """
        Print a visual representation of the grid.
        The agent's position is marked by 'A'.
        """
        grid = np.array([["."]*self.n_cols for _ in range(self.n_rows)])
        row, col = self.state
        grid[row, col] = "A"
        for t in self.terminal_states:
            grid[t] = "T"
        print("\n".join(" ".join(row) for row in grid))