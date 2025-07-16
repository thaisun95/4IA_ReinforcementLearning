class LineWorld:
    def __init__(self, size=5, start_state=2):
        """
        Initialize the LineWorld environment.

        Args:
            size (int): Number of states in the environment (default 5).
            start_state (int): The agent's initial position (default 2).
        """
        self.size = size
        self.start_state = start_state
        self.state = start_state
        self.terminal_states = [0, size - 1]
        self.action_space = [0, 1]  # 0: left, 1: right

    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            int: The initial state.
        """
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        """
        Check if a given state is terminal.
        
        Args:
            state (int): State to check.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return state in self.terminal_states

    def get_reward(self, next_state):
        """
        Get the reward for transitioning into the next state.

        Args:
            next_state (int): The state resulting from the agent's action.

        Returns:
            float: The reward received for the transition.
        """
        if next_state == 0:
            return -1.0
        elif next_state == self.size - 1:
            return 1.0
        else:
            return 0.0

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): Action to take (0 = left, 1 = right).

        Returns:
            tuple: (next_state (int), reward (float), done (bool))
        """
        if self.is_terminal(self.state):
            # No further action, the episode is already done
            return self.state, 0.0, True

        if action == 0:
            next_state = self.state - 1
        elif action == 1:
            next_state = self.state + 1
        else:
            raise ValueError("Invalid action (0=left, 1=right)")

        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def render(self):
        """
        Print a visual representation of the environment.
        The agent's position is marked by 'A'.
        """
        line = ['_'] * self.size
        line[self.state] = 'A'
        print(' '.join(line))