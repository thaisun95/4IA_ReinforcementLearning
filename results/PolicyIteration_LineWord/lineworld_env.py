import numpy as np

class LineWorld:
    def __init__(self, size=5, start_state=2):
        self.size = size
        self.start_state = start_state
        self.state = start_state
        self.terminal_states = [0, size - 1]
        self.action_space = [0, 1]  # 0: left, 1: right

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_reward(self, next_state):
        if next_state == 0:
            return -1.0
        elif next_state == self.size - 1:
            return 1.0
        else:
            return 0.0

    def step(self, action):
        if self.is_terminal(self.state):
            return self.state, 0.0, True

        if action == 0:
            next_state = max(self.state - 1, 0)
        elif action == 1:
            next_state = min(self.state + 1, self.size - 1)
        else:
            raise ValueError("Invalid action")

        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
        if self.is_terminal(state):
            return state, 0.0, True
        if action == 0:
            next_state = max(state - 1, 0)
        elif action == 1:
            next_state = min(state + 1, self.size - 1)
        else:
            raise ValueError("Invalid action")
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def state_to_index(self, state):
        return state

    def index_to_state(self, index):
        return index

    @property
    def n_states(self):
        return self.size
