import numpy as np
import random

class MontyHallV2:
    def __init__(self, n_doors=5):
        """
        Monty Hall Level 2 environment (generalized for n doors, default 5).
        The agent makes n_doors-1 choices; after each, Monty removes a door
        (not chosen, not winning), until 2 doors remain.
        """
        self.n_doors = n_doors
        self.action_space = list(range(n_doors))  # action: pick door by index
        # Each state: (step, tuple of doors_remaining, chosen)
        self.states = []
        # Initial state: all doors available, nothing chosen
        self.states.append((0, tuple(range(n_doors)), -1))
        # Generate all possible reachable states step by step
        for step in range(1, n_doors):
            # At each step, doors_remaining has (n_doors - step + 1) doors, agent just chose "chosen"
            # Enumerate all possible remaining doors and last chosen
            for doors_remaining in self._all_possible_remaining_doors(n_doors, n_doors - step + 1):
                for chosen in doors_remaining:
                    self.states.append((step, doors_remaining, chosen))
        # Add terminal states: step=n_doors, only 1 door remains, chosen=which door
        for doors_remaining in self._all_possible_remaining_doors(n_doors, 1):
            for chosen in doors_remaining:
                self.states.append((n_doors, doors_remaining, chosen))
        self.n_states = len(self.states)
        self.state = (0, tuple(range(n_doors)), -1)
        self._winning_door = None  # internal (hidden) winning door

    def _all_possible_remaining_doors(self, n_doors, k):
        # Return all sorted tuples of length k from n_doors doors
        from itertools import combinations
        return [tuple(sorted(comb)) for comb in combinations(range(n_doors), k)]

    def state_to_index(self, state):
        return self.states.index(state)

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self):
        self._winning_door = random.choice(range(self.n_doors))
        self.state = (0, tuple(range(self.n_doors)), -1)
        return self.state

    def is_terminal(self, state):
        step, doors_remaining, chosen = state
        return step == self.n_doors

    def simulate_step(self, state, action):
        """
        Simulates (does not modify self.state!).
        state: (step, doors_remaining, last_chosen)
        action: chosen door at this step (must be in doors_remaining)
        Returns: (next_state, reward, done)
        """
        step, doors_remaining, last_chosen = state
        if self.is_terminal(state):
            # Already at terminal state
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return state, reward, True

        # 1. Agent chooses a door among doors_remaining
        chosen = action
        # 2. If not the last step, Monty removes a door (not chosen, not winning)
        if step < self.n_doors - 1:
            # Find all removable doors: not agent's choice, not winning door
            removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
            # Monty must remove exactly one door, always possible if n_doors >= 3
            if len(removable) == 0:
                # Should not happen in standard Monty Hall logic
                raise ValueError("No door for Monty to remove!")
            door_to_remove = random.choice(removable)
            # Next set of doors
            next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
            next_state = (step + 1, next_doors, chosen)
            reward = 0.0
            done = False
        else:
            # After the last pick, terminal state: agent chooses among 2 remaining doors
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
        return next_state, reward, done