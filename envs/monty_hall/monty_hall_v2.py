import numpy as np
import random
from itertools import combinations
from collections import deque

class MontyHallV2:
    """
    Generalized Monty Hall environment (n doors, n=5 by default).
    The state is (step, sorted tuple of remaining doors, chosen).
    Only reachable states are created (solves ValueError bugs!).
    """
    def __init__(self, n_doors=5):
        self.n_doors = n_doors
        self.action_space = list(range(n_doors))
        self.states = []
        self._state_set = set()  # for fast membership check

        # Build only reachable states (BFS from initial)
        initial_state = (0, tuple(range(n_doors)), -1)
        queue = deque([initial_state])
        self._state_set.add(initial_state)

        while queue:
            state = queue.popleft()
            self.states.append(state)
            step, doors_remaining, last_chosen = state
            if step == n_doors or len(doors_remaining) == 1:
                continue  # terminal state

            # For all valid actions from this state
            for action in doors_remaining:
                # Next state logic
                chosen = action
                if len(doors_remaining) == 2:
                    next_state = (n_doors, tuple(sorted(doors_remaining)), chosen)
                else:
                    # Monty removes a door (not chosen, not winning), we enumerate all possible winning doors for construction
                    # But here we add all possible outcomes (for any winning door choice)
                    for possible_winning in doors_remaining:
                        removable = [d for d in doors_remaining if d != chosen and d != possible_winning]
                        if removable:
                            for door_to_remove in removable:
                                next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
                                next_step = step + 1
                                next_state = (next_step, next_doors, chosen)
                                if next_state not in self._state_set:
                                    self._state_set.add(next_state)
                                    queue.append(next_state)
                if len(doors_remaining) == 2:
                    if next_state not in self._state_set:
                        self._state_set.add(next_state)
                        queue.append(next_state)

        self.n_states = len(self.states)
        self.state = (0, tuple(range(n_doors)), -1)
        self._winning_door = None
        self._revealed = []

    def state_to_index(self, state):
        step, doors_remaining, chosen = state
        state_key = (step, tuple(sorted(doors_remaining)), chosen)
        return self.states.index(state_key)

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self):
        self._winning_door = random.choice(range(self.n_doors))
        self.state = (0, tuple(range(self.n_doors)), -1)
        self._revealed = []
        return self.state

    def is_terminal(self, state):
        step, doors_remaining, chosen = state
        return step == self.n_doors or len(doors_remaining) == 1

    def get_valid_actions(self, state):
        step, doors_remaining, last_chosen = state
        if self.is_terminal(state):
            return []
        return list(doors_remaining)

    def simulate_step(self, state, action):
        """
        Simulates a step (does not affect self.state).
        Returns (next_state, reward, done).
        """
        step, doors_remaining, last_chosen = state
        doors_remaining = tuple(sorted(doors_remaining))
        if self.is_terminal(state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return state, reward, True

        chosen = action
        # If only 2 doors remain, next state is terminal after this choice
        if len(doors_remaining) == 2:
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
            return next_state, reward, done

        # Otherwise, Monty removes a door (not chosen, not winning)
        removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
        if not removable:
            raise ValueError("No door for Monty to remove!")
        door_to_remove = random.choice(removable)
        next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
        next_state = (step + 1, next_doors, chosen)
        reward = 0.0
        done = False
        return next_state, reward, done

    def step(self, action):
        step, doors_remaining, last_chosen = self.state
        doors_remaining = tuple(sorted(doors_remaining))
        if self.is_terminal(self.state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return self.state, reward, True

        chosen = action
        if len(doors_remaining) == 2:
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
        else:
            removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
            if not removable:
                raise ValueError("No door for Monty to remove!")
            door_to_remove = random.choice(removable)
            self._revealed.append(door_to_remove)
            next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
            next_state = (step + 1, next_doors, chosen)
            reward = 0.0
            done = False
        self.state = next_state
        return next_state, reward, done

    def render(self):
        step, doors_remaining, last_chosen = self.state
        print(f"\nStep {step + 1}/{self.n_doors}:")
        print(f"Doors remaining: {list(doors_remaining)}")
        if step == 0:
            print("Pick your first door!")
        elif step < self.n_doors and len(self._revealed) > 0:
            print(f"Doors revealed by Monty so far: {self._revealed}")
            print(f"Your last chosen door: {last_chosen}")
            print("Pick your next door!")
        elif self.is_terminal(self.state):
            print(f"Final door chosen: {last_chosen}")
            print(f"The winning door was: {self._winning_door}")
            if last_chosen == self._winning_door:
                print("You win!")
            else:
                print("You lose.")
        print()