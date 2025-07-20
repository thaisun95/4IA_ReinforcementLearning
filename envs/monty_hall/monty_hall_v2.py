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
        self.action_space = list(range(n_doors))
        self.states = []
        self.states.append((0, tuple(range(n_doors)), -1))
        for step in range(1, n_doors):
            for doors_remaining in self._all_possible_remaining_doors(n_doors, n_doors - step + 1):
                for chosen in doors_remaining:
                    self.states.append((step, doors_remaining, chosen))
        for doors_remaining in self._all_possible_remaining_doors(n_doors, 1):
            for chosen in doors_remaining:
                self.states.append((n_doors, doors_remaining, chosen))
        self.n_states = len(self.states)
        self.state = (0, tuple(range(n_doors)), -1)
        self._winning_door = None
        self._revealed = []

    def _all_possible_remaining_doors(self, n_doors, k):
        from itertools import combinations
        return [tuple(sorted(comb)) for comb in combinations(range(n_doors), k)]

    def state_to_index(self, state):
        return self.states.index(state)

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self):
        self._winning_door = random.choice(range(self.n_doors))
        self.state = (0, tuple(range(self.n_doors)), -1)
        self._revealed = []
        return self.state

    def is_terminal(self, state):
        step, doors_remaining, chosen = state
        return step == self.n_doors

    def simulate_step(self, state, action):
        step, doors_remaining, last_chosen = state
        if self.is_terminal(state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return state, reward, True

        chosen = action
        if step < self.n_doors - 1:
            removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
            if len(removable) == 0:
                raise ValueError("No door for Monty to remove!")
            door_to_remove = random.choice(removable)
            next_doors = tuple(sorted([d for d in doors_remaining if d != door_to_remove]))
            next_state = (step + 1, next_doors, chosen)
            reward = 0.0
            done = False
        else:
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
        return next_state, reward, done

    def step(self, action):
        step, doors_remaining, last_chosen = self.state
        if self.is_terminal(self.state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return self.state, reward, True

        chosen = action
        if step < self.n_doors - 1:
            removable = [d for d in doors_remaining if d != chosen and d != self._winning_door]
            if len(removable) == 0:
                raise ValueError("No door for Monty to remove!")
            door_to_remove = random.choice(removable)
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

    def render(self):
        step, doors_remaining, last_chosen = self.state
        print(f"\nStep {step+1}/{self.n_doors}:")
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