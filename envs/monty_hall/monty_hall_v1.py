import numpy as np
import random

class MontyHallV1:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 0/1/2=choisir porte, 3=stay, 4=switch
        # States: (step, chosen, remaining) where step=0 (choose), step=1 (stay/switch), step=2 (terminal)
        self.states = []
        # Step 0: agent chooses among 0,1,2, chosen=-1, remaining=-1
        self.states += [(0, -1, -1)]
        # Step 1: agent can choose to stay/switch, chosen=0/1/2, remaining=other
        for chosen in range(3):
            for remaining in range(3):
                if remaining != chosen:
                    self.states.append((1, chosen, remaining))
        # Step 2: terminal
        for chosen in range(3):
            self.states.append((2, chosen, -1))
        self.n_states = len(self.states)
        self.state = (0, -1, -1)
        self._winning_door = None  # kept internal for proper step logic

    def state_to_index(self, state):
        return self.states.index(state)

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self):
        self._winning_door = random.choice([0, 1, 2])
        self.state = (0, -1, -1)
        return self.state

    def is_terminal(self, state):
        return state[0] == 2

    def simulate_step(self, state, action):
        step, chosen, remaining = state
        if step == 0:
            # First action: pick a door (0, 1, 2)
            chosen = action
            # Monty removes one non-winning, non-chosen door
            non_chosen = [d for d in range(3) if d != chosen]
            # Remove a door that is NOT the winning door if possible
            if self._winning_door == chosen:
                monty_opens = random.choice(non_chosen)
            else:
                monty_opens = [d for d in non_chosen if d != self._winning_door][0]
            remaining_door = [d for d in range(3) if d not in [chosen, monty_opens]][0]
            next_state = (1, chosen, remaining_door)
            reward = 0
            done = False
        elif step == 1:
            # Second action: 0=stay, 1=switch
            if action == 0:  # stay
                final_choice = chosen
            elif action == 1:  # switch
                final_choice = remaining
            else:
                raise ValueError("Invalid action for MontyHall step 1 (0=stay, 1=switch)")
            next_state = (2, final_choice, -1)
            reward = 1.0 if final_choice == self._winning_door else 0.0
            done = True
        else:
            next_state = state
            reward = 0
            done = True
        return next_state, reward, done