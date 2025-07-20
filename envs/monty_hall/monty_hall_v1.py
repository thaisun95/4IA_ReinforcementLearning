import numpy as np
import random

class MontyHallV1:
    def __init__(self):
        self.action_space = [0, 1, 2, 0, 1]  # 0/1/2=pick door, 0=stay, 1=switch (for step 1)
        # States: (step, chosen, remaining) where step=0 (choose), step=1 (stay/switch), step=2 (terminal)
        self.states = [(0, -1, -1)]
        for chosen in range(3):
            for remaining in range(3):
                if remaining != chosen:
                    self.states.append((1, chosen, remaining))
        for chosen in range(3):
            self.states.append((2, chosen, -1))
        self.n_states = len(self.states)
        self.state = (0, -1, -1)
        self._winning_door = None  # hidden from agent
        self._revealed = None      # door revealed by Monty

    def state_to_index(self, state):
        return self.states.index(state)

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self):
        self._winning_door = random.choice([0, 1, 2])
        self.state = (0, -1, -1)
        self._revealed = None
        return self.state

    def is_terminal(self, state):
        return state[0] == 2

    def simulate_step(self, state, action):
        # Use simulate_step for RL algorithms, does not modify self.state!
        step, chosen, remaining = state
        if step == 0:
            chosen = action
            non_chosen = [d for d in range(3) if d != chosen]
            if self._winning_door == chosen:
                monty_opens = random.choice(non_chosen)
            else:
                monty_opens = [d for d in non_chosen if d != self._winning_door][0]
            remaining_door = [d for d in range(3) if d not in [chosen, monty_opens]][0]
            next_state = (1, chosen, remaining_door)
            reward = 0
            done = False
        elif step == 1:
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

    def step(self, action):
        # Human/manual play, updates self.state
        step, chosen, remaining = self.state
        if step == 0:
            chosen = action
            non_chosen = [d for d in range(3) if d != chosen]
            if self._winning_door == chosen:
                monty_opens = random.choice(non_chosen)
            else:
                monty_opens = [d for d in non_chosen if d != self._winning_door][0]
            self._revealed = monty_opens
            remaining_door = [d for d in range(3) if d not in [chosen, monty_opens]][0]
            next_state = (1, chosen, remaining_door)
            reward = 0
            done = False
        elif step == 1:
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
            next_state = self.state
            reward = 0
            done = True
        self.state = next_state
        return next_state, reward, done

    def render(self):
        # Pretty print state (for demo or manual play)
        step, chosen, remaining = self.state
        if step == 0:
            print("Step 1: Choose a door (0, 1, 2)")
        elif step == 1:
            print(f"Step 2: You picked door {chosen}. Monty opens door {self._revealed} (not winning, not yours).")
            print(f"Do you Stay (0) with door {chosen} or Switch (1) to door {remaining}?")
        elif step == 2:
            print(f"Final choice: door {chosen}")
            print(f"The winning door was: {self._winning_door}")
            if chosen == self._winning_door:
                print("You win! 🎉")
            else:
                print("You lose.")
        print()