import numpy as np
import random

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

    def render(self):
        """
        Print a human-readable representation of the environment state.
        """
        move_map = {0: "Rock", 1: "Paper", 2: "Scissors", -1: "--"}
        round_id, my_first_move = self.state
        print(f"Round: {round_id + 1 if round_id < 2 else 'Terminal'}")
        if round_id == 0:
            print(f"Agent's move:    --")
            print(f"Opponent's move: --")
        elif round_id == 1:
            print(f"Agent's first move: {move_map[my_first_move]}")
            print(f"Agent's move:    --")
            print(f"Opponent's move: -- (will be revealed after your move)")
        else:
            print(f"Agent's first move: {move_map[my_first_move]}")
            print(f"Last opponent move: {move_map.get(self.last_opp_move, '--')}")
        print()