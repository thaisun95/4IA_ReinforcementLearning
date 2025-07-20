import numpy as np
import random

class TwoRoundRPS:
    """
    Environnement Rock-Paper-Scissors à deux manches (Two-Round RPS).
    - Etat = (round, my_first_move)
        round: 0 (premier round), 1 (second round), 2 (terminal)
        my_first_move: -1 (pas encore joué), 0=Rock, 1=Paper, 2=Scissors
    - Actions: 0=Rock, 1=Paper, 2=Scissors
    """

    def __init__(self):
        """
        Initialise l'environnement TwoRoundRPS.
        """
        self.action_space = [0, 1, 2]  # 0=Rock, 1=Paper, 2=Scissors
        # Définition des états possibles
        self.states = [(0, -1)]  # round 1, pas encore joué
        self.states += [(1, a) for a in self.action_space]  # round 2, on se souvient du 1er coup
        self.states += [(2, a) for a in self.action_space]  # états terminaux
        self.n_states = len(self.states)
        self.state = (0, -1)  # état initial
        self.last_opp_move = None  # pour affichage dans render

    def state_to_index(self, state):
        """
        Convertit un état en son indice dans la liste self.states.
        """
        return self.states.index(state)

    def index_to_state(self, index):
        """
        Convertit un indice en état dans la liste self.states.
        """
        return self.states[index]

    def reset(self):
        """
        Réinitialise l'environnement à l'état initial.
        Retourne:
            tuple: état initial (0, -1)
        """
        self.state = (0, -1)
        self.last_opp_move = None
        return self.state

    def is_terminal(self, state):
        """
        Indique si un état est terminal.
        Args:
            state (tuple): état sous forme (round, my_first_move)
        Retourne:
            bool: True si terminal
        """
        return state[0] == 2

    def get_valid_actions(self, state):
        """
        Retourne la liste des actions valides pour un état donné.
        Args:
            state (tuple): état (round, my_first_move)
        Retourne:
            list: liste des actions valides
        """
        round_id, my_first_move = state
        if round_id in [0, 1]:
            return [0, 1, 2]
        else:
            return []

    def get_reward(self, my_move, opp_move):
        """
        Calcule la récompense d'une manche selon le résultat.
        Args:
            my_move (int): 0, 1, 2
            opp_move (int): 0, 1, 2
        Retourne:
            int: 1=victoire, 0=nul, -1=défaite
        """
        if my_move == opp_move:
            return 0
        elif (my_move == 0 and opp_move == 2) or (my_move == 1 and opp_move == 0) or (my_move == 2 and opp_move == 1):
            return 1
        else:
            return -1

    def simulate_step(self, state, action):
        """
        Simule une action depuis un état donné (ne modifie PAS self.state).
        Args:
            state (tuple): état courant (round, my_first_move)
            action (int): 0, 1 ou 2
        Retourne:
            next_state (tuple), reward (float), done (bool)
        """
        round_id, my_first_move = state
        if round_id == 0:
            opp_move = random.choice(self.action_space)  # l'adversaire joue aléatoire au 1er tour
            reward = self.get_reward(action, opp_move)
            next_state = (1, action)
            done = False
        elif round_id == 1:
            opp_move = my_first_move  # l'adversaire copie le 1er coup de l'agent
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
        Joue une action dans l'environnement (modifie self.state).
        Args:
            action (int): 0, 1 ou 2
        Retourne:
            next_state, reward, done
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
        Affiche un état humainement lisible de l'environnement.
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