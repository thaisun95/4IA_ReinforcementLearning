import numpy as np
import random
from itertools import combinations
from collections import deque
import random
from collections import defaultdict
import matplotlib.pyplot as plt

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

## Fonction Dyna-Q pour Monty Hall Level 2 avec statistiques exploration/exploitation

def dyna_q_monty_v2(env, num_episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10):
    Q = defaultdict(lambda: defaultdict(float))
    Model = defaultdict(lambda: defaultdict(lambda: None))
    visited_state_actions = set()
    rewards_per_episode = []
    n_explore = 0
    n_exploit = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
            if random.random() < epsilon:
                action = random.choice(valid_actions)
                n_explore += 1
            else:
                q_values = [Q[state][a] for a in valid_actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                action = random.choice(best_actions)
                n_exploit += 1
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_valid_actions = env.get_valid_actions(next_state)
            if next_valid_actions:
                max_next_q = max([Q[next_state][a] for a in next_valid_actions])
            else:
                max_next_q = 0.0
            Q[state][action] += alpha * (reward + gamma * max_next_q - Q[state][action])
            Model[state][action] = (reward, next_state)
            visited_state_actions.add((state, action))
            for _ in range(n_planning):
                s_p, a_p = random.choice(list(visited_state_actions))
                r_p, s2_p = Model[s_p][a_p]
                next_valid_actions_p = env.get_valid_actions(s2_p)
                if next_valid_actions_p:
                    max_next_q_p = max([Q[s2_p][a] for a in next_valid_actions_p])
                else:
                    max_next_q_p = 0.0
                Q[s_p][a_p] += alpha * (r_p + gamma * max_next_q_p - Q[s_p][a_p])
            state = next_state
        rewards_per_episode.append(total_reward)
    # Politique greedy
    policy = {}
    for s in env.states:
        valid_actions = env.get_valid_actions(s)
        if valid_actions:
            q_values = [Q[s][a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            policy[s] = random.choice(best_actions)
        else:
            policy[s] = None
    return Q, policy, rewards_per_episode, n_explore, n_exploit

## Entraînement Dyna-Q sur Monty Hall Level 2 et affichage de la politique apprise

env_mh2 = MontyHallV2(n_doors=5)
Q_mh2, policy_mh2, rewards_mh2, n_explore_mh2, n_exploit_mh2 = dyna_q_monty_v2(env_mh2, num_episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10)
print("Politique apprise (état : action recommandée) :")
for s in env_mh2.states[:15]:  # Affiche les 15 premiers états pour lisibilité
    print(f"  État {s} : {policy_mh2[s]}")

## Politique optimale (stratégie : toujours switcher)

# Pour Monty Hall généralisé, la stratégie optimale est de toujours switcher à chaque étape
opt_policy_mh2 = {}
for s in env_mh2.states:
    step, doors_remaining, last_chosen = s
    if env_mh2.is_terminal(s):
        opt_policy_mh2[s] = None
    else:
        # Toujours choisir une porte différente de la précédente si possible
        valid = env_mh2.get_valid_actions(s)
        if step == 0:
            opt_policy_mh2[s] = valid[0]  # premier choix arbitraire
        else:
            # switch = choisir une porte différente de last_chosen
            switch_doors = [d for d in valid if d != last_chosen]
            opt_policy_mh2[s] = switch_doors[0] if switch_doors else valid[0]
print("Politique optimale (état : action optimale) :")
for s in env_mh2.states[:15]:
    print(f"  État {s} : {opt_policy_mh2[s]}")

## Comparaison automatique des politiques (apprise vs optimale)

print('Comparaison des politiques (affichage sur 15 premiers états) :')
for s in env_mh2.states[:15]:
    pa = policy_mh2[s]
    po = opt_policy_mh2[s]
    if pa == po:
        res = 'OK'
    else:
        res = 'DIFF'
    print(f"État {s} : apprise={pa} / optimale={po} --> {res}")

## Courbe d'apprentissage : récompense cumulée par épisode

plt.figure(figsize=(8,4))
plt.plot(rewards_mh2)
plt.xlabel('Épisode')
plt.ylabel('Récompense cumulée')
plt.title("Courbe d'apprentissage Dyna-Q sur Monty Hall Level 2")
plt.grid(True)
plt.show()

## Statistiques exploration vs exploitation pendant l'apprentissage

print(f"Nombre d'actions explorées (aléatoires) : {n_explore_mh2}")
print(f"Nombre d'actions exploitées (greedy) : {n_exploit_mh2}")
print(f"Taux d'exploration : {n_explore_mh2/(n_explore_mh2+n_exploit_mh2):.2%}")

## Tester différents paramètres (alpha, gamma, epsilon, n_planning)

# Modifie les paramètres ici pour tester leur effet
alpha = 0.1
gamma = 0.95
epsilon = 0.1
n_planning = 10
Q2_mh2, policy2_mh2, rewards2_mh2, _, _ = dyna_q_monty_v2(env_mh2, num_episodes=3000, alpha=alpha, gamma=gamma, epsilon=epsilon, n_planning=n_planning)
plt.figure(figsize=(8,4))
plt.plot(rewards2_mh2)
plt.xlabel('Épisode')
plt.ylabel('Récompense cumulée')
plt.title(f'alpha={alpha}, gamma={gamma}, epsilon={epsilon}, n_planning={n_planning}')
plt.grid(True)
plt.show()