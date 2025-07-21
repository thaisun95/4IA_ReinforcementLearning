import sys
import os
import time
import pygame
import random
from collections import defaultdict, deque

# --- Classe MontyHallV2 (copiée ici pour autonomie) ---
class MontyHallV2:
    def __init__(self, n_doors=5):
        self.n_doors = n_doors
        self.action_space = list(range(n_doors))
        self.states = []
        self._state_set = set()
        initial_state = (0, tuple(range(n_doors)), -1)
        queue = deque([initial_state])
        self._state_set.add(initial_state)
        while queue:
            state = queue.popleft()
            self.states.append(state)
            step, doors_remaining, last_chosen = state
            if step == n_doors or len(doors_remaining) == 1:
                continue
            for action in doors_remaining:
                chosen = action
                if len(doors_remaining) == 2:
                    next_state = (n_doors, tuple(sorted(doors_remaining)), chosen)
                else:
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
        self.state = (0, tuple(range(self.n_doors)), -1)
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
        step, doors_remaining, last_chosen = state
        doors_remaining = tuple(sorted(doors_remaining))
        if self.is_terminal(state):
            reward = 1.0 if last_chosen == self._winning_door else 0.0
            return state, reward, True
        chosen = action
        if len(doors_remaining) == 2:
            next_state = (self.n_doors, doors_remaining, chosen)
            reward = 1.0 if chosen == self._winning_door else 0.0
            done = True
            return next_state, reward, done
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

# --- Dyna-Q pour Monty Hall V2 ---
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

# --- Politique optimale (toujours switcher si possible) ---
def get_opt_policy_mh2(env):
    opt_policy = {}
    for s in env.states:
        step, doors_remaining, last_chosen = s
        if env.is_terminal(s):
            opt_policy[s] = None
        else:
            for a in env.get_valid_actions(s):
                if a != last_chosen:
                    opt_policy[s] = a
                    break
            else:
                opt_policy[s] = last_chosen
    return opt_policy

# --- Pygame Visualisation ---
WIN = 800
HEIGHT = 400
BG_COLOR = (30, 30, 30)
Q_COLOR = (0, 0, 0)
POLICY_COLOR = (0, 200, 0)
OPT_POLICY_COLOR = (255, 215, 0)

def render_state_pygame(screen, state, step, doors_remaining, last_chosen, reward, done, Q, policy, opt_policy, font, bigfont):
    screen.fill(BG_COLOR)
    txt = bigfont.render(f"Step: {step}", True, (255,255,255))
    screen.blit(txt, (20, 20))
    txt = font.render(f"Doors remaining: {list(doors_remaining)}", True, (200,200,255))
    screen.blit(txt, (20, 70))
    txt = font.render(f"Last chosen: {last_chosen}", True, (200,200,255))
    screen.blit(txt, (20, 100))
    txt = font.render(f"Reward: {reward}", True, (255,255,0))
    screen.blit(txt, (20, 130))
    txt = font.render(f"Done: {done}", True, (255,255,0))
    screen.blit(txt, (20, 160))
    # Affiche Q pour chaque action possible
    y0 = 200
    txt = font.render("Valeurs Q :", True, (255,255,255))
    screen.blit(txt, (20, y0))
    for i, a in enumerate(doors_remaining):
        q = Q[state][a]
        txt = font.render(f"Action {a} : Q={q:.2f}", True, Q_COLOR)
        screen.blit(txt, (40, y0+25+i*22))
    # Affiche la politique apprise et optimale
    txt = font.render(f"Politique apprise : {policy[state]}", True, POLICY_COLOR)
    screen.blit(txt, (400, 70))
    txt = font.render(f"Politique optimale : {opt_policy[state]}", True, OPT_POLICY_COLOR)
    screen.blit(txt, (400, 100))
    pygame.display.flip()

# --- MAIN ---
if __name__ == '__main__':
    env = MontyHallV2(n_doors=5)
    Q, policy, rewards_per_episode, n_explore, n_exploit = dyna_q_monty_v2(env, num_episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10)
    opt_policy = get_opt_policy_mh2(env)

    # Affichage console (tronqué)
    print("Politique apprise (état : action recommandée) :")
    for s in env.states[:10]:
        print(f"  État {s} : {policy[s]}")
    print("... (affichage tronqué)")
    print("Politique optimale (état : action optimale) :")
    for s in env.states[:10]:
        print(f"  État {s} : {opt_policy[s]}")
    print("... (affichage tronqué)")

    # Visualisation Pygame automatique (par la politique apprise)
    pygame.init()
    screen = pygame.display.set_mode((WIN, HEIGHT))
    pygame.display.set_caption("Monty Hall V2 Dyna-Q Visualization")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    reward = 0
    while not done:
        step, doors_remaining, last_chosen = state
        render_state_pygame(screen, state, step, doors_remaining, last_chosen, reward, done, Q, policy, opt_policy, font, bigfont)
        time.sleep(0.7)
        action = policy[state]
        if action is None:
            break
        state, reward, done = env.step(action)
    time.sleep(1)
    pygame.quit()

    # Test manuel
    print("\n=== Test manuel de Monty Hall V2 (clavier) ===")
    print("À chaque étape, choisis une porte parmi celles restantes (0, 1, 2, ...) avec le clavier. Ferme la fenêtre pour quitter.")
    pygame.init()
    screen = pygame.display.set_mode((WIN, HEIGHT))
    pygame.display.set_caption("Monty Hall V2 - Test manuel")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    reward = 0
    while True:
        step, doors_remaining, last_chosen = state
        render_state_pygame(screen, state, step, doors_remaining, last_chosen, reward, done, Q, policy, opt_policy, font, bigfont)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if done:
                    continue
                # On accepte les touches 0 à 9 pour choisir une porte
                if event.unicode.isdigit():
                    action = int(event.unicode)
                    if action in env.get_valid_actions(state):
                        state, reward, done = env.step(action)
                        print(f"Action : {action} | Nouvel état : {state} | Récompense : {reward} | Terminé : {done}")
                    else:
                        print("Action invalide depuis cet état.")
        if done:
            print("Épisode terminé. Ferme la fenêtre pour quitter.") 