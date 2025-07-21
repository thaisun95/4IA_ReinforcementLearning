import sys
import os
import time
import pygame
import random
from collections import defaultdict

# --- Classe MontyHallV1 (copiée ici pour autonomie) ---
class MontyHallV1:
    def __init__(self):
        self.states = [(0, -1, -1)]
        for chosen in range(3):
            for remaining in range(3):
                if remaining != chosen:
                    self.states.append((1, chosen, remaining))
        for chosen in range(3):
            self.states.append((2, chosen, -1))
        self.n_states = len(self.states)
        self.state = (0, -1, -1)
        self._winning_door = None
        self._revealed = None
        self.action_space = [0, 1, 2]

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

    def get_valid_actions(self, state):
        step, chosen, remaining = state
        if step == 0:
            return [0, 1, 2]
        elif step == 1:
            return [0, 1]   # 0=stay, 1=switch
        else:
            return []

    def simulate_step(self, state, action):
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
            if action == 0:
                final_choice = chosen
            elif action == 1:
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
            if action == 0:
                final_choice = chosen
            elif action == 1:
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
                print("You win! ")
            else:
                print("You lose.")
        print()

# --- Dyna-Q pour Monty Hall ---
def dyna_q_monty(env, num_episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10):
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

# --- Politique optimale analytique ---
def get_opt_policy_mh(env):
    opt_policy = {}
    for s in env.states:
        step, chosen, remaining = s
        if step == 0:
            opt_policy[s] = 0  # ou 1 ou 2, équivalent
        elif step == 1:
            opt_policy[s] = 1  # switch
        else:
            opt_policy[s] = None
    return opt_policy

# --- Pygame Visualisation ---
WIN = 600
HEIGHT = 300
BG_COLOR = (30, 30, 30)
AGENT_COLOR = (0, 128, 255)
Q_COLOR = (0, 0, 0)
POLICY_COLOR = (0, 200, 0)
OPT_POLICY_COLOR = (255, 215, 0)

def render_state_pygame(screen, state, step, chosen, remaining, reward, done, Q, policy, opt_policy, font, bigfont):
    screen.fill(BG_COLOR)
    txt = bigfont.render(f"Step: {step}", True, (255,255,255))
    screen.blit(txt, (20, 20))
    txt = font.render(f"Chosen: {chosen}", True, (200,200,255))
    screen.blit(txt, (20, 70))
    txt = font.render(f"Remaining: {remaining}", True, (200,200,255))
    screen.blit(txt, (20, 100))
    txt = font.render(f"Reward: {reward}", True, (255,255,0))
    screen.blit(txt, (20, 130))
    txt = font.render(f"Done: {done}", True, (255,255,0))
    screen.blit(txt, (20, 160))
    # Affiche Q pour chaque action possible
    y0 = 200
    txt = font.render("Valeurs Q :", True, (255,255,255))
    screen.blit(txt, (20, y0))
    for i, a in enumerate(env.get_valid_actions(state)):
        q = Q[state][a]
        txt = font.render(f"Action {a} : Q={q:.2f}", True, Q_COLOR)
        screen.blit(txt, (40, y0+25+i*22))
    # Affiche la politique apprise et optimale
    txt = font.render(f"Politique apprise : {policy[state]}", True, POLICY_COLOR)
    screen.blit(txt, (300, 70))
    txt = font.render(f"Politique optimale : {opt_policy[state]}", True, OPT_POLICY_COLOR)
    screen.blit(txt, (300, 100))
    pygame.display.flip()

# --- MAIN ---
if __name__ == '__main__':
    env = MontyHallV1()
    Q, policy, rewards_per_episode, n_explore, n_exploit = dyna_q_monty(env, num_episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10)
    opt_policy = get_opt_policy_mh(env)

    # Affichage console
    print("Politique apprise (état : action recommandée) :")
    for s in env.states:
        print(f"  État {s} : {policy[s]}")
    print("Politique optimale (état : action optimale) :")
    for s in env.states:
        print(f"  État {s} : {opt_policy[s]}")

    # Visualisation Pygame automatique (par la politique apprise)
    pygame.init()
    screen = pygame.display.set_mode((WIN, HEIGHT))
    pygame.display.set_caption("Monty Hall Dyna-Q Visualization")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    reward = 0
    while not done:
        step, chosen, remaining = state
        render_state_pygame(screen, state, step, chosen, remaining, reward, done, Q, policy, opt_policy, font, bigfont)
        time.sleep(0.7)
        action = policy[state]
        if action is None:
            break
        state, reward, done = env.step(action)
    time.sleep(1)
    pygame.quit()

    # Test manuel
    print("\n=== Test manuel de Monty Hall (clavier) ===")
    print("Étape 1 : Choisis une porte (0, 1, 2) avec le clavier.\nÉtape 2 : 0=Stay, 1=Switch. Ferme la fenêtre pour quitter.")
    pygame.init()
    screen = pygame.display.set_mode((WIN, HEIGHT))
    pygame.display.set_caption("Monty Hall - Test manuel")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    reward = 0
    while True:
        step, chosen, remaining = state
        render_state_pygame(screen, state, step, chosen, remaining, reward, done, Q, policy, opt_policy, font, bigfont)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if done:
                    continue
                if step == 0 and event.key in [pygame.K_0, pygame.K_1, pygame.K_2]:
                    action = int(event.unicode)
                elif step == 1 and event.key in [pygame.K_0, pygame.K_1]:
                    action = int(event.unicode)
                else:
                    continue
                if action in env.get_valid_actions(state):
                    state, reward, done = env.step(action)
                    print(f"Action : {action} | Nouvel état : {state} | Récompense : {reward} | Terminé : {done}")
                else:
                    print("Action invalide depuis cet état.")
        if done:
            print("Épisode terminé. Ferme la fenêtre pour quitter.") 