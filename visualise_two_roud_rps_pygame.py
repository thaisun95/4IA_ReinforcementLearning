import sys
import os
import time
import pygame
import random
from collections import defaultdict

# --- Classe TwoRoundRPS (copiée ici pour autonomie) ---
class TwoRoundRPS:
    def __init__(self):
        self.action_space = [0, 1, 2]  # 0=Rock, 1=Paper, 2=Scissors
        self.states = [(0, -1)]
        self.states += [(1, a) for a in self.action_space]
        self.states += [(2, a) for a in self.action_space]
        self.n_states = len(self.states)
        self.state = (0, -1)
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
    def get_valid_actions(self, state):
        round_id, my_first_move = state
        if round_id in [0, 1]:
            return [0, 1, 2]
        else:
            return []
    def get_reward(self, my_move, opp_move):
        if my_move == opp_move:
            return 0
        elif (my_move == 0 and opp_move == 2) or (my_move == 1 and opp_move == 0) or (my_move == 2 and opp_move == 1):
            return 1
        else:
            return -1
    def simulate_step(self, state, action):
        round_id, my_first_move = state
        if round_id == 0:
            opp_move = random.choice(self.action_space)
            reward = self.get_reward(action, opp_move)
            next_state = (1, action)
            done = False
        elif round_id == 1:
            opp_move = my_first_move
            reward = self.get_reward(action, opp_move)
            next_state = (2, my_first_move)
            done = True
        else:
            next_state = state
            reward = 0
            done = True
        return next_state, reward, done
    def step(self, action):
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

# --- Dyna-Q pour Two-Round RPS ---
def dyna_q_rps(env, num_episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10):
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

# --- Politique optimale (stratégie théorique) ---
def get_opt_policy_rps(env):
    opt_policy = {}
    for s in env.states:
        round_id, my_first_move = s
        if round_id == 0:
            opt_policy[s] = 0  # ou 1 ou 2, équivalent
        elif round_id == 1:
            if my_first_move == 0:
                opt_policy[s] = 1  # Paper bat Rock
            elif my_first_move == 1:
                opt_policy[s] = 2  # Scissors bat Paper
            elif my_first_move == 2:
                opt_policy[s] = 0  # Rock bat Scissors
            else:
                opt_policy[s] = 0
        else:
            opt_policy[s] = None
    return opt_policy

# --- Pygame Visualisation ---
WIN = 600
HEIGHT = 350
BG_COLOR = (30, 30, 30)
Q_COLOR = (0, 0, 0)
POLICY_COLOR = (0, 200, 0)
OPT_POLICY_COLOR = (255, 215, 0)
MOVE_MAP = {0: "Rock", 1: "Paper", 2: "Scissors", -1: "--"}

def render_state_pygame(screen, state, round_id, my_first_move, reward, done, Q, policy, opt_policy, font, bigfont):
    screen.fill(BG_COLOR)
    txt = bigfont.render(f"Round: {round_id + 1 if round_id < 2 else 'Terminal'}", True, (255,255,255))
    screen.blit(txt, (20, 20))
    txt = font.render(f"First move: {MOVE_MAP[my_first_move]}", True, (200,200,255))
    screen.blit(txt, (20, 70))
    txt = font.render(f"Reward: {reward}", True, (255,255,0))
    screen.blit(txt, (20, 110))
    txt = font.render(f"Done: {done}", True, (255,255,0))
    screen.blit(txt, (20, 140))
    # Affiche Q pour chaque action possible
    y0 = 180
    txt = font.render("Valeurs Q :", True, (255,255,255))
    screen.blit(txt, (20, y0))
    for i, a in enumerate([0,1,2]):
        q = Q[state][a]
        txt = font.render(f"Action {a} ({MOVE_MAP[a]}) : Q={q:.2f}", True, Q_COLOR)
        screen.blit(txt, (40, y0+25+i*22))
    # Affiche la politique apprise et optimale
    txt = font.render(f"Politique apprise : {policy[state]}", True, POLICY_COLOR)
    screen.blit(txt, (320, 70))
    txt = font.render(f"Politique optimale : {opt_policy[state]}", True, OPT_POLICY_COLOR)
    screen.blit(txt, (320, 100))
    pygame.display.flip()

# --- MAIN ---
if __name__ == '__main__':
    env = TwoRoundRPS()
    Q, policy, rewards_per_episode, n_explore, n_exploit = dyna_q_rps(env, num_episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10)
    opt_policy = get_opt_policy_rps(env)

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
    pygame.display.set_caption("Two-Round RPS Dyna-Q Visualization")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    reward = 0
    while not done:
        round_id, my_first_move = state
        render_state_pygame(screen, state, round_id, my_first_move, reward, done, Q, policy, opt_policy, font, bigfont)
        time.sleep(0.7)
        action = policy[state]
        if action is None:
            break
        state, reward, done = env.step(action)
    time.sleep(1)
    pygame.quit()

    # Test manuel
    print("\n=== Test manuel de Two-Round RPS (clavier) ===")
    print("À chaque round, choisis une action (0=Rock, 1=Paper, 2=Scissors) avec le clavier. Ferme la fenêtre pour quitter.")
    pygame.init()
    screen = pygame.display.set_mode((WIN, HEIGHT))
    pygame.display.set_caption("Two-Round RPS - Test manuel")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    reward = 0
    while True:
        round_id, my_first_move = state
        render_state_pygame(screen, state, round_id, my_first_move, reward, done, Q, policy, opt_policy, font, bigfont)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if done:
                    continue
                if event.unicode in ['0','1','2']:
                    action = int(event.unicode)
                    if action in env.get_valid_actions(state):
                        state, reward, done = env.step(action)
                        print(f"Action : {action} | Nouvel état : {state} | Récompense : {reward} | Terminé : {done}")
                    else:
                        print("Action invalide depuis cet état.")
        if done:
            print("Épisode terminé. Ferme la fenêtre pour quitter.") 