import sys
import os
import time
import pygame
import random
from collections import defaultdict
import numpy as np

# --- Pour que l'import GridWorld fonctionne ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from envs.grid_world.grid_world import GridWorld

# --- Dyna-Q pour GridWorld ---
def dyna_q_grid(env, num_episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10):
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
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        valid_actions = env.get_valid_actions(state)
        if valid_actions:
            q_values = [Q[state][a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            policy[idx] = random.choice(best_actions)
        else:
            policy[idx] = -1
    return Q, policy, rewards_per_episode, n_explore, n_exploit

# --- Value Iteration pour la politique optimale ---
def value_iteration_grid(env, gamma=0.95, theta=1e-6):
    V = [0.0 for _ in range(env.n_states)]
    while True:
        delta = 0
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
            v = V[idx]
            q_values = []
            for a in env.get_valid_actions(state):
                next_state, reward, _ = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                q = reward + gamma * V[next_idx]
                q_values.append(q)
            V[idx] = max(q_values)
            delta = max(delta, abs(v - V[idx]))
        if delta < theta:
            break
    # Politique optimale
    policy = {}
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        valid_actions = env.get_valid_actions(state)
        if valid_actions:
            q_values = []
            for a in valid_actions:
                next_state, reward, _ = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                q = reward + gamma * V[next_idx]
                q_values.append((a, q))
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [a for a, q in q_values if q == max_q]
            policy[idx] = random.choice(best_actions)
        else:
            policy[idx] = -1
    return V, policy

# --- Pygame Visualisation ---
CELL_SIZE = 80
MARGIN = 20
AGENT_COLOR = (0, 128, 255)
TERMINAL_COLOR = (255, 100, 100)
STATE_COLOR = (220, 220, 220)
POLICY_COLOR = (0, 200, 0)
OPT_POLICY_COLOR = (255, 215, 0)
BG_COLOR = (30, 30, 30)
Q_COLOR = (0, 0, 0)
TRAJECTORY_COLOR = (128, 0, 128)
ARROW_SIZE = 20
ARROW_UNICODE = {0: "↑", 1: "↓", 2: "←", 3: "→"}

def draw_policy(screen, env, policy, color):
    font = pygame.font.SysFont(None, 32)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        row, col = state
        if policy[idx] == -1:
            continue
        x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
        y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
        arrow = ARROW_UNICODE.get(policy[idx], "?")
        txt = font.render(arrow, True, color)
        screen.blit(txt, (x - 10, y - 16))

def draw_q_values(screen, env, Q):
    font = pygame.font.SysFont(None, 18)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        row, col = state
        x = MARGIN + col * CELL_SIZE
        y = MARGIN + row * CELL_SIZE
        for a in env.get_valid_actions(state):
            q = Q[state][a]
            if a == 0:
                pos = (x + CELL_SIZE//2 - 15, y + 2)
            elif a == 1:
                pos = (x + CELL_SIZE//2 - 15, y + CELL_SIZE - 20)
            elif a == 2:
                pos = (x + 2, y + CELL_SIZE//2 - 10)
            elif a == 3:
                pos = (x + CELL_SIZE - 30, y + CELL_SIZE//2 - 10)
            else:
                pos = (x + 5, y + 5)
            txt = font.render(f"Q{a}={q:.2f}", True, Q_COLOR)
            screen.blit(txt, pos)

def draw_rewards(screen, env):
    font = pygame.font.SysFont(None, 18)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        row, col = state
        x = MARGIN + col * CELL_SIZE
        y = MARGIN + row * CELL_SIZE
        r = env.get_reward(state)
        txt = font.render(f"R={r:.1f}", True, (100, 100, 100))
        screen.blit(txt, (x + CELL_SIZE - 38, y + CELL_SIZE - 22))

def draw_trajectory(screen, env, trajectory):
    if len(trajectory) < 2:
        return
    points = [
        (MARGIN + col * CELL_SIZE + CELL_SIZE // 2, MARGIN + row * CELL_SIZE + CELL_SIZE // 2)
        for (row, col) in trajectory
    ]
    pygame.draw.lines(screen, TRAJECTORY_COLOR, False, points, 4)

def visualise_gridworld_pygame(env, policy, Q, rewards_per_episode, opt_policy=None, delay=0.5):
    pygame.init()
    width = env.n_cols * CELL_SIZE + 2 * MARGIN
    height = env.n_rows * CELL_SIZE + 2 * MARGIN + 60
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("GridWorld - Dyna-Q Policy Visualization")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)

    running = True
    state = env.reset()
    done = False
    trajectory = [state]

    while running:
        screen.fill(BG_COLOR)

        # Dessine la grille
        for row in range(env.n_rows):
            for col in range(env.n_cols):
                rect = pygame.Rect(MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                state_here = (row, col)
                color = TERMINAL_COLOR if state_here in env.terminal_states else STATE_COLOR
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                txt = bigfont.render(str(row * env.n_cols + col), True, (0, 0, 0))
                screen.blit(txt, (rect.x + 5, rect.y + 5))

        # Trajectoire de l'agent
        draw_trajectory(screen, env, trajectory)

        # Politique optimale (si fournie)
        if opt_policy:
            draw_policy(screen, env, opt_policy, OPT_POLICY_COLOR)
        # Politique apprise
        draw_policy(screen, env, policy, POLICY_COLOR)

        # Valeurs Q
        draw_q_values(screen, env, Q)
        # Récompenses immédiates
        draw_rewards(screen, env)

        # Agent
        row, col = state
        agent_x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
        agent_y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, AGENT_COLOR, (agent_x, agent_y), CELL_SIZE // 4)

        # Récompense moyenne
        if rewards_per_episode:
            avg_reward = sum(rewards_per_episode) / len(rewards_per_episode)
            txt = bigfont.render(f"Récompense moyenne: {avg_reward:.2f}", True, (255, 255, 255))
            screen.blit(txt, (MARGIN, height - 50))

        pygame.display.flip()

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Avance selon la politique
        if not done:
            action = policy[env.state_to_index(state)]
            if action == -1:
                break
            next_state, reward, done = env.step(action)
            trajectory.append(next_state)
            state = next_state
            time.sleep(delay)
        else:
            time.sleep(1)
            running = False

    pygame.quit()

# --- MAIN ---
if __name__ == '__main__':
    env = GridWorld(n_rows=4, n_cols=4, start_state=(0,0))
    Q, policy, rewards_per_episode, n_explore, n_exploit = dyna_q_grid(env, num_episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10)
    V_opt, opt_policy = value_iteration_grid(env, gamma=0.95)

    # Affichage lisible de la politique apprise
    print("Politique apprise (état : action recommandée) :")
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        action = policy[idx]
        if action == -1:
            action_str = "Terminal"
        elif action == 0:
            action_str = "Haut"
        elif action == 1:
            action_str = "Bas"
        elif action == 2:
            action_str = "Gauche"
        elif action == 3:
            action_str = "Droite"
        else:
            action_str = str(action)
        print(f"  État {state} : {action_str}")

    # Affichage lisible de la politique optimale
    print("Politique optimale (état : action optimale) :")
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        action = opt_policy[idx]
        if action == -1:
            action_str = "Terminal"
        elif action == 0:
            action_str = "Haut"
        elif action == 1:
            action_str = "Bas"
        elif action == 2:
            action_str = "Gauche"
        elif action == 3:
            action_str = "Droite"
        else:
            action_str = str(action)
        print(f"  État {state} : {action_str}")

    visualise_gridworld_pygame(env, policy, Q, rewards_per_episode, opt_policy=opt_policy, delay=0.5)

    # Test manuel en console avec contrôle clavier (flèches)
    print("\n=== Test manuel de GridWorld (contrôle clavier flèches) ===")
    print("Utilise les flèches du clavier pour déplacer l'agent. Ferme la fenêtre pour quitter.")
    pygame.init()
    width = env.n_cols * CELL_SIZE + 2 * MARGIN
    height = env.n_rows * CELL_SIZE + 2 * MARGIN + 60
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("GridWorld - Test manuel")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)
    state = env.reset()
    done = False
    while True:
        screen.fill(BG_COLOR)
        for row in range(env.n_rows):
            for col in range(env.n_cols):
                rect = pygame.Rect(MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                state_here = (row, col)
                color = TERMINAL_COLOR if state_here in env.terminal_states else STATE_COLOR
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                txt = bigfont.render(str(row * env.n_cols + col), True, (0, 0, 0))
                screen.blit(txt, (rect.x + 5, rect.y + 5))
        # Agent
        row, col = state
        agent_x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
        agent_y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, AGENT_COLOR, (agent_x, agent_y), CELL_SIZE // 4)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if done:
                    continue
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                else:
                    continue
                if action in env.get_valid_actions(state):
                    next_state, reward, done = env.step(action)
                    print(f"Action : {action} | Nouvel état : {next_state} | Récompense : {reward} | Terminé : {done}")
                    state = next_state
                else:
                    print("Action invalide depuis cet état.")
        if done:
            print("État terminal atteint. Ferme la fenêtre pour quitter.") 