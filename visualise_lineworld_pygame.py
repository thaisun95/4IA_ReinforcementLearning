import sys
import os
import time
import pygame
import random
from collections import defaultdict

# --- Pour que l'import LineWorld fonctionne ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from envs.line_world.line_world import LineWorld

# --- Dyna-Q ---
def dyna_q(env, num_episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10):
    Q = defaultdict(lambda: defaultdict(float))
    Model = defaultdict(lambda: defaultdict(lambda: None))
    visited_state_actions = set()
    rewards_per_episode = []
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
            else:
                q_values = [Q[state][a] for a in valid_actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                action = random.choice(best_actions)
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
    for s in range(env.n_states):
        valid_actions = env.get_valid_actions(s)
        if valid_actions:
            q_values = [Q[s][a] for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            policy[s] = random.choice(best_actions)
        else:
            policy[s] = None
    return Q, policy, rewards_per_episode

# --- Value Iteration pour la politique optimale ---
def value_iteration(env, gamma=0.95, theta=1e-6):
    V = [0.0 for _ in range(env.n_states)]
    while True:
        delta = 0
        for s in range(env.n_states):
            if env.is_terminal(s):
                continue
            v = V[s]
            q_values = []
            for a in env.get_valid_actions(s):
                next_state, reward, _ = env.simulate_step(s, a)
                q = reward + gamma * V[next_state]
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # Politique optimale
    policy = {}
    for s in range(env.n_states):
        valid_actions = env.get_valid_actions(s)
        if valid_actions:
            q_values = []
            for a in valid_actions:
                next_state, reward, _ = env.simulate_step(s, a)
                q = reward + gamma * V[next_state]
                q_values.append((a, q))
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [a for a, q in q_values if q == max_q]
            policy[s] = random.choice(best_actions)
        else:
            policy[s] = None
    return V, policy

# --- Pygame Visualisation ---
CASE_SIZE = 80
MARGIN = 20
AGENT_COLOR = (0, 128, 255)
TERMINAL_COLOR = (255, 100, 100)
STATE_COLOR = (220, 220, 220)
POLICY_COLOR = (0, 200, 0)
OPT_POLICY_COLOR = (255, 215, 0)
BG_COLOR = (30, 30, 30)
ARROW_SIZE = 20
Q_COLOR = (0, 0, 0)
TRAJECTORY_COLOR = (128, 0, 128)

def draw_policy(screen, policy, n_states, color):
    for i in range(n_states):
        x = MARGIN + i * CASE_SIZE + CASE_SIZE // 2
        y = MARGIN + CASE_SIZE // 2
        action = policy.get(i, None)
        if action == 0:  # gauche
            pygame.draw.polygon(screen, color, [
                (x - ARROW_SIZE, y),
                (x, y - ARROW_SIZE // 2),
                (x, y + ARROW_SIZE // 2)
            ])
        elif action == 1:  # droite
            pygame.draw.polygon(screen, color, [
                (x + ARROW_SIZE, y),
                (x, y - ARROW_SIZE // 2),
                (x, y + ARROW_SIZE // 2)
            ])

def draw_q_values(screen, Q, env, font):
    for s in range(env.n_states):
        x = MARGIN + s * CASE_SIZE
        y = MARGIN
        for a in env.get_valid_actions(s):
            q = Q[s][a]
            txt = font.render(f"Q{a}={q:.2f}", True, Q_COLOR)
            offset = 5 if a == 0 else CASE_SIZE - 30
            screen.blit(txt, (x + 5, y + offset))

def draw_rewards(screen, env, font):
    for s in range(env.n_states):
        x = MARGIN + s * CASE_SIZE
        y = MARGIN
        r = env.get_reward(s)
        txt = font.render(f"R={r:.1f}", True, (100, 100, 100))
        screen.blit(txt, (x + CASE_SIZE - 45, y + CASE_SIZE - 25))

def draw_trajectory(screen, trajectory):
    if len(trajectory) < 2:
        return
    points = [
        (MARGIN + s * CASE_SIZE + CASE_SIZE // 2, MARGIN + CASE_SIZE // 2)
        for s in trajectory
    ]
    pygame.draw.lines(screen, TRAJECTORY_COLOR, False, points, 4)

def visualise_lineworld_pygame(env, policy, Q, rewards_per_episode, opt_policy=None, delay=0.5):
    pygame.init()
    n_states = env.n_states
    width = n_states * CASE_SIZE + 2 * MARGIN
    height = CASE_SIZE + 2 * MARGIN + 60
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("LineWorld - Dyna-Q Policy Visualization")
    font = pygame.font.SysFont(None, 28)
    bigfont = pygame.font.SysFont(None, 36)

    running = True
    state = env.reset()
    done = False
    trajectory = [state]

    while running:
        screen.fill(BG_COLOR)

        # Dessine les cases
        for i in range(n_states):
            rect = pygame.Rect(MARGIN + i * CASE_SIZE, MARGIN, CASE_SIZE, CASE_SIZE)
            color = TERMINAL_COLOR if i in env.terminal_states else STATE_COLOR
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)
            txt = bigfont.render(str(i), True, (0, 0, 0))
            screen.blit(txt, (rect.x + CASE_SIZE // 2 - 10, rect.y + CASE_SIZE // 2 - 18))

        # Trajectoire de l'agent
        draw_trajectory(screen, trajectory)

        # Politique optimale (si fournie)
        if opt_policy:
            draw_policy(screen, opt_policy, n_states, OPT_POLICY_COLOR)
        # Politique apprise
        draw_policy(screen, policy, n_states, POLICY_COLOR)

        # Valeurs Q
        draw_q_values(screen, Q, env, font)
        # Récompenses immédiates
        draw_rewards(screen, env, font)

        # Agent
        agent_x = MARGIN + state * CASE_SIZE + CASE_SIZE // 2
        agent_y = MARGIN + CASE_SIZE // 2
        pygame.draw.circle(screen, AGENT_COLOR, (agent_x, agent_y), CASE_SIZE // 4)

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
            action = policy.get(state, None)
            if action is None:
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
    env = LineWorld(size=7, start_state=3)
    Q, policy, rewards_per_episode = dyna_q(env, num_episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10)
    V_opt, opt_policy = value_iteration(env, gamma=0.95)

    # Affichage lisible de la politique apprise
    print("Politique apprise (état : action recommandée) :")
    for s in range(env.n_states):
        action = policy[s]
        if action is None:
            action_str = "Terminal"
        elif action == 0:
            action_str = "Gauche"
        elif action == 1:
            action_str = "Droite"
        else:
            action_str = str(action)
        print(f"  État {s} : {action_str}")

    # Affichage lisible de la politique optimale
    print("Politique optimale (état : action optimale) :")
    for s in range(env.n_states):
        action = opt_policy[s]
        if action is None:
            action_str = "Terminal"
        elif action == 0:
            action_str = "Gauche"
        elif action == 1:
            action_str = "Droite"
        else:
            action_str = str(action)
        print(f"  État {s} : {action_str}")

    visualise_lineworld_pygame(env, policy, Q, rewards_per_episode, opt_policy=opt_policy, delay=0.7)

    # Test manuel en console
    print("\n=== Test manuel de LineWorld ===")
    state = env.reset()
    done = False
    env.render()
    while not done:
        print(f"État actuel : {state}")
        actions = env.get_valid_actions(state)
        print(f"Actions valides : {actions} (0=gauche, 1=droite)")
        try:
            action = int(input("Quelle action ? (0 ou 1) : "))
        except Exception:
            print("Entrée invalide, réessaie.")
            continue
        if action not in actions:
            print("Action invalide, réessaie.")
            continue
        next_state, reward, done = env.step(action)
        print(f"--> Nouvel état : {next_state}, Récompense : {reward}, Terminé : {done}")
        env.render()
        state = next_state
    print("Épisode terminé.") 