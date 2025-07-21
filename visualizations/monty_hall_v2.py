import pygame
import time
import numpy as np

def visualize_montyhall_v2_policy(env, policy, delay=0.8):
    """
    Visualisation interactive d'une policy RL pour MontyHallV2 (n portes).
    policy: dict(state) -> action (toujours un int)
    """
    pygame.init()
    WIDTH, HEIGHT = 1200, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Monty Hall V2 RL Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    font_big = pygame.font.SysFont(None, 46)

    n_doors = env.n_doors
    door_width = 100
    door_height = 260
    margin = 30
    total_w = n_doors * door_width + (n_doors + 1) * margin
    door_y = 140
    door_xs = [WIDTH//2 - total_w//2 + margin + i*(door_width+margin) for i in range(n_doors)]

    COLORS = {
        'bg': (245, 245, 245),
        'door': (180, 180, 210),
        'door_open': (245, 255, 200),
        'selected': (255, 230, 120),
        'revealed': (230, 90, 80),
        'winning': (60, 210, 80),
        'border': (60, 60, 60),
        'text': (40, 40, 40),
        'button_bg': (70, 130, 180),
        'button_fg': (245, 245, 245),
        'button_inactive': (170, 170, 170),
    }

    # Boutons
    btn_bar_top = HEIGHT - 90
    btns = [
        {"label": "Next", "rect": pygame.Rect(WIDTH//2-320, btn_bar_top, 140, 54)},
        {"label": "Reset", "rect": pygame.Rect(WIDTH//2-160, btn_bar_top, 140, 54)},
        {"label": "Show Solution", "rect": pygame.Rect(WIDTH//2+10, btn_bar_top, 210, 54)},
        {"label": "Quit", "rect": pygame.Rect(WIDTH//2+250, btn_bar_top, 110, 54)},
    ]

    def draw_buttons(active=True):
        for btn in btns:
            color = COLORS['button_bg'] if active or btn['label'] == "Quit" else COLORS['button_inactive']
            pygame.draw.rect(screen, color, btn['rect'], border_radius=12)
            pygame.draw.rect(screen, COLORS['border'], btn['rect'], 2, border_radius=12)
            txt = font.render(btn['label'], True, COLORS['button_fg'])
            rect = txt.get_rect(center=btn['rect'].center)
            screen.blit(txt, rect)

    def draw_doors(state, revealed, chosen, winning, terminal=False):
        _, doors_remaining, last_chosen = state
        for i in range(n_doors):
            x = door_xs[i]
            color = COLORS['door']
            label = f"{i}"
            if i in revealed:
                color = COLORS['revealed']
            if i == last_chosen and last_chosen != -1 and not terminal:
                color = COLORS['selected']
            if terminal and i == chosen:
                color = COLORS['winning'] if chosen == winning else COLORS['revealed']
            pygame.draw.rect(screen, color, (x, door_y, door_width, door_height), border_radius=15)
            pygame.draw.rect(screen, COLORS['border'], (x, door_y, door_width, door_height), 3, border_radius=15)
            txt = font_big.render(label, True, COLORS['border'])
            rect = txt.get_rect(center=(x + door_width//2, door_y + door_height//2 + 6))
            screen.blit(txt, rect)
            # Terminal: affichage WIN/LOSE
            if terminal and i == chosen:
                r = pygame.Rect(x, door_y + door_height + 10, door_width, 34)
                res_txt = "WIN!" if chosen == winning else "LOSE"
                res_color = COLORS['winning'] if chosen == winning else COLORS['revealed']
                pygame.draw.rect(screen, res_color, r, border_radius=8)
                pygame.draw.rect(screen, COLORS['border'], r, 1, border_radius=8)
                t = font.render(res_txt, True, COLORS['button_fg'])
                screen.blit(t, t.get_rect(center=r.center))

    def info_text(state, revealed, winning, reward):
        step, doors_remaining, last_chosen = state
        lines = []
        if step == 0:
            lines.append(f"Step 1: Pick your first door (0 to {n_doors-1})")
        elif not env.is_terminal(state):
            lines.append(f"Step {step+1}/{n_doors}: Remaining doors: {list(doors_remaining)}")
            lines.append(f"Doors revealed so far: {revealed}")
            if last_chosen != -1:
                lines.append(f"Your last chosen door: {last_chosen}")
        else:
            lines.append(f"Final choice: door {last_chosen}")
            lines.append(f"The winning door was: {winning}")
            lines.append("You win!" if reward > 0 else "You lose.")
        return lines

    auto_solution = False
    running = True
    state = env.reset()
    revealed = []
    chosen = -1
    reward = None
    done = False
    winning = env._winning_door

    # --- PATCH POLICY ACCESS ---
    def safe_policy_action(s):
        """
        Retourne toujours un int action valide pour un état s.
        """
        valid = env.get_valid_actions(s)
        act = None
        if isinstance(policy, dict):
            act = policy.get(s, None)
        elif isinstance(policy, np.ndarray):
            # On cherche l’index correspondant à s
            state_idx = env.state_to_index(s)
            act = policy[state_idx]
        else:
            raise RuntimeError("Policy type not supported")

        # Parfois la policy renvoie une séquence
        if isinstance(act, (list, tuple)):
            act = act[0] if act else None
        if (act not in valid) and valid:
            act = valid[0]
        return act


    def advance_step():
        nonlocal state, revealed, chosen, reward, done, winning
        if env.is_terminal(state):
            done = True
            return
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            return
        action = safe_policy_action(state)
        prev_step = state[0]
        prev_chosen = state[2]
        state, reward, finished = env.step(action)
        if prev_step == 0:
            chosen = action
            revealed = []  # Start new sequence
        else:
            chosen = action
            if hasattr(env, "_revealed") and env._revealed:
                revealed = list(env._revealed)
        if env.is_terminal(state):
            done = True

    while running:
        screen.fill(COLORS['bg'])
        terminal = env.is_terminal(state)
        draw_doors(state, revealed, chosen, env._winning_door, terminal)
        draw_buttons(active=not auto_solution)
        # Info
        lines = info_text(state, revealed, env._winning_door, reward)
        for i, txt in enumerate(lines):
            t = font.render(txt, True, COLORS['text'])
            screen.blit(t, (55, 32 + i*38))

        pygame.display.flip()

        if auto_solution and not done:
            time.sleep(delay)
            advance_step()
            continue
        elif auto_solution and done:
            auto_solution = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not auto_solution:
                for btn in btns:
                    if btn['rect'].collidepoint(event.pos):
                        if btn['label'] == "Next" and not done:
                            advance_step()
                        elif btn['label'] == "Reset":
                            state = env.reset()
                            revealed = []
                            chosen = -1
                            reward = None
                            done = False
                            winning = env._winning_door
                        elif btn['label'] == "Show Solution" and not done:
                            auto_solution = True
                        elif btn['label'] == "Quit":
                            running = False
        clock.tick(60)
    pygame.quit()