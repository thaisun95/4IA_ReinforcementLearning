import pygame
import time

def visualize_montyhall_policy(env, policy, delay=0.7):
    """
    Visualize optimal (or learned) policy for MontyHallV1 using Pygame.
    """
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Monty Hall RL Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 34)
    font_big = pygame.font.SysFont(None, 48)
    door_width = 110
    door_height = 220
    door_y = 150
    door_xs = [170, 390, 610]

    # Colors
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

    # Buttons (Next, Reset, Show Solution, Quit)
    btn_bar_top = HEIGHT - 90
    btns = [
        {"label": "Next", "rect": pygame.Rect(WIDTH//2-280, btn_bar_top, 130, 54)},
        {"label": "Reset", "rect": pygame.Rect(WIDTH//2-130, btn_bar_top, 130, 54)},
        {"label": "Show Solution", "rect": pygame.Rect(WIDTH//2+30, btn_bar_top, 200, 54)},
        {"label": "Quit", "rect": pygame.Rect(WIDTH//2+250, btn_bar_top, 100, 54)},
    ]

    def draw_buttons(active=True):
        for btn in btns:
            color = COLORS['button_bg'] if active or btn['label'] == "Quit" else COLORS['button_inactive']
            pygame.draw.rect(screen, color, btn['rect'], border_radius=11)
            pygame.draw.rect(screen, COLORS['border'], btn['rect'], 2, border_radius=11)
            txt = font.render(btn['label'], True, COLORS['button_fg'])
            rect = txt.get_rect(center=btn['rect'].center)
            screen.blit(txt, rect)

    def draw_doors(state, chosen=None, revealed=None, winning=None):
        step, sel, rem = state
        for i in range(3):
            x = door_xs[i]
            color = COLORS['door']
            label = f"{i}"
            if i == winning and step == 2:
                color = COLORS['winning']
            if i == chosen and step > 0:
                color = COLORS['selected']
            if i == revealed and step == 1:
                color = COLORS['revealed']
            pygame.draw.rect(screen, color, (x, door_y, door_width, door_height), border_radius=15)
            pygame.draw.rect(screen, COLORS['border'], (x, door_y, door_width, door_height), 3, border_radius=15)
            txt = font_big.render(label, True, COLORS['border'])
            rect = txt.get_rect(center=(x + door_width//2, door_y + door_height//2 + 5))
            screen.blit(txt, rect)

            # If terminal state, show WIN text
            if step == 2 and i == sel:
                r = pygame.Rect(x, door_y + door_height + 10, door_width, 34)
                res_txt = "WIN!" if sel == winning else "LOSE"
                res_color = COLORS['winning'] if sel == winning else COLORS['revealed']
                pygame.draw.rect(screen, res_color, r, border_radius=8)
                pygame.draw.rect(screen, COLORS['border'], r, 1, border_radius=8)
                t = font.render(res_txt, True, COLORS['button_fg'])
                screen.blit(t, t.get_rect(center=r.center))

    def info_text(state, chosen, revealed, winning, step, reward):
        lines = []
        if step == 0:
            lines.append("Step 1: Choose a door (0, 1, 2)")
            if chosen != -1:
                lines.append(f"You selected door {chosen}")
        elif step == 1:
            lines.append(f"Step 2: Monty opens door {revealed} (never the winning door, never your choice)")
            lines.append(f"Stay (0) with door {chosen} or Switch (1) to door {state[2]}?")
        elif step == 2:
            lines.append(f"Final choice: door {chosen}")
            lines.append(f"The winning door was: {winning}")
            lines.append("You win!" if reward > 0 else "You lose.")
        return lines

    # Main loop
    auto_solution = False
    running = True
    state = env.reset()
    chosen = -1
    revealed = None
    reward = None
    step = 0
    winning = env._winning_door
    done = False

    def advance_step():
        nonlocal state, chosen, revealed, reward, step, done, winning
        idx = env.state_to_index(state)
        if env.is_terminal(state):
            done = True
            return
        valid_actions = env.get_valid_actions(state)
        # Choose policy action
        if not valid_actions:
            return
        action = policy[idx] if policy[idx] in valid_actions else valid_actions[0]
        prev_step = state[0]
        state, reward, finished = env.step(action)
        step = state[0]
        if prev_step == 0:
            chosen = action
            revealed = env._revealed
        elif prev_step == 1:
            # action: 0=stay, 1=switch
            pass
        if env.is_terminal(state):
            done = True

    while running:
        screen.fill(COLORS['bg'])
        draw_doors(state, chosen, revealed, env._winning_door)
        draw_buttons(active=not auto_solution)
        # Info
        lines = info_text(state, chosen, revealed, env._winning_door, state[0], reward)
        for i, txt in enumerate(lines):
            t = font.render(txt, True, COLORS['text'])
            screen.blit(t, (60, 30 + i*34))

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
                            chosen = -1
                            revealed = None
                            reward = None
                            done = False
                            winning = env._winning_door
                            step = 0
                        elif btn['label'] == "Show Solution" and not done:
                            auto_solution = True
                        elif btn['label'] == "Quit":
                            running = False
        clock.tick(60)
    pygame.quit()