import pygame
import time

def visualize_rps_policy(env, policy, delay=0.8):
    """
    Visualisation interactive d'une policy RL pour TwoRoundRPS (Rock-Paper-Scissors).
    policy: dict(state) -> action (0=Rock, 1=Paper, 2=Scissors)
    """
    pygame.init()
    WIDTH, HEIGHT = 820, 540
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Two-Round RPS RL Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 34)
    font_big = pygame.font.SysFont(None, 48)

    move_map = {0: "Rock", 1: "Paper", 2: "Scissors", -1: "--"}
    ICONS = {
        0: pygame.image.load("images/rock_icon.png"),
        1: pygame.image.load("images/paper_icon.png"),
        2: pygame.image.load("images/scissors_icon.png"),
    }
    for v in ICONS.values():
        v = pygame.transform.scale(v, (60, 60))  # Resize for display

    COLORS = {
        'bg': (246, 248, 252),
        'border': (90, 90, 90),
        'button': (95, 156, 255),
        'button_fg': (255, 255, 255),
        'button_inactive': (200, 200, 200),
        'chosen': (220, 255, 220),
        'win': (90, 220, 140),
        'lose': (240, 120, 100),
        'draw': (180, 180, 180),
    }

    # Boutons
    btn_bar_top = HEIGHT - 80
    btns = [
        {"label": "Next", "rect": pygame.Rect(WIDTH//2-210, btn_bar_top, 120, 50)},
        {"label": "Reset", "rect": pygame.Rect(WIDTH//2-70, btn_bar_top, 120, 50)},
        {"label": "Show Solution", "rect": pygame.Rect(WIDTH//2+70, btn_bar_top, 170, 50)},
        {"label": "Quit", "rect": pygame.Rect(WIDTH//2+260, btn_bar_top, 90, 50)},
    ]

    def draw_buttons(active=True):
        for btn in btns:
            color = COLORS['button'] if active or btn['label'] == "Quit" else COLORS['button_inactive']
            pygame.draw.rect(screen, color, btn['rect'], border_radius=12)
            pygame.draw.rect(screen, COLORS['border'], btn['rect'], 2, border_radius=12)
            txt = font.render(btn['label'], True, COLORS['button_fg'])
            rect = txt.get_rect(center=btn['rect'].center)
            screen.blit(txt, rect)

    def draw_round_info(state, opp_move, reward, terminal):
        round_id, my_first_move = state
        y = 55
        txt = font_big.render(f"Round: {round_id+1 if round_id < 2 else 'Terminal'}", True, (20, 35, 95))
        screen.blit(txt, (WIDTH//2 - txt.get_width()//2, y))
        y += 60
        if round_id == 0:
            s = f"First move - Agent: --      Opponent: --"
        elif round_id == 1:
            s = f"First move: {move_map[my_first_move]}"
        else:
            s = f"First move: {move_map[my_first_move]}"
        txt = font.render(s, True, (45, 50, 50))
        screen.blit(txt, (WIDTH//2 - txt.get_width()//2, y))

        # Moves
        y += 65
        if round_id == 0:
            # Only one action
            agent_move = "--"
            opp = "--"
        elif round_id == 1:
            agent_move = "--"
            opp = "--"
        else:
            agent_move = move_map[my_first_move]
            opp = move_map.get(opp_move, "--")
        txt = font.render(f"Agent's move: {agent_move}     Opponent's move: {opp}", True, (80, 80, 80))
        screen.blit(txt, (WIDTH//2 - txt.get_width()//2, y))

        # Final outcome
        if terminal:
            y += 60
            if reward == 1:
                msg = "Agent wins!"
                c = COLORS['win']
            elif reward == -1:
                msg = "Agent loses."
                c = COLORS['lose']
            else:
                msg = "Draw."
                c = COLORS['draw']
            pygame.draw.rect(screen, c, (WIDTH//2-110, y, 220, 50), border_radius=8)
            t = font_big.render(msg, True, (255, 255, 255))
            screen.blit(t, (WIDTH//2 - t.get_width()//2, y+6))

    auto_solution = False
    running = True
    state = env.reset()
    opp_move = None
    reward = None
    done = False

    def advance_step():
        nonlocal state, opp_move, reward, done
        if env.is_terminal(state):
            done = True
            return
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            return
        action = policy[state] if state in policy and policy[state] in valid_actions else valid_actions[0]
        prev_state = state
        state, reward, finished = env.step(action)
        if prev_state[0] == 0:
            opp_move = env.last_opp_move
        if env.is_terminal(state):
            done = True

    while running:
        screen.fill(COLORS['bg'])
        terminal = env.is_terminal(state)
        draw_round_info(state, opp_move, reward, terminal)
        draw_buttons(active=not auto_solution)

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
                            opp_move = None
                            reward = None
                            done = False
                        elif btn['label'] == "Show Solution" and not done:
                            auto_solution = True
                        elif btn['label'] == "Quit":
                            running = False
        clock.tick(60)
    pygame.quit()