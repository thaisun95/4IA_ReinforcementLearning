import pygame
import sys
import numpy as np
from lineworld_env import LineWorld
from policy_iteration import policy_iteration

WIDTH, HEIGHT = 600, 100
FPS = 1

def draw_env(screen, env, policy=None, mode='manual'):
    screen.fill((255, 255, 255))
    block_width = WIDTH // env.size

    for i in range(env.size):
        color = (200, 200, 200)
        if i == env.state:
            color = (0, 100, 255)
        elif i in env.terminal_states:
            color = (0, 255, 0) if i == env.size - 1 else (255, 0, 0)
        pygame.draw.rect(screen, color, (i * block_width, 20, block_width - 5, 60))

    pygame.display.flip()

def main(mode='manual'):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LineWorld")
    clock = pygame.time.Clock()

    env = LineWorld(size=5)
    policy, V = policy_iteration(env)

    # Save
    np.save("policy.npy", policy)
    np.save("values.npy", V)

    state = env.reset()
    done = False

    while not done:
        draw_env(screen, env, policy, mode)
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if mode == "manual":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                _, _, done = env.step(0)
            elif keys[pygame.K_RIGHT]:
                _, _, done = env.step(1)

        elif mode == "agent":
            action = policy[state]
            if action == -1:
                break
            state, _, done = env.step(action)

    print("🎯 Finished!")
    pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual", "agent"], default="manual")
    args = parser.parse_args()

    main(mode=args.mode)
