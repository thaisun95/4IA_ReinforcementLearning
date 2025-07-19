import numpy as np
import random
import pygame
import sys
import json
import os

# =============== ALGORITHMES ===============

def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    """
    Perform Policy Iteration for a generic environment with state-index mapping.
    """
    policy = np.random.choice(env.action_space, size=env.n_states)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if env.is_terminal(state):
            policy[idx] = -1  # No action for terminal states

    V = np.zeros(env.n_states)

    for it in range(max_iterations):
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for idx in range(env.n_states):
                state = env.index_to_state(idx)
                if env.is_terminal(state):
                    continue
                v = V[idx]
                a = policy[idx]
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                V[idx] = reward + gamma * V[next_idx]
                delta = max(delta, abs(v - V[idx]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
            old_action = policy[idx]
            action_values = []
            for a in env.action_space:
                next_state, reward, done = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                action_values.append(reward + gamma * V[next_idx])
            best_action = np.argmax(action_values)
            policy[idx] = env.action_space[best_action]
            if old_action != policy[idx]:
                policy_stable = False
        if policy_stable:
            print(f"Policy iteration converged after {it+1} iterations.")
            break

    return policy, V


class MontyHallV1:
    def __init__(self):
        self.action_space = [0, 1, 2]  # Simplifié: actions possibles selon le step
        # States: (step, chosen, remaining) where step=0 (choose), step=1 (stay/switch), step=2 (terminal)
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

    def get_valid_actions(self, state):
        """Retourne les actions valides pour un état donné"""
        step, chosen, remaining = state
        if step == 0:
            return [0, 1, 2]  # Choisir une porte
        elif step == 1:
            return [0, 1]  # Stay ou Switch
        else:
            return []  # Terminal state

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

    def simulate_step(self, state, action):
        step, chosen, remaining = state
        if step == 0:
            if action not in [0, 1, 2]:
                action = 0  # Action par défaut
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
            if action == 0:  # stay
                final_choice = chosen
            elif action == 1:  # switch
                final_choice = remaining
            else:
                final_choice = chosen  # Par défaut stay
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
            if action == 0:  # stay
                final_choice = chosen
            elif action == 1:  # switch
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


# =============== VISUALISATION PYGAME ===============

class MontyHallVisualization:
    def __init__(self, env, policy=None, V=None):
        pygame.init()
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Monty Hall - Policy Iteration")
        
        self.env = env
        self.policy = policy
        self.V = V
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Couleurs
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.GOLD = (255, 215, 0)
        
        # Mode de jeu
        self.mode = "manual"  # "manual", "auto", "step"
        self.auto_step_delay = 1000  # ms
        self.last_auto_step = 0
        self.step_index = 0
        
        # Statistiques
        self.stats = {"wins": 0, "games": 0}
        
    def draw_door(self, x, y, width, height, door_num, is_chosen=False, is_revealed=False, is_winning=False):
        """Dessine une porte"""
        color = self.LIGHT_GRAY
        if is_chosen:
            color = self.BLUE
        elif is_revealed:
            color = self.RED
        elif is_winning and self.env.state[0] == 2:  # Révéler à la fin
            color = self.GOLD
            
        pygame.draw.rect(self.screen, color, (x, y, width, height))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, width, height), 3)
        
        # Numéro de la porte
        text = self.font.render(f"Porte {door_num}", True, self.BLACK)
        text_rect = text.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text, text_rect)
        
        # Indications
        if is_chosen:
            label = self.small_font.render("Choisie", True, self.WHITE)
            self.screen.blit(label, (x, y - 25))
        elif is_revealed:
            label = self.small_font.render("Révélée", True, self.WHITE)
            self.screen.blit(label, (x, y - 25))
        elif is_winning and self.env.state[0] == 2:
            label = self.small_font.render("Gagnante!", True, self.BLACK)
            self.screen.blit(label, (x, y - 25))

    def draw_buttons(self):
        """Dessine les boutons de contrôle"""
        buttons = []
        y_pos = self.height - 150
        
        # Boutons de mode
        modes = [("Manuel", "manual"), ("Auto", "auto"), ("Pas à pas", "step")]
        for i, (text, mode) in enumerate(modes):
            x = 50 + i * 120
            color = self.GREEN if self.mode == mode else self.LIGHT_GRAY
            pygame.draw.rect(self.screen, color, (x, y_pos, 100, 40))
            pygame.draw.rect(self.screen, self.BLACK, (x, y_pos, 100, 40), 2)
            
            text_surf = self.small_font.render(text, True, self.BLACK)
            text_rect = text_surf.get_rect(center=(x + 50, y_pos + 20))
            self.screen.blit(text_surf, text_rect)
            buttons.append((x, y_pos, 100, 40, mode))
        
        # Bouton Reset
        x = 400
        pygame.draw.rect(self.screen, self.RED, (x, y_pos, 100, 40))
        pygame.draw.rect(self.screen, self.BLACK, (x, y_pos, 100, 40), 2)
        text_surf = self.small_font.render("Reset", True, self.WHITE)
        text_rect = text_surf.get_rect(center=(x + 50, y_pos + 20))
        self.screen.blit(text_surf, text_rect)
        buttons.append((x, y_pos, 100, 40, "reset"))
        
        return buttons

    def draw_info(self):
        """Affiche les informations sur l'état actuel"""
        step, chosen, remaining = self.env.state
        
        # État actuel
        if step == 0:
            info_text = "Choisissez une porte (cliquez ou touches 0,1,2)"
        elif step == 1:
            info_text = f"Porte choisie: {chosen}, Monty révèle: {self.env._revealed}"
            info_text += f"\nRester (R) ou Changer (C) pour porte {remaining}?"
        else:
            result = "Gagné!" if chosen == self.env._winning_door else "Perdu!"
            info_text = f"Résultat: {result} - Porte gagnante: {self.env._winning_door}"
        
        lines = info_text.split('\n')
        for i, line in enumerate(lines):
            text = self.font.render(line, True, self.BLACK)
            self.screen.blit(text, (50, 50 + i * 30))
        
        # Statistiques
        if self.stats["games"] > 0:
            win_rate = self.stats["wins"] / self.stats["games"] * 100
            stats_text = f"Parties: {self.stats['games']}, Victoires: {self.stats['wins']}, Taux: {win_rate:.1f}%"
            text = self.small_font.render(stats_text, True, self.BLACK)
            self.screen.blit(text, (50, 150))
        
        # Policy actuelle si disponible
        if self.policy is not None and step < 2:
            state_idx = self.env.state_to_index(self.env.state)
            if state_idx < len(self.policy) and self.policy[state_idx] != -1:
                if step == 0:
                    rec_text = f"Policy recommande: Porte {self.policy[state_idx]}"
                else:
                    rec_text = f"Policy recommande: {'Rester' if self.policy[state_idx] == 0 else 'Changer'}"
                text = self.small_font.render(rec_text, True, self.BLUE)
                self.screen.blit(text, (50, 180))
        
        # Value function
        if self.V is not None:
            state_idx = self.env.state_to_index(self.env.state)
            if state_idx < len(self.V):
                value_text = f"Valeur de l'état: {self.V[state_idx]:.3f}"
                text = self.small_font.render(value_text, True, self.GREEN)
                self.screen.blit(text, (50, 200))

    def draw(self):
        """Dessine l'interface complète"""
        self.screen.fill(self.WHITE)
        
        # Dessiner les portes
        door_width = 120
        door_height = 200
        start_x = (self.width - 3 * door_width - 2 * 50) // 2
        door_y = 250
        
        step, chosen, remaining = self.env.state
        
        for i in range(3):
            x = start_x + i * (door_width + 50)
            is_chosen = (chosen == i)
            is_revealed = (self.env._revealed == i)
            is_winning = (self.env._winning_door == i)
            
            self.draw_door(x, door_y, door_width, door_height, i, is_chosen, is_revealed, is_winning)
        
        # Informations et contrôles
        self.draw_info()
        buttons = self.draw_buttons()
        
        pygame.display.flip()
        return buttons

    def handle_manual_input(self, event):
        """Gère les entrées manuelles"""
        step, chosen, remaining = self.env.state
        
        if event.type == pygame.KEYDOWN:
            if step == 0:  # Choix de porte
                if event.key == pygame.K_0:
                    self.make_move(0)
                elif event.key == pygame.K_1:
                    self.make_move(1)
                elif event.key == pygame.K_2:
                    self.make_move(2)
            elif step == 1:  # Rester ou changer
                if event.key == pygame.K_r:  # Rester
                    self.make_move(0)
                elif event.key == pygame.K_c:  # Changer
                    self.make_move(1)
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if step == 0:  # Clic sur les portes
                mouse_x, mouse_y = event.pos
                door_width = 120
                start_x = (self.width - 3 * door_width - 2 * 50) // 2
                door_y = 250
                
                for i in range(3):
                    x = start_x + i * (door_width + 50)
                    if x <= mouse_x <= x + door_width and door_y <= mouse_y <= door_y + 200:
                        self.make_move(i)
                        break

    def make_move(self, action):
        """Exécute une action"""
        if not self.env.is_terminal(self.env.state):
            next_state, reward, done = self.env.step(action)
            if done:
                self.stats["games"] += 1
                if reward > 0:
                    self.stats["wins"] += 1

    def auto_play(self):
        """Jeu automatique selon la policy"""
        if self.policy is None or self.env.is_terminal(self.env.state):
            return
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_auto_step > self.auto_step_delay:
            state_idx = self.env.state_to_index(self.env.state)
            if state_idx < len(self.policy) and self.policy[state_idx] != -1:
                action = self.policy[state_idx]
                self.make_move(action)
            self.last_auto_step = current_time

    def step_play(self):
        """Jeu pas à pas"""
        if self.policy is None or self.env.is_terminal(self.env.state):
            return
        
        state_idx = self.env.state_to_index(self.env.state)
        if state_idx < len(self.policy) and self.policy[state_idx] != -1:
            action = self.policy[state_idx]
            self.make_move(action)

    def run(self):
        """Boucle principale"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    buttons = self.draw_buttons()
                    mouse_pos = event.pos
                    
                    # Vérifier les clics sur les boutons
                    for x, y, w, h, action in buttons:
                        if x <= mouse_pos[0] <= x + w and y <= mouse_pos[1] <= y + h:
                            if action in ["manual", "auto", "step"]:
                                self.mode = action
                            elif action == "reset":
                                self.env.reset()
                            break
                    else:
                        if self.mode == "manual":
                            self.handle_manual_input(event)
                
                elif event.type == pygame.KEYDOWN:
                    if self.mode == "manual":
                        self.handle_manual_input(event)
                    elif self.mode == "step" and event.key == pygame.K_SPACE:
                        self.step_play()
            
            # Jeu automatique
            if self.mode == "auto":
                self.auto_play()
            
            self.draw()
            clock.tick(60)
        
        pygame.quit()


# =============== SAUVEGARDE/CHARGEMENT ===============

def save_policy_and_values(policy, V, filename="monty_hall_policy.json"):
    """Sauvegarde la policy et les valeurs"""
    data = {
        "policy": policy.tolist(),
        "V": V.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Policy et valeurs sauvegardées dans {filename}")

def load_policy_and_values(filename="monty_hall_policy.json"):
    """Charge la policy et les valeurs"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        policy = np.array(data["policy"])
        V = np.array(data["V"])
        print(f"Policy et valeurs chargées depuis {filename}")
        return policy, V
    return None, None


# =============== EVALUATION DES PERFORMANCES ===============

def evaluate_policy(env, policy, n_episodes=1000):
    """Évalue les performances d'une policy"""
    wins = 0
    
    for _ in range(n_episodes):
        env.reset()
        total_reward = 0
        
        while not env.is_terminal(env.state):
            state_idx = env.state_to_index(env.state)
            if state_idx < len(policy) and policy[state_idx] != -1:
                action = policy[state_idx]
            else:
                # Action aléatoire si pas de policy
                valid_actions = env.get_valid_actions(env.state)
                action = random.choice(valid_actions) if valid_actions else 0
            
            _, reward, _ = env.step(action)
            total_reward += reward
        
        if total_reward > 0:
            wins += 1
    
    win_rate = wins / n_episodes
    print(f"Performance sur {n_episodes} épisodes:")
    print(f"Taux de victoire: {win_rate:.3f} ({wins}/{n_episodes})")
    print(f"Taux de victoire théorique optimal (toujours changer): 0.667")
    
    return win_rate


# =============== FONCTION PRINCIPALE ===============

def main():
    print("=== Monty Hall - Policy Iteration ===\n")
    
    # Créer l'environnement
    env = MontyHallV1()
    
    # Charger ou entraîner la policy
    policy, V = load_policy_and_values()
    
    if policy is None:
        print("Entraînement de la policy avec Policy Iteration...")
        policy, V = policy_iteration(env, gamma=0.99, theta=1e-6)
        save_policy_and_values(policy, V)
        print("Entraînement terminé!\n")
    else:
        print("Policy chargée depuis le fichier.\n")
    
    # Afficher la policy apprise
    print("Policy optimale:")
    for idx, state in enumerate(env.states):
        step, chosen, remaining = state
        if not env.is_terminal(state):
            action = policy[idx]
            value = V[idx]
            if step == 0:
                print(f"État {state}: Choisir porte {action} (V={value:.3f})")
            else:
                action_name = "Rester" if action == 0 else "Changer"
                print(f"État {state}: {action_name} (V={value:.3f})")
    print()
    
    # Évaluer les performances
    evaluate_policy(env, policy)
    print()
    
    # Lancer la visualisation
    print("Lancement de la visualisation Pygame...")
    print("Contrôles:")
    print("- Mode Manuel: Cliquez sur les portes ou utilisez touches 0,1,2 puis R(rester)/C(changer)")
    print("- Mode Auto: La policy joue automatiquement")
    print("- Mode Pas à pas: Appuyez sur ESPACE pour avancer")
    print("- Bouton Reset: Nouvelle partie")
    
    env.reset()
    viz = MontyHallVisualization(env, policy, V)
    viz.run()


if __name__ == "__main__":
    main()