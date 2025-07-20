#!/usr/bin/env python3
"""
Script de test rapide pour les algorithmes TD
Q-Learning, SARSA et Expected SARSA
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path
sys.path.append('.')

from envs.line_world.line_world import LineWorld
from rl_algorithms.q_learning import q_learning, get_policy_from_q, evaluate_policy
from rl_algorithms.sarsa import sarsa, get_policy_from_q as sarsa_get_policy
from rl_algorithms.expectedsarsa import expected_sarsa, get_policy_from_q as exp_sarsa_get_policy

def test_td_algorithms():
    """Test des trois algorithmes TD sur LineWorld"""
    
    print("=== Test des Algorithmes TD ===\n")
    
    # Création de l'environnement
    env = LineWorld(size=5, start_state=2)
    print(f"Environnement: LineWorld (5 états, position initiale: 2)")
    print(f"États terminaux: {env.terminal_states}")
    print(f"Actions: {env.action_space} (0=Gauche, 1=Droite)\n")
    
    # Paramètres d'entraînement
    episodes = 300
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    print(f"Paramètres d'entraînement:")
    print(f"  Épisodes: {episodes}")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Epsilon initial: {epsilon}")
    print(f"  Epsilon decay: {epsilon_decay}")
    print(f"  Epsilon minimum: {min_epsilon}\n")
    
    # Entraînement des algorithmes
    print("=== Entraînement Q-Learning ===")
    Q_ql, rewards_ql, lengths_ql = q_learning(
        env, episodes=episodes, alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon
    )
    
    print("\n=== Entraînement SARSA ===")
    Q_sarsa, rewards_sarsa, lengths_sarsa = sarsa(
        env, episodes=episodes, alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon
    )
    
    print("\n=== Entraînement Expected SARSA ===")
    Q_exp_sarsa, rewards_exp_sarsa, lengths_exp_sarsa = expected_sarsa(
        env, episodes=episodes, alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon
    )
    
    # Extraction des politiques
    policy_ql = get_policy_from_q(Q_ql, env)
    policy_sarsa = sarsa_get_policy(Q_sarsa, env)
    policy_exp_sarsa = exp_sarsa_get_policy(Q_exp_sarsa, env)
    
    # Évaluation des politiques
    print("\n=== Évaluation des politiques (100 épisodes) ===")
    avg_reward_ql, avg_length_ql = evaluate_policy(env, policy_ql, episodes=100)
    avg_reward_sarsa, avg_length_sarsa = evaluate_policy(env, policy_sarsa, episodes=100)
    avg_reward_exp_sarsa, avg_length_exp_sarsa = evaluate_policy(env, policy_exp_sarsa, episodes=100)
    
    print(f"Q-Learning:      Récompense = {avg_reward_ql:.3f}, Longueur = {avg_length_ql:.1f}")
    print(f"SARSA:           Récompense = {avg_reward_sarsa:.3f}, Longueur = {avg_length_sarsa:.1f}")
    print(f"Expected SARSA:  Récompense = {avg_reward_exp_sarsa:.3f}, Longueur = {avg_length_exp_sarsa:.1f}")
    
    # Affichage des politiques
    print("\n=== Politiques apprises ===")
    action_names = ['Gauche', 'Droite']
    print("État | Q-Learning | SARSA | Expected SARSA")
    print("-----|------------|-------|----------------")
    for i in range(env.n_states):
        if i in env.terminal_states:
            print(f"  {i}  |    Terminal   | Terminal |    Terminal   ")
        else:
            print(f"  {i}  |    {action_names[policy_ql[i]]:<8} | {action_names[policy_sarsa[i]]:<5} |    {action_names[policy_exp_sarsa[i]]:<8}")
    
    # Visualisation des résultats
    print("\n=== Génération des graphiques ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Courbes de récompenses
    window = 30
    ax1.plot(np.convolve(rewards_ql, np.ones(window)/window, mode='valid'), 
             label='Q-Learning', linewidth=2)
    ax1.plot(np.convolve(rewards_sarsa, np.ones(window)/window, mode='valid'), 
             label='SARSA', linewidth=2)
    ax1.plot(np.convolve(rewards_exp_sarsa, np.ones(window)/window, mode='valid'), 
             label='Expected SARSA', linewidth=2)
    ax1.set_title('Récompenses moyennes (fenêtre glissante de 30)')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense moyenne')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-tables
    im1 = ax2.imshow(Q_ql, cmap='viridis', aspect='auto')
    ax2.set_title('Q-Table - Q-Learning')
    ax2.set_xlabel('Action (0=Gauche, 1=Droite)')
    ax2.set_ylabel('État')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Gauche', 'Droite'])
    plt.colorbar(im1, ax=ax2)
    
    im2 = ax3.imshow(Q_sarsa, cmap='viridis', aspect='auto')
    ax3.set_title('Q-Table - SARSA')
    ax3.set_xlabel('Action (0=Gauche, 1=Droite)')
    ax3.set_ylabel('État')
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Gauche', 'Droite'])
    plt.colorbar(im2, ax=ax3)
    
    im3 = ax4.imshow(Q_exp_sarsa, cmap='viridis', aspect='auto')
    ax4.set_title('Q-Table - Expected SARSA')
    ax4.set_xlabel('Action (0=Gauche, 1=Droite)')
    ax4.set_ylabel('État')
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Gauche', 'Droite'])
    plt.colorbar(im3, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('td_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graphiques sauvegardés dans 'td_algorithms_comparison.png'")
    
    return {
        'Q_ql': Q_ql, 'Q_sarsa': Q_sarsa, 'Q_exp_sarsa': Q_exp_sarsa,
        'rewards_ql': rewards_ql, 'rewards_sarsa': rewards_sarsa, 'rewards_exp_sarsa': rewards_exp_sarsa,
        'policy_ql': policy_ql, 'policy_sarsa': policy_sarsa, 'policy_exp_sarsa': policy_exp_sarsa
    }

if __name__ == "__main__":
    results = test_td_algorithms()
    print("\n=== Test terminé avec succès ! ===") 