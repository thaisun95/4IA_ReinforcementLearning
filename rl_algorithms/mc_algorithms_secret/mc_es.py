import numpy as np
import random

def mc_es_blackbox(env, num_episodes=10000, gamma=0.99, verbose=False):
    """
    Monte Carlo Exploring Starts (ES) for blackbox environments.
    The environment must have:
      - reset()
      - step(a)
      - available_actions()
      - is_game_over()
      - state_id()
      - num_states(), num_actions()
      - score() (returns the terminal reward)
    Returns:
      Q: state-action value table (n_states x n_actions)
      policy: optimal deterministic policy (n_states,)
      episode_rewards: list of final rewards for each episode
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    episode_rewards = []

    valid_episodes = 0
    for ep in range(num_episodes):
        env.reset()
        # Exploring starts: start with a random (state, action)
        state = env.state_id()
        actions = env.available_actions()
        action = random.choice(actions)
        trajectory = []
        done = False
        first = True
        while not done:
            if first:
                # On the first step, use the chosen exploring start action
                s, a = state, action
                first = False
            else:
                actions = env.available_actions()
                if len(actions) == 0:
                    break
                a = policy[state]
                # If the greedy action is not valid, pick a random valid one
                if a not in actions:
                    a = random.choice(actions)
                s = state
            env.step(a)
            next_state = env.state_id()
            reward = env.score() if env.is_game_over() else 0.0
            done = env.is_game_over()
            trajectory.append((s, a, reward))
            state = next_state

        # Only update if episode terminated correctly
        if done:
            valid_episodes += 1
            episode_rewards.append(reward)
            G = 0
            visited = set()
            for t in reversed(range(len(trajectory))):
                s, a, r = trajectory[t]
                G = gamma * G + r
                if (s, a) not in visited:
                    N[s, a] += 1
                    Q[s, a] += (G - Q[s, a]) / N[s, a]
                    # Update policy: greedy among valid actions
                    acts = env.available_actions() if not env.is_game_over() else []
                    if not acts:
                        acts = list(range(n_actions))
                    best_a = acts[np.argmax([Q[s, act] for act in acts])]
                    policy[s] = best_a
                    visited.add((s, a))
    if verbose:
        print(f"Valid episodes (terminated): {valid_episodes}/{num_episodes}")
        print(f"Mean final reward: {np.mean(episode_rewards):.4f}")
        print(f"Std of final rewards: {np.std(episode_rewards):.4f}")
    return Q, policy, episode_rewards