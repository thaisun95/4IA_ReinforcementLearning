import numpy as np
import random

def mc_on_policy_blackbox(env, num_episodes=10000, gamma=0.99, epsilon=0.1, verbose=False):
    """
    On-policy First-Visit Monte Carlo control with epsilon-soft policies for blackbox envs.
    The environment must have:
      - reset(), step(a), available_actions(), is_game_over(), state_id(), num_states(), num_actions(), score()
    Returns:
      Q: state-action value table (n_states x n_actions)
      policy: stochastic policy (n_states x n_actions)
      episode_rewards: list of final rewards for each episode
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    policy = np.ones((n_states, n_actions)) / n_actions
    episode_rewards = []

    valid_episodes = 0
    for ep in range(num_episodes):
        env.reset()
        state = env.state_id()
        trajectory = []
        done = False
        steps = 0
        while not done:
            actions = env.available_actions()
            if len(actions) == 0:
                break
            # Epsilon-greedy policy over valid actions
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                q_valid = [Q[state, act] for act in actions]
                a = actions[np.argmax(q_valid)]
            s = state
            env.step(a)
            state = env.state_id()
            reward = env.score() if env.is_game_over() else 0.0
            done = env.is_game_over()
            trajectory.append((s, a, reward))
            steps += 1

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
                    # Update epsilon-soft policy only among valid actions
                    acts = env.available_actions() if not env.is_game_over() else []
                    if not acts:
                        acts = list(range(n_actions))
                    best_a = acts[np.argmax([Q[s, act] for act in acts])]
                    for act in range(n_actions):
                        if act in acts:
                            if act == best_a:
                                policy[s, act] = 1 - epsilon + epsilon / len(acts)
                            else:
                                policy[s, act] = epsilon / len(acts)
                        else:
                            policy[s, act] = 0
                    visited.add((s, a))
    if verbose:
        print(f"Valid episodes (terminated): {valid_episodes}/{num_episodes}")
        print(f"Mean final reward: {np.mean(episode_rewards):.4f}")
        print(f"Std of final rewards: {np.std(episode_rewards):.4f}")
    return Q, policy, episode_rewards