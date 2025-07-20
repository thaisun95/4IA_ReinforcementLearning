import numpy as np
import random

def mc_off_policy_blackbox(env, num_episodes=10000, gamma=0.99, verbose=False):
    """
    Off-policy MC control using weighted importance sampling for blackbox envs.
    The environment must have:
      - reset(), step(a), available_actions(), is_game_over(), state_id(), num_states(), num_actions(), score()
    Returns:
      Q: state-action value table (n_states x n_actions)
      policy: optimal deterministic policy (n_states,)
      episode_rewards: list of final rewards for each episode
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    Q = np.zeros((n_states, n_actions))
    C = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    episode_rewards = []

    valid_episodes = 0
    for ep in range(num_episodes):
        env.reset()
        state = env.state_id()
        trajectory = []
        done = False
        steps = 0
        # Generate an episode using uniform random policy over available actions
        while not done:
            actions = env.available_actions()
            if len(actions) == 0:
                break
            a = random.choice(actions)
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
            W = 1
            for t in reversed(range(len(trajectory))):
                s, a, r = trajectory[t]
                G = gamma * G + r
                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
                # Greedy policy update over valid actions
                acts = env.available_actions() if not env.is_game_over() else []
                if not acts:
                    acts = list(range(n_actions))
                best_a = acts[np.argmax([Q[s, act] for act in acts])]
                policy[s] = best_a
                # If not the greedy action, stop (IS weight = 0)
                if a != policy[s]:
                    break
                W = W / (1.0 / len(acts)) if len(acts) > 0 else 1.0
    if verbose:
        print(f"Valid episodes (terminated): {valid_episodes}/{num_episodes}")
        print(f"Mean final reward: {np.mean(episode_rewards):.4f}")
        print(f"Std of final rewards: {np.std(episode_rewards):.4f}")
    return Q, policy, episode_rewards