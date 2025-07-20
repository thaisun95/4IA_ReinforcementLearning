import numpy as np

def value_iteration_blackbox(env, gamma=0.99, theta=1e-6, max_iterations=1000, verbose=True):
    """
    Value Iteration for generic blackbox environments (SecretEnv).
    The env should provide:
      - num_states()
      - num_actions()
      - reset(), state_id(), step(a), available_actions(), is_game_over(), score()
      - The agent can only 'simulate' via actual env transitions (reset/step).
    Returns:
      V: np.ndarray of optimal state values.
      policy: np.ndarray of optimal actions for each state.
    """
    n_states = env.num_states()
    n_actions = env.num_actions()
    V = np.zeros(n_states)

    # Main Value Iteration loop
    for it in range(max_iterations):
        delta = 0
        for s in range(n_states):
            # For each state, skip if terminal (i.e. no available actions)
            env.reset()
            # Try to go to state s
            while env.state_id() != s and not env.is_game_over():
                # This assumes we can sample actions to try to reach s; otherwise, skip unreachable states
                acts = env.available_actions()
                if len(acts) == 0: break
                env.step(acts[0])
            if env.is_game_over():
                continue
            actions = env.available_actions()
            if len(actions) == 0:
                continue
            v = V[s]
            action_values = []
            for a in actions:
                # Simulate one step from state s with action a
                env.reset()
                while env.state_id() != s and not env.is_game_over():
                    acts = env.available_actions()
                    if len(acts) == 0: break
                    env.step(acts[0])
                if env.is_game_over():
                    action_values.append(0.0)
                    continue
                env.step(a)
                next_s = env.state_id()
                r = env.score() if env.is_game_over() else 0.0
                action_values.append(r + gamma * V[next_s])
            if len(action_values) > 0:
                V[s] = max(action_values)
                delta = max(delta, abs(v - V[s]))
        if verbose:
            print(f"Value iteration round {it+1} done")
        if delta < theta:
            print(f"Value iteration converged after {it+1} rounds.")
            break

    # Policy extraction
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        env.reset()
        while env.state_id() != s and not env.is_game_over():
            acts = env.available_actions()
            if len(acts) == 0: break
            env.step(acts[0])
        if env.is_game_over():
            policy[s] = -1
            continue
        actions = env.available_actions()
        if len(actions) == 0:
            policy[s] = -1
            continue
        action_values = []
        for a in actions:
            env.reset()
            while env.state_id() != s and not env.is_game_over():
                acts = env.available_actions()
                if len(acts) == 0: break
                env.step(acts[0])
            if env.is_game_over():
                action_values.append(0.0)
                continue
            env.step(a)
            next_s = env.state_id()
            r = env.score() if env.is_game_over() else 0.0
            action_values.append(r + gamma * V[next_s])
        if len(action_values) > 0:
            policy[s] = actions[np.argmax(action_values)]
        else:
            policy[s] = -1

    return V, policy