import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------- Environnement GridWorld ----------
class GridWorld:
    def __init__(self, n_rows=4, n_cols=4, start_state=(0,0), terminal_states=None, reward_matrix=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.state = start_state
        self.start_state = start_state
        if terminal_states is None:
            self.terminal_states = [(n_rows-1, n_cols-1)]
        else:
            self.terminal_states = terminal_states
        self.action_space = [0, 1, 2, 3]  # up, down, left, right
        if reward_matrix is None:
            self.reward_matrix = -np.ones((n_rows, n_cols))
            for t in self.terminal_states:
                self.reward_matrix[t] = 0
        else:
            self.reward_matrix = reward_matrix

    def state_to_index(self, state):
        row, col = state
        return row * self.n_cols + col

    def index_to_state(self, index):
        row = index // self.n_cols
        col = index % self.n_cols
        return (row, col)

    @property
    def n_states(self):
        return self.n_rows * self.n_cols

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_reward(self, state):
        return self.reward_matrix[state]

    def simulate_step(self, state, action):
        row, col = state
        if self.is_terminal(state):
            return state, self.get_reward(state), True
        if action == 0:     # Up
            next_row, next_col = max(row - 1, 0), col
        elif action == 1:   # Down
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 2:   # Left
            next_row, next_col = row, max(col - 1, 0)
        elif action == 3:   # Right
            next_row, next_col = row, min(col + 1, self.n_cols - 1)
        next_state = (next_row, next_col)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

# --------- Algorithme Policy Iteration ----------
def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    policy = np.random.choice(env.action_space, size=env.n_states)
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        if env.is_terminal(state):
            policy[idx] = -1
    V = np.zeros(env.n_states)
    for it in range(max_iterations):
        # Évaluation de la politique
        while True:
            delta = 0
            for idx in range(env.n_states):
                state = env.index_to_state(idx)
                if env.is_terminal(state):
                    continue
                v = V[idx]
                a = policy[idx]
                next_state, reward, _ = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                V[idx] = reward + gamma * V[next_idx]
                delta = max(delta, abs(v - V[idx]))
            if delta < theta:
                break
        # Amélioration de la politique
        policy_stable = True
        for idx in range(env.n_states):
            state = env.index_to_state(idx)
            if env.is_terminal(state):
                continue
            old_action = policy[idx]
            action_values = []
            for a in env.action_space:
                next_state, reward, _ = env.simulate_step(state, a)
                next_idx = env.state_to_index(next_state)
                action_values.append(reward + gamma * V[next_idx])
            best_action = np.argmax(action_values)
            policy[idx] = env.action_space[best_action]
            if old_action != policy[idx]:
                policy_stable = False
        if policy_stable:
            break
    return policy, V

# --------- Interface Streamlit ----------
st.set_page_config(page_title="Policy Iteration - GridWorld", layout="centered")
st.title("🧠 Policy Iteration - GridWorld")
st.markdown("Visualisez comment un agent apprend une politique optimale dans un environnement GridWorld 🎯.")

n_rows = st.slider("Nombre de lignes", 2, 10, 4)
n_cols = st.slider("Nombre de colonnes", 2, 10, 4)

env = GridWorld(n_rows=n_rows, n_cols=n_cols)

policy, V = policy_iteration(env)
env.reset()

# Initialiser l'état de l'agent
if "agent_state" not in st.session_state:
    st.session_state.agent_state = env.start_state
state = st.session_state.agent_state

# Bouton pour faire avancer l'agent
step_btn = st.button("▶️ Avancer d'une étape")

if step_btn and not env.is_terminal(state):
    idx = env.state_to_index(state)
    action = policy[idx]
    state, _, done = env.simulate_step(state, action)
    st.session_state.agent_state = state

# --------- Affichage Matplotlib ----------
fig, ax = plt.subplots(figsize=(6, 6))
grid_img = np.zeros((env.n_rows, env.n_cols))
for t in env.terminal_states:
    grid_img[t] = 1.0

ax.imshow(grid_img, cmap="Greys", vmin=0, vmax=1)

# Affichage de la politique sous forme de flèches
for idx in range(env.n_states):
    r, c = env.index_to_state(idx)
    a = policy[idx]
    if a == 0:
        ax.text(c, r, "↑", ha="center", va="center")
    elif a == 1:
        ax.text(c, r, "↓", ha="center", va="center")
    elif a == 2:
        ax.text(c, r, "←", ha="center", va="center")
    elif a == 3:
        ax.text(c, r, "→", ha="center", va="center")
    elif a == -1:
        ax.text(c, r, "🏁", ha="center", va="center", fontsize=12)

# Agent rouge
agent_r, agent_c = state
ax.plot(agent_c, agent_r, "ro", markersize=12)

ax.set_xticks(np.arange(env.n_cols))
ax.set_yticks(np.arange(env.n_rows))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)
st.pyplot(fig)
plt.close(fig)
