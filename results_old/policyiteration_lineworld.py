import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------- LineWorld --------
class LineWorld:
    def __init__(self, size=5, start_state=2):
        self.size = size
        self.start_state = start_state
        self.state = start_state
        self.terminal_states = [0, size - 1]
        self.action_space = [0, 1]  # 0: left, 1: right

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_reward(self, next_state):
        if next_state == 0:
            return -1.0
        elif next_state == self.size - 1:
            return 1.0
        else:
            return 0.0

    def step(self, action):
        if self.is_terminal(self.state):
            return self.state, 0.0, True
        if action == 0:
            next_state = max(self.state - 1, 0)
        elif action == 1:
            next_state = min(self.state + 1, self.size - 1)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def reset(self):
        self.state = self.start_state
        return self.state


# -------- Policy Iteration --------
def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    policy = np.ones(env.size, dtype=int)
    for idx in range(env.size):
        if env.is_terminal(idx):
            policy[idx] = -1
    V = np.zeros(env.size)

    for _ in range(max_iterations):
        while True:
            delta = 0
            for idx in range(env.size):
                if env.is_terminal(idx):
                    continue
                v = V[idx]
                a = policy[idx]
                next_state = idx - 1 if a == 0 else idx + 1
                next_state = max(0, min(next_state, env.size - 1))
                reward = env.get_reward(next_state)
                V[idx] = reward + gamma * V[next_state]
                delta = max(delta, abs(v - V[idx]))
            if delta < theta:
                break

        policy_stable = True
        for idx in range(env.size):
            if env.is_terminal(idx):
                continue
            old_action = policy[idx]
            action_values = []
            for a in env.action_space:
                next_state = idx - 1 if a == 0 else idx + 1
                next_state = max(0, min(next_state, env.size - 1))
                reward = env.get_reward(next_state)
                action_values.append(reward + gamma * V[next_state])
            best_action = np.argmax(action_values)
            policy[idx] = env.action_space[best_action]
            if old_action != policy[idx]:
                policy_stable = False

        if policy_stable:
            break

    return policy, V


# -------- Visualisation --------
def draw_lineworld(env, policy, agent_pos=None):
    fig, ax = plt.subplots(figsize=(env.size * 1.5, 2))
    for i in range(env.size):
        if i in env.terminal_states:
            color = 'lightcoral' if i == 0 else 'lightgreen'
        else:
            color = 'lightblue'

        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        if i == agent_pos:
            ax.text(i + 0.5, 0.5, 'AGENT', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='red')
        elif i in env.terminal_states:
            reward_text = f"R={env.get_reward(i)}"
            ax.text(i + 0.5, 0.3, 'TERMINAL', ha='center', va='center',
                    fontsize=10, fontweight='bold')
            ax.text(i + 0.5, 0.7, reward_text, ha='center', va='center',
                    fontsize=8, color='darkred')
        else:
            symbol = '←' if policy[i] == 0 else '→' if policy[i] == 1 else 'T'
            ax.text(i + 0.5, 0.5, symbol, ha='center', va='center',
                    fontsize=20, fontweight='bold')

        ax.text(i + 0.5, 0.1, f'{i}', ha='center', va='center', fontsize=8, color='gray')

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('LineWorld - Policy Iteration', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# -------- Streamlit Interface --------
def main():
    st.set_page_config(page_title="Policy Iteration - LineWorld", layout="centered")
    st.title("🤖 Policy Iteration dans LineWorld")

    # Paramètres
    size = st.slider("Taille de l'environnement", 3, 15, 7)
    start = st.slider("Position de départ", 1, size - 2, size // 2)

    env = LineWorld(size=size, start_state=start)
    policy, V = policy_iteration(env)

    # Initialisation session state
    if 'state' not in st.session_state:
        st.session_state.state = env.start_state
        st.session_state.done = False
        st.session_state.step_count = 0

    st.subheader("📌 Politique optimale")
    fig = draw_lineworld(env, policy, agent_pos=st.session_state.state)
    st.pyplot(fig)
    plt.close()

    st.subheader("📈 Valeurs optimales par état")
    col_states = st.columns(min(size, 7))
    for i in range(size):
        with col_states[i % 7]:
            action_text = "←" if policy[i] == 0 else "→" if policy[i] == 1 else "Terminal"
            st.metric(f"État {i}", f"{V[i]:.3f}", delta=action_text)

    st.subheader("🎬 Contrôle manuel de l'agent")

    if st.session_state.done:
        st.success(f"Agent est arrivé à l'état terminal {st.session_state.state} avec récompense {env.get_reward(st.session_state.state)}")
        if st.button("🔄 Réinitialiser"):
            st.session_state.state = env.start_state
            st.session_state.done = False
            st.session_state.step_count = 0
            st.rerun()

    else:
        if st.button("▶️ Avancer d'une étape"):
            action = policy[st.session_state.state]
            next_state, reward, done = env.step(action)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.step_count += 1
            st.rerun()



if __name__ == "__main__":
    main()
