import numpy as np

class GridWorld:
    def __init__(self, n_rows=4, n_cols=4, start_state=(0,0), terminal_states=None, reward_matrix=None):
        """
<<<<<<< HEAD
        Initialize the GridWorld environment.

        Args:
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            start_state (tuple): Starting position (row, col).
            terminal_states (list of tuples): Terminal states positions.
            reward_matrix (np.ndarray): Matrix of rewards (optional).
=======
        Grille NxM : l'agent se déplace sur la grille avec actions (haut, bas, gauche, droite).
        Par défaut, la case en bas à droite est terminale.
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        """
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
<<<<<<< HEAD
=======
        """Transforme (row, col) en index unique."""
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        row, col = state
        return row * self.n_cols + col

    def index_to_state(self, index):
<<<<<<< HEAD
=======
        """Transforme un index unique en (row, col)."""
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        row = index // self.n_cols
        col = index % self.n_cols
        return (row, col)

    @property
    def n_states(self):
<<<<<<< HEAD
        return self.n_rows * self.n_cols

    def reset(self):
=======
        """Nombre d'états total."""
        return self.n_rows * self.n_cols

    def reset(self):
        """Reset l'agent à l'état initial."""
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
<<<<<<< HEAD
        return state in self.terminal_states

    def get_reward(self, state):
        return self.reward_matrix[state]

    def step(self, action):
=======
        """Vrai si état terminal."""
        return state in self.terminal_states

    def get_reward(self, state):
        """Récompense associée à l'état."""
        return self.reward_matrix[state]

    def step(self, action):
        """
        Applique une action, modifie l'état de l'agent, retourne (next_state, reward, done).
        0: haut, 1: bas, 2: gauche, 3: droite
        """
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        if self.is_terminal(self.state):
            return self.state, self.get_reward(self.state), True

        row, col = self.state
        if action == 0:   # up
<<<<<<< HEAD
            next_row = max(row - 1, 0)
            next_col = col
        elif action == 1: # down
            next_row = min(row + 1, self.n_rows - 1)
            next_col = col
        elif action == 2: # left
            next_row = row
            next_col = max(col - 1, 0)
        elif action == 3: # right
            next_row = row
            next_col = min(col + 1, self.n_cols - 1)
=======
            next_row, next_col = max(row - 1, 0), col
        elif action == 1: # down
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 2: # left
            next_row, next_col = row, max(col - 1, 0)
        elif action == 3: # right
            next_row, next_col = row, min(col + 1, self.n_cols - 1)
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        else:
            raise ValueError("Invalid action (0=up, 1=down, 2=left, 3=right)")

        next_state = (next_row, next_col)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
<<<<<<< HEAD
        """
        Simulate a step from a given state and action, without modifying the internal state.
        Returns next_state, reward, done.
        """
=======
        """Simule une action à partir d'un état, sans modifier self.state."""
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        row, col = state
        if self.is_terminal(state):
            return state, self.get_reward(state), True
        if action == 0:
            next_row, next_col = max(row - 1, 0), col
        elif action == 1:
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 2:
            next_row, next_col = row, max(col - 1, 0)
        elif action == 3:
            next_row, next_col = row, min(col + 1, self.n_cols - 1)
        else:
            raise ValueError("Invalid action (0=up, 1=down, 2=left, 3=right)")
        next_state = (next_row, next_col)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def render(self):
        """
<<<<<<< HEAD
        Pretty print the grid with agent and terminal(s).
        'A': Agent
        'T': Terminal state
        '.': Empty cell
=======
        Affiche la grille avec l'agent (A), les terminaux (T), et les cases vides (.).
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        """
        grid = np.array([["."] * self.n_cols for _ in range(self.n_rows)])
        row, col = self.state
        grid[row, col] = "A"
        for t in self.terminal_states:
            if grid[t] != "A":
                grid[t] = "T"
        print("\n".join(" ".join(row) for row in grid))
<<<<<<< HEAD
        print()  # blank line for spacing

    def render_policy(self, policy):
        """
        Visualize a policy as arrows on the grid.
        0: up (↑), 1: down (↓), 2: left (←), 3: right (→), -1: terminal (X)
=======
        print()

    def render_policy(self, policy):
        """
        Affiche une politique sous forme de flèches sur la grille.
        0: ↑, 1: ↓, 2: ←, 3: →, -1: X (terminal)
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
        """
        arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→", -1: "X"}
        grid = np.empty((self.n_rows, self.n_cols), dtype="<U2")
        for idx in range(self.n_states):
            state = self.index_to_state(idx)
            if state in self.terminal_states:
                grid[state] = "X"
            else:
                grid[state] = arrow_map.get(policy[idx], "?")
        print("Policy visualization:")
        print("\n".join(" ".join(row) for row in grid))
<<<<<<< HEAD
        print()
=======
        print()

    def get_valid_actions(self, state):
        """
        Retourne la liste des actions valides depuis l'état (row, col).
        Actions possibles : 0=up, 1=down, 2=left, 3=right
        """
        if self.is_terminal(state):
            return []
        row, col = state
        actions = []
        if row > 0:
            actions.append(0)  # up
        if row < self.n_rows - 1:
            actions.append(1)  # down
        if col > 0:
            actions.append(2)  # left
        if col < self.n_cols - 1:
            actions.append(3)  # right
        return actions
>>>>>>> 79ea47c97567ca5f47a2c6286364e1e680df022e
