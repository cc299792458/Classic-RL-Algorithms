import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorld(gym.Env):
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        
        # Define action space: up, down, left, right
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: a single integer representing the agent's position
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        
        # Initialize the transition dynamics
        self.P = self._create_transition_matrix()

        # Set maximum episode length as a quadratic function of grid size
        self.max_episode_length = 10 * self.grid_size * self.grid_size
        self.current_step = 0

        # Initial state
        self.reset()

    def _create_transition_matrix(self):
        """
        Create the transition matrix P[state][action] -> [(probability, next_state, reward, done)]
        """
        P = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = x * self.grid_size + y
                P[state] = {a: [] for a in range(self.action_space.n)}
                
                # Define possible transitions for each action
                for action in range(self.action_space.n):
                    new_x, new_y = x, y
                    if action == 0:  # up
                        new_x = max(x - 1, 0)
                    elif action == 1:  # down
                        new_x = min(x + 1, self.grid_size - 1)
                    elif action == 2:  # left
                        new_y = max(y - 1, 0)
                    elif action == 3:  # right
                        new_y = min(y + 1, self.grid_size - 1)
                    
                    next_state = new_x * self.grid_size + new_y
                    done = self.is_terminal_state(next_state)
                    reward = -1  # Constant reward for each step
                    
                    # Add this transition to the P dictionary
                    P[state][action].append((1.0, next_state, reward, done))
                    
        return P
    
    def is_terminal_state(self, state):
        return (state == 0) or (state == self.grid_size * self.grid_size - 1)

    def reset(self):
        """
        Reset the grid to the initial state
        """
        self.position = np.random.randint(self.observation_space.n)
        self.current_step = 0
        return self.position

    def step(self, action):
        self.current_step += 1
        transitions = self.P[self.position][action]
        prob, next_state, reward, done = transitions[0]  # Since deterministic, only one transition
        self.position = next_state
        if self.current_step >= self.max_episode_length:
            done = True
        info = {'terminal': self.is_terminal_state(next_state)}

        return self.position, reward, done, info

    def render(self):
        # Create an empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Convert position index to 2D grid coordinates and set it to 1
        x, y = divmod(self.position, self.grid_size)
        grid[x, y] = 1
        
        # Simple print render
        print(grid)

    def _print_value_function(self, value_function):
        grid_size = self.grid_size
        max_value = max(value_function)
        min_value = min(value_function)
        max_width = max(len(f"{max_value:.2f}"), len(f"{min_value:.2f}"))  # Ensure consistent width for alignment

        for i in range(grid_size):
            row_values = []
            for j in range(grid_size):
                value = value_function[i * grid_size + j]
                row_values.append(f"{value:>{max_width}.2f}")  # Align values by setting a consistent column width
            print(" | ".join(row_values))  # Use " | " as a separator between columns
            if i < grid_size - 1:
                print("-" * (max_width * grid_size + (grid_size - 1) * 3))  # Print separator line between rows

    def _print_q_function(self, q_function):
        grid_size = self.grid_size
        action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # Mapping of actions to arrows
        cell_width = max(25, grid_size * 5 + 5)  # Adjust the width based on grid size

        for i in range(grid_size):
            # Print the top row of Q-values (↑)
            top_row = []
            for j in range(grid_size):
                state = i * grid_size + j
                if self.is_terminal_state(state):
                    top_row.append("T".center(cell_width))
                else:
                    q_value = f"{action_arrows[0]} {q_function[state][0]:6.2f}"
                    top_row.append(q_value.center(cell_width))
            print(" | ".join(top_row))
            
            # Print the middle row with left and right Q-values (←, →)
            middle_row = []
            for j in range(grid_size):
                state = i * grid_size + j
                if self.is_terminal_state(state):
                    middle_row.append(" ".center(cell_width))
                else:
                    left_q_value = f"{action_arrows[2]} {q_function[state][2]:6.2f}"
                    right_q_value = f"{q_function[state][3]:6.2f} {action_arrows[3]}"
                    middle_row.append(f"{left_q_value.ljust(cell_width//2)}{right_q_value.rjust(cell_width//2)}")
            print(" | ".join(middle_row))
            
            # Print the bottom row of Q-values (↓)
            bottom_row = []
            for j in range(grid_size):
                state = i * grid_size + j
                if self.is_terminal_state(state):
                    bottom_row.append(" ".center(cell_width))
                else:
                    q_value = f"{action_arrows[1]} {q_function[state][1]:6.2f}"
                    bottom_row.append(q_value.center(cell_width))
            print(" | ".join(bottom_row))
            
            if i < grid_size - 1:
                print("-" * (cell_width * grid_size + (grid_size - 1) * 3))  # Separator line between rows

    def _print_policy(self, policy):
        grid_size = self.grid_size
        policy_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # Corrected action to arrow mapping
        max_width = 4  # Define a fixed width for each cell to accommodate up to 4 arrows
        
        for i in range(grid_size):
            row_policy = []
            for j in range(grid_size):
                state = i * grid_size + j
                if self.is_terminal_state(state):
                    cell_content = 'T'.center(max_width)  # Center 'T' within the fixed width
                else:
                    best_actions = np.argwhere(policy[state] == np.max(policy[state])).flatten()
                    arrows = ''.join([policy_arrows[action] for action in best_actions])
                    cell_content = arrows.center(max_width)  # Center arrows within the fixed width
                row_policy.append(cell_content)
            
            # Join the row content with ' | ' separator
            print(" | ".join(row_policy))
            
            if i < grid_size - 1:
                print("-" * (max_width * grid_size + (grid_size - 1) * 3))  # Print separator line between rows

if __name__ == '__main__':
    env = GridWorld()
    env.reset()
    env.render()
    state, reward, done, info = env.step(3)  # Example action: move right
    env.render()
    print(f"Info: {info}")