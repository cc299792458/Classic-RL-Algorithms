"""
    GridWorld.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorld(gym.Env):
    """
        A simple grid-based environment where an agent navigates from a start position ('S') to a goal position ('G').
        The grid is defined by a specified width and height, and the agent can move in four directions: up, down, left, and right.
        Each step incurs a reward of -1, encouraging the agent to reach the goal in as few steps as possible.
        The environment is deterministic, and episodes terminate when the agent reaches the goal or when a maximum number of steps is exceeded.
    """
    def __init__(self, height=4, width=4, start_position=(0, 0), goal_position=None, max_episode_length=True):
        """Initialize the GridWorld environment."""
        self.height = height
        self.width = width
        
        # Convert start and goal positions from (x, y) to state indices
        self.start_state = self._xy_to_state(start_position)
        self.goal_state = self._xy_to_state(goal_position if goal_position is not None else (self.height - 1, self.width - 1))
        
        # Define action space: up, down, right, left
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: a single integer representing the agent's position
        self.observation_space = spaces.Discrete(self.width * self.height)
        
        # Initialize the transition dynamics
        self.P = self._create_transition_matrix()

        # Set maximum episode length; if max_episode_length is True, use the default limit
        if max_episode_length is True:
            self.max_episode_length = 10 * self.width * self.height
        elif max_episode_length is False:
            self.max_episode_length = None
        else:
            self.max_episode_length = max_episode_length

        self.current_step = 0

        # Initial state
        self.reset()

    def _xy_to_state(self, position):
        """Convert (x, y) coordinates to a linear state index."""
        x, y = position
        return x * self.width + y

    def _state_to_xy(self, state):
        """Convert a linear state index to (x, y) coordinates."""
        return divmod(state, self.width)

    def _create_transition_matrix(self):
        """
        Create the transition matrix P[state][action] -> [(probability, next_state, reward, terminated)]
        """
        P = {}
        for x in range(self.height):
            for y in range(self.width):
                state = self._xy_to_state((x, y))
                P[state] = {a: [] for a in range(self.action_space.n)}
                
                # Define possible transitions for each action
                for action in range(self.action_space.n):
                    new_x, new_y = x, y
                    if action == 0:  # up
                        new_x = max(x - 1, 0)
                    elif action == 1:  # down
                        new_x = min(x + 1, self.height - 1)
                    elif action == 2:  # right
                        new_y = min(y + 1, self.width - 1)
                    elif action == 3:  # left
                        new_y = max(y - 1, 0)
                    
                    next_state = self._xy_to_state((new_x, new_y))
                    terminated = self.is_terminal_state(next_state)
                    reward = -1  # Constant reward for each step
                    
                    # Add this transition to the P dictionary
                    P[state][action].append((1.0, next_state, reward, terminated))
                    
        return P
    
    def is_terminal_state(self, state):
        return state == self.goal_state

    def reset(self):
        """
        Reset the grid to the initial state
        """
        self.position = self.start_state
        self.current_step = 0
        return self.position, {}

    def step(self, action):
        self.current_step += 1
        transitions = self.P[self.position][action]
        prob, next_state, reward, terminated = transitions[0]  # Since deterministic, only one transition
        self.position = next_state
        if self.max_episode_length is not None:
            truncated = self.current_step >= self.max_episode_length
        else:
            truncated = False
        info = {'terminal': self.is_terminal_state(next_state)}

        return self.position, reward, terminated, truncated, info

    def render(self):
        # Create an empty grid filled with '.'
        grid = np.full((self.height, self.width), '.', dtype=str)
        
        # Mark the start position with 'S'
        start_x, start_y = self._state_to_xy(self.start_state)
        grid[start_x, start_y] = 'S'

        # Mark the goal position with 'G'
        goal_x, goal_y = self._state_to_xy(self.goal_state)
        grid[goal_x, goal_y] = 'G'

        # Mark the current position with 'A'
        x, y = self._state_to_xy(self.position)
        grid[x, y] = 'A'
        
        # Simple print render
        for row in grid:
            print(' '.join(row))

    def _print_value_function(self, value_function):
        """
        Print the value function as a grid. 
        Each cell shows the value for the corresponding state.
        """
        for x in range(self.height):
            row_values = []
            for y in range(self.width):
                state = self._xy_to_state((x, y))
                row_values.append(f"{value_function[state]:.2f}")
            print(" | ".join(row_values))
            if x < self.height - 1:
                print("-" * (6 * self.width - 1))

    def _print_q_function(self, q_function):
        """
        Print the Q-function as a grid. 
        Each cell shows the Q-values for all actions in the corresponding state.
        """
        action_arrows = {0: '↑', 1: '↓', 2: '→', 3: '←'}
        for x in range(self.height):
            for a in range(4):  # For each action
                row_values = []
                for y in range(self.width):
                    state = self._xy_to_state((x, y))
                    value = q_function[state][a]
                    row_values.append(f"{action_arrows[a]}:{value:.2f}")
                print(" | ".join(row_values))
            if x < self.height - 1:
                print("-" * (8 * self.width - 1))

    def _print_policy(self, policy):
        """
        Print the policy as a grid.
        Each cell shows the best action(s) for the corresponding state.
        """
        action_arrows = {0: '↑', 1: '↓', 2: '→', 3: '←'}
        for x in range(self.height):
            row_policy = []
            for y in range(self.width):
                state = self._xy_to_state((x, y))
                best_actions = np.argwhere(policy[state] == np.max(policy[state])).flatten()
                arrows = ''.join([action_arrows[a] for a in best_actions])
                row_policy.append(arrows.center(4))
            print(" | ".join(row_policy))
            if x < self.height - 1:
                print("-" * (6 * self.width - 1))

if __name__ == '__main__':
    # Example of using the GridWorld environment with custom start and goal positions
    env = GridWorld(width=9, height=6, start_position=(0, 0), goal_position=None)  # Start at top-left, goal at bottom-right
    env.reset()
    env.render()
    state, reward, terminated, truncated, info = env.step(2)  # Example action: move right
    env.render()
    print(f"Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
