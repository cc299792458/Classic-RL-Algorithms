import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorld(gym.Env):
    def __init__(self):
        # Define the grid size
        self.grid_size = 4
        
        # Define action space: up, down, left, right
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: a single integer representing the agent's position
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        
        # Initialize the transition dynamics
        self.P = self._create_transition_matrix()
        
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
        return self.position

    def step(self, action):
        transitions = self.P[self.position][action]
        prob, next_state, reward, done = transitions[0]  # Since deterministic, only one transition
        self.position = next_state
        return self.position, reward, done, {}

    def render(self):
        # Create an empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Convert position index to 2D grid coordinates and set it to 1
        x, y = divmod(self.position, self.grid_size)
        grid[x, y] = 1
        
        # Simple print render
        print(grid)

if __name__ == '__main__':
    env = GridWorld()
    env.reset()
    env.render()
    state, reward, done, info = env.step(3)  # Example action: move right
    env.render()
