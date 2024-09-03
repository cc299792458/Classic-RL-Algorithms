"""
    Corridor GridWorld with Stochastic Switched Action. 

    The Example 13.1 in RL Book.
"""

from gymnasium import spaces
from .grid_world import GridWorld

class CorridorGridWorld(GridWorld):
    def __init__(self):
        super(CorridorGridWorld, self).__init__(width=4, height=1)  # Override the dimensions for the corridor

        # Override the number of states to match the corridor gridworld setup
        self.n_states = 4  # States: 0 (start), 1, 2, 3 (terminal)
        self.n_actions = 2  # Actions: 0 (left), 1 (right)

        # Adjust action space and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)

        # Reset the environment to set the initial state
        self.reset()

    def reset(self):
        # Start in the first state
        self.state = 0
        return self.state, {}

    def step(self, action):
        reward = -1  # Default reward for each step

        if self.state == 0:  # Start state
            if action == 1:  # Move right
                self.state = 1

        elif self.state == 1:  # Second state with reversed actions
            if action == 0:  # Move left (reversed, actually moves right)
                self.state = 2
            elif action == 1:  # Move right (reversed, actually moves left)
                self.state = 0

        elif self.state == 2:  # Third state
            if action == 0:  # Move left
                self.state = 1
            elif action == 1:  # Move right
                self.state = 3

        if self.state == 3:  # Terminal state
            terminated = True
        else:
            terminated = False

        return self.state, reward, terminated, False, {}

    def render(self):
        grid = ["S", " ", " ", "G"]
        grid[self.state] = "A"
        print(" | ".join(grid))

# Example usage
if __name__ == "__main__":
    env = CorridorGridWorld()
    state, _ = env.reset()
    print(f"Initial state: {state}")

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        print(f"Action: {'Right' if action == 1 else 'Left'}, Next state: {next_state}, Reward: {reward}, Done: {done}")
