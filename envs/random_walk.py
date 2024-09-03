"""
    Random Walk Env, for the RL Book Example 9.1.
"""

import gymnasium as gym
from gymnasium import spaces

class RandomWalk(gym.Env):
    """
        This environment simulates a random walk on a 1D line with a specified number
        of states. The agent starts in the middle and moves left or right by a random
        amount, determined by the action. The episode ends when the agent reaches
        either end of the line, receiving a reward of -1 or 1 depending on the side.
    """
    def __init__(self, n_states=1000, n_neighbors=100):
        super(RandomWalk, self).__init__()
        self.n_states = n_states
        self.n_neighbors = n_neighbors
        self.start_state = n_states // 2
        self.current_state = self.start_state

        # Define action space: 0 to 199, where 0-99 = move left, 100-199 = move right
        self.action_space = spaces.Discrete(2 * n_neighbors)

        # Define observation space: a single integer representing the agent's position
        self.observation_space = spaces.Discrete(self.n_states)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the starting state.
        """
        super().reset(seed=seed)
        self.current_state = self.start_state
        return self.current_state, {}

    def step(self, action):
        """
        Take an action in the environment and return the results.
        """
        # Determine the magnitude and direction of the move
        if action < self.n_neighbors:  # Move left
            move = -(action + 1)
        else:  # Move right
            move = (action - self.n_neighbors + 1)

        # Update the current state
        self.current_state += move

        # Check if we have gone out of bounds (i.e., reached a terminal state)
        terminated = False
        reward = 0.0
        if self.current_state < 0:
            self.current_state = 0
            reward = -1.0
            terminated = True
        elif self.current_state >= self.n_states:
            self.current_state = self.n_states - 1
            reward = 1.0
            terminated = True

        truncated = False  # No time limit truncation
        return self.current_state, reward, terminated, truncated, {}

    def render(self):
        """
        Print the current state of the environment (optional).
        """
        print(f"State: {self.current_state}")

    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    env = RandomWalk()

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Random action (move left or right by some amount)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Total Reward: {total_reward}")
