"""
    K Arm Bandit Problem, both Stationary and Non-stationary Versions.

    This examples come from the RL Book, chapter 2
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class KArmedBandit(gym.Env):
    """
        Simulates a multi-armed bandit problem where each arm (action) has a different
        reward distribution. The agent needs to choose actions over time to maximize 
        the cumulative reward. This implementation supports both stationary and 
        non-stationary reward distributions.
    """
    def __init__(self, k=10, max_time_steps=1000):
        super(KArmedBandit, self).__init__()
        
        self.k = k
        self.max_time_steps = max_time_steps

        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Discrete(1)

        self.reward_means = None
        self.time_step = 0

        self.reset()

    def reset(self):
        self.time_step = 0
        # Reinitialize the reward means for each arm
        # TODO: Why the variance increases when using uniform sampling?
        # self.reward_means = np.random.uniform(-1, 1, self.k)
        self.reward_means = np.random.randn(self.k)
        
        return 0, {}  # No meaningful observation, return arbitrary
    
    def step(self, action):
        # Generate a reward based on the selected action's reward distribution
        reward = np.random.randn() + self.reward_means[action]
        
        self.time_step += 1
        
        terminated = False
        truncated = self.time_step >= self.max_time_steps

        return 0, reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        # Render the current reward means and time step (for debugging purposes)
        print(f"Current reward means: {self.reward_means}")
        print(f"Time step: {self.time_step}/{self.max_time_steps}")

    @property
    def optimal_actions(self):
        return np.flatnonzero(self.reward_means == np.max(self.reward_means))
    
class NonStationaryBandit(KArmedBandit):
    """
        Extends the stationary bandit by allowing the reward means to change over time 
        according to a random walk process.
    """
    def __init__(self, k=10, max_time_steps=1000, walk_std=0.01):
        super().__init__(k=k, max_time_steps=max_time_steps)
        self.walk_std = walk_std

    def reset(self):
        # Start all q*(a) equal, for example, all equal to zero
        self.reward_means = np.zeros(self.k)
        self.current_step = 0

        return 0, {}

    def step(self, action):
        # Random walk: increment each q*(a) with a normal distribution (mean 0, std walk_std)
        self.reward_means += np.random.normal(0, self.walk_std, self.k)
        reward = np.random.randn() + self.reward_means[action]
        self.current_step += 1

        terminated = False
        truncated = self.max_time_steps is not None and self.current_step >= self.max_time_steps

        return 0, reward, terminated, truncated, {}
    
if __name__ == '__main__':
    env = KArmedBandit(k=10, max_time_steps=1000)
    state, _ = env.reset()

    total_reward = 0
    # Simulate taking actions
    for time_step in range(100):
        action = env.action_space.sample()  # Randomly pick an action
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Time step: {time_step}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        total_reward += reward

        if terminated or truncated:
            print("Episode finished!")
            break
    print(f"Total reward: {total_reward}")
