import numpy as np
import gymnasium as gym
from gymnasium import spaces

class KArmedBandit(gym.Env):
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
        self.reward_means = np.random.uniform(0, 1, self.k)
        
        return 0  # No meaningful observation, return arbitrary
    
    def step(self, action):
        # Generate a reward based on the selected action's reward distribution
        reward = np.random.randn() + self.reward_means[action]
        
        self.time_step += 1
        
        done = self.time_step >= self.max_time_steps

        return 0, reward, done, {}
    
    def render(self, mode='human'):
        # Render the current reward means and time step (for debugging purposes)
        print(f"Current reward means: {self.reward_means}")
        print(f"Time step: {self.time_step}/{self.max_time_steps}")
    
if __name__ == '__main__':
    env = KArmedBandit(k=10, max_time_steps=1000)
    state = env.reset()

    total_reward = 0
    # Simulate taking actions
    for time_step in range(100):
        action = env.action_space.sample()  # Randomly pick an action
        state, reward, done, info = env.step(action)
        print(f"Time step: {time_step}, Action: {action}, Reward: {reward}, Done: {done}")

        total_reward += reward

        if done:
            print("Episode finished!")
            break
    print(f"Total reward: {total_reward}")