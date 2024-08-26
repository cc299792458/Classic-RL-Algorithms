import numpy as np
import gymnasium as gym

from gymnasium import spaces

class BairdsCounterexample(gym.Env):
    def __init__(self):
        super(BairdsCounterexample, self).__init__()

        # There are 7 states in this environment
        self.n_states = 7

        # There are 2 actions: dashed action and solid action
        self.n_actions = 2

        # Define the action space and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)

        # The transition probabilities for the dashed action (action=0)
        # This takes the agent to one of the 6 upper states (state 0 to state 5)
        self.transition_probs_dashed = np.array([
            [1/6] * 6 + [0],
            [1/6] * 6 + [0],
            [1/6] * 6 + [0],
            [1/6] * 6 + [0],
            [1/6] * 6 + [0],
            [1/6] * 6 + [0],
            [1/6] * 6 + [0]
        ])

        # The transition probabilities for the solid action (action=1)
        # This always takes the agent to the 7th state (state 6)
        self.transition_probs_solid = np.array([
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Reward is always zero in this environment
        self.reward = 0

        # Discount factor
        self.gamma = 0.99

        # Start each episode at a random state among the first 6 states
        self.reset()

    def reset(self):
        # Start in one of the first 6 states uniformly at random
        self.state = np.random.choice(np.arange(6))
        return self.state, {}

    def step(self, action):
        if action == 0:
            # Dashed action: choose one of the first 6 states randomly
            next_state = np.random.choice(np.arange(self.n_states), p=self.transition_probs_dashed[self.state])
        else:
            # Solid action: always go to state 6
            next_state = np.random.choice(np.arange(self.n_states), p=self.transition_probs_solid[self.state])
        
        return next_state, self.reward, False, False, {}

    def render(self, mode='human'):
        print(f"Current state: {self.state}")

if __name__ == "__main__":
    env = BairdsCounterexample()
    state, _ = env.reset()
    print(f"Initial state: {state}")

    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break