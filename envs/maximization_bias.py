import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MaximizationBias(gym.Env):
    def __init__(self):
        super(MaximizationBias, self).__init__()
        
        # Define action space: two actions in state A (left (0) and right (1)), and 10 actions in state B
        self.action_space = spaces.Discrete(10)
        
        # Define observation space: two non-terminal states A (0) and B (1), and a terminal state (2)
        self.observation_space = spaces.Discrete(3)
        
        self.state = 0  # Initial state is A
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0  # Reset to the initial state A
        self.done = False
        return self.state, {}

    def step(self, action):
        if self.state == 0:  # In state A
            if action == 0:  # Choosing left, transition to state B
                self.state = 1
                reward = 0
            elif action == 1:  # Choosing right, transition to terminal state
                self.state = 2
                reward = 0
                self.done = True
            else:
                raise ValueError(f"Invalid action in state A: {action}")
        elif self.state == 1:  # In state B
            if action in range(10):  # Any action in state B leads to termination
                self.state = 2  # Transition to terminal state
                reward = np.random.normal(0.1, 1.0)  # Sample reward from N(0.1, 1.0)
                self.done = True
            else:
                raise ValueError(f"Invalid action in state B: {action}")
        else:
            raise ValueError(f"Invalid state: {self.state}")

        return self.state, reward, self.done, False, {}

    def render(self):
        if self.state == 0:
            print("Current state: A")
        elif self.state == 1:
            print("Current state: B")
        else:
            print("Current state: Terminal")

    def close(self):
        pass

if __name__ == "__main__":
    env = MaximizationBias()
    
    # Run a few episodes to demonstrate environment behavior
    for episode in range(5):
        state, _ = env.reset()
        done = False
        print(f"Episode {episode} starts")
        
        while not done:
            if state == 0:
                action = env.action_space.sample() % 2  # Randomly sample an action for state A
            else:
                action = env.action_space.sample()  # Randomly sample an action for state B
                
            next_state, reward, done, _, _ = env.step(action)
            env.render()
            print(f"Action: {action}, Reward: {reward}")
        print(f"Episode {episode} ends\n")
