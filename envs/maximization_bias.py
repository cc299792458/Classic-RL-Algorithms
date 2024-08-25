import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MaximizationBias(gym.Env):
    def __init__(self, num_action_at_state_B=10):
        super(MaximizationBias, self).__init__()
        
        self.num_action_at_state_B = num_action_at_state_B
        # Define action space: two actions in state A (left (0) and right (1)), and several actions in state B
        self.action_space = spaces.Discrete(self.num_action_at_state_B)
        
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
            if action in range(self.num_action_at_state_B):  # Any action in state B leads to termination
                self.state = 2  # Transition to terminal state
                reward = np.random.normal(-0.1, 1.0)  # Sample reward from N(-0.1, 1.0)
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
    env = MaximizationBias(num_action_at_state_B=1)
    num_episode = 100
    rewards_at_B = []

    # Run 100 episodes
    for episode in range(num_episode):
        state, _ = env.reset()
        done = False

        while not done:
            if state == 0:
                action = 0  # Always choose left to go to state B
            else:
                action = env.action_space.sample()  # Randomly sample an action for state B
                
            next_state, reward, done, _, _ = env.step(action)
            
            # If we are in state B, record the reward
            if state == 1:
                rewards_at_B.append(reward)

            state = next_state

    # Calculate the average reward received in state B
    average_reward_at_B = np.mean(rewards_at_B)
    print(f"Average reward at state B over {num_episode} episodes: {average_reward_at_B}")