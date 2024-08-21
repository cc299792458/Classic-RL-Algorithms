import numpy as np
import gymnasium as gym

class EpsilonGreedy:
    def __init__(self, env: gym.Env, epsilon=0.1, step_size=None):
        self.env = env
        self.epsilon = epsilon
        self.step_size = step_size

        self.n_actions = env.action_space.n
        self.reset()
        
    def reset(self):
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)


    def select_action(self):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values)
        
    def update_q_values(self, action, reward):
        self.action_counts[action] += 1
        if self.step_size is None:
            # Sample-average method
            alpha = 1.0 / self.action_counts[action]
        else:
            # Constant step-size method
            alpha = self.step_size
        self.q_values[action] += alpha * (reward - self.q_values[action])
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon