"""
    Gradient Monte Carlo
"""

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from traditional_algos.monte_carlo import MonteCarlo

class FunctionApproximator:
    """Base class for all function approximators."""
    
    def reset(self):
        """Reset the parameters of the approximator."""
        raise NotImplementedError("The reset method must be implemented by subclasses.")
    
    def update(self, state, target, alpha):
        """Update the parameters based on the state, target value, and learning rate."""
        raise NotImplementedError("The update method must be implemented by subclasses.")
    
    def print_parameters(self):
        """Print the current parameters of the approximator."""
        raise NotImplementedError("The print_parameters method must be implemented by subclasses.")

class GradientMonteCarlo(MonteCarlo):
    """Gradient Monte Carlo with General Function Approximation"""

    def __init__(self, env: gym.Env, gamma=1.0, alpha=0.01, approximation_function: FunctionApproximator = None):
        self.alpha = alpha
        # Ensure an approximation function is provided
        if approximation_function is None:
            raise ValueError("An approximation function must be provided.")
        
        self.approximation_function = approximation_function
        super().__init__(env, gamma=gamma, epsilon=0.0)

    def reset(self):
        """Reset the approximation function's parameters."""
        self.approximation_function.reset()

    def generate_episode(self):
        """Generate an episode following the current policy."""
        episode = []
        state, info = self.env.reset()
        done = False

        while not done:
            action = np.random.choice(self.env.action_space.n)
            next_state, reward, terminated, truncated, info = self.env.step(action=action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        return episode
    def update_value_function(self, episode):
        """Perform the Gradient Monte Carlo update using the provided approximation function."""
        G = 0

        for state, _, reward in reversed(episode):
            G = self.gamma * G + reward
            self.approximation_function.update(state, G, self.alpha)

    def prediction(self, num_episode):
        """Estimate the value function using Gradient Monte Carlo."""
        self.reset()
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                episode = self.generate_episode()
                self.update_value_function(episode)
                pbar.update(1)

    def print_value_function(self):
        """Print the current approximation function's parameters."""
        self.approximation_function.print_parameters()