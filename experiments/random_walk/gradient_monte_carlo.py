"""
    Use Gradient Monte Carlo Method to solve 1000 States Random Walk.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from envs import RandomWalk
from traditional_algos.monte_carlo import GradientMonteCarlo, FunctionApproximator

class StateAggregation(FunctionApproximator):
    def __init__(self, n_states, n_bins):
        self.n_bins = n_bins
        self.theta = np.zeros(n_bins)
        self.bin_width = n_states // n_bins

    def reset(self):
        """Reset the parameter vector."""
        self.theta = np.zeros(self.n_bins)

    def update(self, state, target, alpha):
        """Update the parameters using the Gradient Monte Carlo rule."""
        bin_index = min(state // self.bin_width, self.n_bins - 1)
        value_estimate = self.theta[bin_index]
        self.theta[bin_index] += alpha * (target - value_estimate)

    def print_parameters(self):
        """Print the parameter vector."""
        print("Parameter vector (theta):", self.theta)

    def get_value_function(self, n_states):
        """Get the value function for all states."""
        value_function = np.zeros(n_states)
        for i in range(n_states):
            bin_index = min(i // self.bin_width, self.n_bins - 1)
            value_function[i] = self.theta[bin_index]
        return value_function

if __name__ == "__main__":
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initialize agent #####
    env = RandomWalk()
    approximation_function = StateAggregation(n_states=1000, n_bins=10)
    agent = GradientMonteCarlo(env=env, alpha=2e-5, approximation_function=approximation_function)
    
    ##### 1. Approximate the value function #####
    num_episode = 100_000
    agent.prediction(num_episode=num_episode)

    ##### 2. Plot the value function #####
    # Get the value function from the approximation
    value_function = approximation_function.get_value_function(n_states=1000)

    # Plot the value function
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 1001), value_function, label="Estimated Value Function")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title("Estimated Value Function for 1000-State Random Walk")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(log_dir, 'value_function_1000_state_random_walk.png')
    plt.savefig(plot_path)
    plt.show()