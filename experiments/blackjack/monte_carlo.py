"""
    Use Monte Carlo Method to solve Black Jack
"""

import numpy as np
import gymnasium as gym
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from traditional_algos.monte_carlo import MonteCarlo
from utils.misc_utils import set_seed, get_observation_shape

def create_initial_policy(env):
    """
        Create an initial policy as a numpy array for the MonteCarlo class.
        - Stick (action 0) if the player's current sum is 20 or 21.
        - Hit (action 1) if the player's current sum is less than 20.
        
        The policy is a 2D NumPy array where each row corresponds to a state and
        each column corresponds to an action. Each entry in the array represents
        the probability of taking that action in that state.
    """
    observation_shape = get_observation_shape(observation_space=env.observation_space)
    num_actions = env.action_space.n
    policy = np.zeros((*observation_shape, num_actions))

    for state_index in np.ndindex(observation_shape):
        player_sum_index, dealer_card_index, usable_ace_index = state_index

        if player_sum_index >= 20:
            # Stick with probability 1.0 if sum is 20 or 21
            policy[state_index][0] = 1.0  # Action 0: Stick
        else:
            # Hit with probability 1.0 if sum is less than 20
            policy[state_index][1] = 1.0  # Action 1: Hit

    return policy

def plot_value_function(value_function, usable_ace, title):
    """
    Plot the value function as a 3D surface plot.
    
    Args:
        value_function: The state-value function to plot.
        usable_ace: Whether to plot the values with a usable ace or without.
        title: The title for the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Player sum ranges from 12 to 21 (index 8 to 17)
    x = np.arange(12, 22)
    # Dealer's showing card ranges from 1 to 10 (index 1 to 10)
    y = np.arange(1, 11)
    X, Y = np.meshgrid(x, y)

    # Extract the corresponding part of the value function
    Z = np.array([[value_function[player_sum - 12, dealer_card - 1, usable_ace] 
                   for player_sum in x] for dealer_card in y])

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)

    plt.show()
if __name__ == '__main__':
    set_seed()
    ##### 0. Load environment and initiate policy #####
    env = gym.make('Blackjack-v1', natural=False, sab=True)

    initial_policy = create_initial_policy(env)
    policy = MonteCarlo(env=env, initial_policy=initial_policy)

    ##### Step 1: Try monte carlo prediction to estimate the value function #####
    num_episode = 500_000
    policy.prediction(num_episode=num_episode)

    ##### Step 2: Plot the value function #####
    value_function = policy.value_function

    # Plot for states with a usable ace
    plot_value_function(value_function, usable_ace=1, title="State-Value Function (Usable Ace)")

    # Plot for states without a usable ace
    plot_value_function(value_function, usable_ace=0, title="State-Value Function (No Usable Ace)")