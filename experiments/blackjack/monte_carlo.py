"""
    Use Monte Carlo Method to solve Black Jack
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pylab as plt

from utils.misc_utils import set_seed
from mpl_toolkits.mplot3d import Axes3D
from utils.gym_utils import get_observation_shape
from traditional_algos.monte_carlo import MonteCarlo

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

def plot_value_function(value_function, usable_ace, title, save_path=None):
    """
    Plot the value function as a 3D surface plot.
    
    Args:
        value_function: The state-value function to plot.
        usable_ace: Whether to plot the values with a usable ace or without.
        title: The title for the plot.
        save_path: If provided, save the plot to this path.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Player sum ranges from 12 to 21
    x = np.arange(12, 22)
    # Dealer's showing card ranges from 1 to 10
    y = np.arange(1, 11)
    X, Y = np.meshgrid(x, y)

    # Extract the corresponding part of the value function
    Z = np.array([[value_function[player_sum, dealer_card, usable_ace] 
                   for player_sum in x] for dealer_card in y])

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_policy(policy, usable_ace, title, save_path=None):
    """
    Plot the optimal policy as a 2D heatmap.
    
    Args:
        policy: The policy to plot.
        usable_ace: Whether to plot the policy with a usable ace or without.
        title: The title for the plot.
        save_path: If provided, save the plot to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

    # Player sum ranges from 12 to 21
    y = np.arange(12, 22)
    # Dealer's showing card ranges from 1 to 10
    x = np.arange(1, 11)
    X, Y = np.meshgrid(x, y)

    # Extract the corresponding part of the policy
    Z = np.array([[np.argmax(policy[player_sum, dealer_card, usable_ace]) 
                   for dealer_card in x] for player_sum in y])

    cax = ax.matshow(Z, cmap='coolwarm')

    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels(x)
    ax.set_yticklabels(y[::-1])  # Invert the y-axis to reverse the player sum order
    
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title(title, pad=20)  # Increase pad to move title up

    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Ensure the title is saved correctly

    plt.show()

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initiate policy #####
    env = gym.make('Blackjack-v1', natural=False, sab=True)

    initial_policy = create_initial_policy(env)
    agent = MonteCarlo(env=env, initial_policy=initial_policy)

    ##### Step 1: Try monte carlo prediction to estimate the value function #####
    num_episode = 500_000
    agent.prediction(num_episode=num_episode)
    # Plot the value function #####
    value_function = agent.value_function
    # Plot for states with a usable ace
    usable_ace_plot_path = os.path.join(log_dir, 'state_value_function_usable_ace.png')
    plot_value_function(value_function, usable_ace=1, title=f"State-Value Function (Usable Ace)\nEpisode:{num_episode}", save_path=usable_ace_plot_path)
    # Plot for states without a usable ace
    no_usable_ace_plot_path = os.path.join(log_dir, 'state_value_function_no_usable_ace.png')
    plot_value_function(value_function, usable_ace=0, title=f"State-Value Function (No Usable Ace)\nEpisode:{num_episode}", save_path=no_usable_ace_plot_path)

    ##### Step 2: Try monte carlo estimation and control to estimate the q function and optimal policy pi #####
    num_episode = 1_000_000
    agent.reset()
    agent.estimation_and_control(num_episode=num_episode)
    policy_array = agent.policy

    # Plot and save the optimal policy for states with a usable ace
    usable_ace_policy_path = os.path.join(log_dir, 'optimal_policy_usable_ace.png')
    plot_policy(policy_array, usable_ace=1, title=f"Optimal Policy (Usable Ace)\nEpisode:{num_episode}", save_path=usable_ace_policy_path)
    
    # Plot and save the optimal policy for states without a usable ace
    no_usable_ace_policy_path = os.path.join(log_dir, 'optimal_policy_no_usable_ace.png')
    plot_policy(policy_array, usable_ace=0, title=f"Optimal Policy (No Usable Ace)\nEpisode:{num_episode}", save_path=no_usable_ace_policy_path)