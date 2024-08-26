"""
    Show the Divergence When using Semi-Gradient Q-Learning to solve the Baird's Counterexample
"""

# TODO: Why it converges at the end?

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.bairds_counterexample import BairdsCounterexample
from traditional_algos.td_learning.q_learning import SemiGradientQLearning

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initiate policy #####    
    env = BairdsCounterexample()
    agent = SemiGradientQLearning(env=env, gamma=0.5, epsilon=0.1, alpha=0.01)

    time_steps = 10_000
    weights_history = []

    ##### 1. Use Semi-Gradient Q-Learning to solve the Baird's Counterexample #####
    state, _ = env.reset()

    for _ in tqdm(range(time_steps), desc="Training Progress"):
        action = np.random.choice(np.arange(agent.num_action), p=agent.policy[state])
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update Q function
        agent.update_q_function(state, action, reward, next_state)
        state = next_state

        # Record weights after each episode
        weights_history.append(agent.weights.copy())

    # Convert weights history to a NumPy array for easy plotting
    weights_history = np.array([np.array(w) for w in weights_history])

    # Plotting the weights for each action
    plt.figure(figsize=(12, 8))
    for action in range(agent.num_action):
        for i in range(agent.feature_dim):
            plt.plot(weights_history[:, action, i], label=f'Action {action+1}, Weight {i+1}')

    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('Weight Divergence in Semi-Gradient Q-Learning (Baird\'s Counterexample)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place the legend outside the plot
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(log_dir, 'bairds_counterexample_weights_convergence.png')
    plt.savefig(plot_path, bbox_inches='tight')

    # Show the plot
    plt.show()
