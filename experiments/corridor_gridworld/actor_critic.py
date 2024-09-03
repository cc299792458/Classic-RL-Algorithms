"""
    Use Actor-Critic to solve Corridor GridWorld
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.grid_world import CorridorGridWorld
from traditional_algos.policy_gradient.actor_critic import ActorCritic
from traditional_algos.policy_gradient.reinforce import LinearApproximator, ValueEstimation

class ActorCriticLogging(ActorCritic):
    def train(self, num_episodes):
        """Train the agent and log the returns and weights for each episode."""
        self.reset()
        episode_returns = []
        actor_weights_history = []
        critic_weights_history = []

        for episode in range(num_episodes):
            self.I = 1
            state, _ = self.env.reset()
            done = False

            episode_return = 0

            # Record the weights before each episode
            actor_weights_history.append(self.actor.w.copy())
            critic_weights_history.append(self.critic.w.copy())

            while not done:
                action, dlog_pi = self.actor.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                state_vector = self._to_vector(state)
                next_state_vector = self._to_vector(next_state)

                # Update the actor and critic
                self.update(dlog_pi, state_vector, next_state_vector, reward, done)

                # Move to the next state
                state = next_state
                episode_return += reward

            episode_returns.append(episode_return)

        return episode_returns, actor_weights_history, critic_weights_history

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))

    ##### 0. Load environment and agent #####
    env = CorridorGridWorld()

    actor = LinearApproximator()
    critic = ValueEstimation()
    agent = ActorCriticLogging(env=env, gamma=0.99, actor=actor, critic=critic)

    ##### 1. Train #####
    num_episodes = 1000
    num_runs = 100
    all_runs_returns = []
    all_runs_actor_weights = []
    all_runs_critic_weights = []

    for _ in tqdm(range(num_runs), desc="Total Runs"):
        episode_returns, actor_weights_history, critic_weights_history = agent.train(num_episodes=num_episodes)
        all_runs_returns.append(episode_returns)
        all_runs_actor_weights.append(actor_weights_history)
        all_runs_critic_weights.append(critic_weights_history)

    # Compute the average returns across all runs
    average_returns = np.mean(all_runs_returns, axis=0)
    average_actor_weights = np.mean(all_runs_actor_weights, axis=0)
    average_critic_weights = np.mean(all_runs_critic_weights, axis=0)

    # Plot the average returns over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(average_returns)
    plt.xlabel('Episode')
    plt.ylabel('Average Total Return')
    plt.title(f'Actor Critic\nAverage Total Return over {num_runs} Runs')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'actor_critic_average_total_return.png'))  # Save the plot
    plt.show()

    # Plot the actor weights over episodes
    plt.figure(figsize=(10, 6))
    for i in range(average_actor_weights.shape[1]):
        plt.plot(average_actor_weights[:, i], label=f'Actor Weight {i + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('Actor Critic\nActor Weights Evolution over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'actor_critic_actor_weights_evolution.png'))  # Save the plot
    plt.show()

    # Plot the critic weights over episodes
    plt.figure(figsize=(10, 6))
    for i in range(average_critic_weights.shape[1]):
        plt.plot(average_critic_weights[:, i], label=f'Critic Weight {i + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('Actor Critic\nCritic Weights Evolution over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'actor_critic_critic_weights_evolution.png'))  # Save the plot
    plt.show()
