"""
    REINFORCE with Continuous Action Parameterization solving Pendulum
"""

# TODO: Modify the network architecture or the hyperparameters to train the env successfully.

import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from utils.misc_utils import set_seed
from traditional_algos.reinforce import REINFORCEContinuous, PolicyNetwork, ValueNetwork

class REINFORCEContinuousLogging(REINFORCEContinuous):
    def __init__(self, 
                 env: gym.Env, 
                 gamma=0.99,
                 lr_policy=1e-4, 
                 lr_baseline=3e-4, 
                 policy: nn.Module = None, 
                 baseline: nn.Module = None, 
                 device=None, 
                 save_interval=10, 
                 checkpoint_dir=None):
        super().__init__(env, gamma, lr_policy, lr_baseline, policy, baseline, device)
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, num_episodes):
        self.reset()
        episode_returns = []
        
        for episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes"):
            state, _ = self.env.reset()
            done = False

            log_probs = []
            rewards = []
            states = []

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step([action])
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                states.append(torch.tensor(state, dtype=torch.float32).to(self.device))

                state = next_state

            returns = self.compute_returns(rewards)
            self.update(log_probs, returns, states)

            episode_return = sum(rewards)
            episode_returns.append(episode_return)

            # Save checkpoint at intervals
            if episode % self.save_interval == 0 and self.checkpoint_dir:
                self.save_checkpoint(episode)

        return episode_returns

    def save_checkpoint(self, episode):
        """Save the model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'policy_checkpoint_{episode}.pth')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'baseline_state_dict': self.baseline.state_dict()
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load the model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])
        self.policy.to(self.device)
        self.baseline.to(self.device)

    def test_agent(self, env, num_episodes=1):
        """Test the agent using the loaded policy checkpoint."""
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step([action])
                done = terminated or truncated

                total_reward += reward
                state = next_state

                env.render()
                print(f'state: {state}')

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == '__main__':
    set_seed()
    # Initialize the environment and device
    env = gym.make("Pendulum-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the policy network, value network (baseline), and REINFORCE agent with logging
    policy = PolicyNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.shape[0])
    baseline = ValueNetwork(input_dim=env.observation_space.shape[0])
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    agent = REINFORCEContinuousLogging(
        env=env,
        gamma=0.99,
        lr_policy=1e-4,
        lr_baseline=3e-4,
        policy=policy,
        baseline=baseline,
        device=device,
        save_interval=10,  # Save every 10 episodes
        checkpoint_dir=checkpoint_dir
    )

    # Train the agent
    num_episodes = 2000
    returns = agent.train(num_episodes)

    # Plot the returns over episodes and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('REINFORCE Continuous - Pendulum')
    plt.grid(True)
    log_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(log_dir, 'reinforce_continuous_pendulum.png'))
    plt.show()

    # Create a new environment instance for testing with rendering enabled
    test_env = gym.make("Pendulum-v1", render_mode='human')

    # Test and render the agent with different checkpoints
    for checkpoint_episode in range(10, num_episodes + 1, 10):
        checkpoint_path = os.path.join(checkpoint_dir, f'policy_checkpoint_{checkpoint_episode}.pth')
        agent.load_checkpoint(checkpoint_path)
        agent.test_agent(test_env)
