"""
    Vanilla Policy Gradient with Continuous Action Parameterization
"""

import os
import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim

from tqdm import tqdm  # Import tqdm for progress bar
from torch.distributions import Normal

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)

    def reset_parameters(self):
        """Reset the parameters of the network."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, output_dim)
        self.logstd = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        std = torch.exp(logstd)

        return mu, std
    
    def reset_parameters(self):
        """Reset the parameters of the network."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class VanillaPolicyGradient:
    def __init__(self, 
                 env: gym.Env, 
                 gamma=0.99,
                 lr_policy=1e-4, 
                 lr_baseline=3e-4, 
                 policy: nn.Module = None, 
                 baseline: nn.Module = None, 
                 device=None, 
                 save_interval=10, 
                 checkpoint_dir=None,
                 batch_size=32):
        self.env = env
        self.gamma = gamma
        self.policy = policy
        self.baseline = baseline
        self.device = device if device else torch.device("cpu")
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up optimizers for both the policy and the baseline
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_baseline = optim.Adam(self.baseline.parameters(), lr=lr_baseline)

        # Move models to the appropriate device
        self.policy.to(self.device)
        self.baseline.to(self.device)

    def reset(self):
        self.policy.reset_parameters()
        self.baseline.reset_parameters()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mu, std = self.policy(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # Clamp the action to ensure it stays within the valid range
        action = action.clamp(-2.0, 2.0)

        return action.cpu().item(), log_prob

    def update(self, log_probs, returns, states):
        policy_loss = 0
        value_loss = 0
        for log_prob, return_, state in zip(log_probs, returns, states):
            baseline_value = self.baseline(state)
            advantage = return_ - baseline_value.item()
            
            # Update policy
            policy_loss -= log_prob * advantage

            # Update baseline
            value_loss += nn.functional.mse_loss(baseline_value, torch.tensor([return_], dtype=torch.float32).to(self.device))
        
        # Update policy network
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Update baseline network
        self.optimizer_baseline.zero_grad()
        value_loss.backward()
        self.optimizer_baseline.step()

    def compute_returns(self, rewards):
        """Compute the returns from the rewards."""
        G = 0
        returns = []
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def train(self, total_timesteps):
        self.reset()
        episode_returns = []
        timestep = 0
        
        with tqdm(total=total_timesteps, desc="Training Timesteps") as pbar:
            while timestep < total_timesteps:
                state, _ = self.env.reset()
                done = False

                log_probs = []
                rewards = []
                states = []
                episode_return = 0

                while not done and timestep < total_timesteps:
                    action, log_prob = self.select_action(state)
                    next_state, reward, terminated, truncated, _ = self.env.step([action])
                    done = terminated or truncated

                    log_probs.append(log_prob)
                    rewards.append(reward)
                    states.append(torch.tensor(state, dtype=torch.float32).to(self.device))

                    state = next_state
                    episode_return += reward
                    timestep += 1

                    pbar.update(1)  # Update the progress bar

                    # Update the policy every batch_size timesteps
                    if timestep % self.batch_size == 0:
                        returns = self.compute_returns(rewards)
                        self.update(log_probs, returns, states)
                        log_probs = []
                        rewards = []
                        states = []

                episode_returns.append(episode_return)

                # Save checkpoint at intervals based on timesteps
                if timestep % self.save_interval == 0 and self.checkpoint_dir:
                    self.save_checkpoint(timestep)

        return episode_returns

    def save_checkpoint(self, timestep):
        """Save the model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'policy_checkpoint_{timestep}.pth')
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
