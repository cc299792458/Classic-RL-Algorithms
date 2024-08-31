"""
    REINFORCE with Continuous Action Parameterization 
"""

import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim

from tqdm import tqdm  # Import tqdm for progress bar
from torch.distributions import Normal
from traditional_algos.reinforce import REINFORCE, REINFORCEWithBaseline

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

class REINFORCEContinuous(REINFORCEWithBaseline):
    def __init__(self, 
                 env: gym.Env, 
                 gamma=0.99, 
                 lr_policy=1e-4, 
                 lr_baseline=3e-4, 
                 policy: nn.Module = None, 
                 baseline: nn.Module = None, 
                 device=None):
        self.device = device if device else torch.device("cpu")  # Set the device
        super().__init__(env, gamma, lr_policy, lr_baseline, policy, baseline)
        self.policy.to(self.device)
        self.baseline.to(self.device)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_baseline = optim.Adam(self.baseline.parameters(), lr=lr_baseline)

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

    def train(self, num_episodes):
        self.reset()
        episode_returns = []
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
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

            episode_returns.append(sum(rewards))

        return episode_returns