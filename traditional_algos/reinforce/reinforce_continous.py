"""
    REINFORCE with Continuous Action Parameterization 
"""

import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar

from .reinforce import REINFORCE
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, output_dim)
        self.sigma = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))  # sigma must be positive, so we apply exp
        return mu, sigma
    
    def reset_parameters(self):
        """Reset the parameters of the network."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
class REINFORCEContinuous(REINFORCE):
    def __init__(self, env: gym.Env, gamma=0.99, lr=1e-3, policy: nn.Module = None, device=None):
        self.device = device if device else torch.device("cpu")  # Set the device
        super().__init__(env, gamma, lr, policy)
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def reset(self):
        self.policy.reset_parameters()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mu, sigma = self.policy(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().item(), log_prob

    def update(self, log_probs, returns):
        loss = 0
        for log_prob, return_ in zip(log_probs, returns):
            loss -= log_prob * return_  # REINFORCE loss is negative log_prob times the return
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        self.reset()
        episode_returns = []
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):  # Add tqdm for progress bar
            state, _ = self.env.reset()
            done = False

            log_probs = []
            rewards = []

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step([action])
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state

            returns = self.compute_returns(rewards)
            self.update(log_probs, returns)

            episode_returns.append(sum(rewards))

        return episode_returns
