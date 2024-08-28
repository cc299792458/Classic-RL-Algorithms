"""
    REINFORCE 
"""

import numpy as np
import gymnasium as gym

class PolicyBase:
    def __init__(self):
        pass

    def action_probabilities(self):
        raise NotImplementedError

    def select_action(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

class LinearApproximator(PolicyBase):
    def __init__(self, weights=None):
        self.w = weights if weights is not None else np.array([3.0, 0.0])

    def action_probabilities(self):
        """Compute action probabilities using softmax."""
        exp_w = np.exp(self.w)
        return exp_w / np.sum(exp_w)

    def select_action(self):
        """Select action based on the probabilities."""
        probs = self.action_probabilities()
        action = np.random.choice(len(self.w), p=probs)
        return action, np.log(probs[action])

    def update(self, log_probs, returns, alpha):
        """Update weights using the REINFORCE algorithm."""
        gradients = np.zeros_like(self.w)
        for log_prob, return_ in zip(log_probs, returns):
            gradients += log_prob * return_
        self.w += alpha * gradients

class REINFORCE:
    def __init__(self, env: gym.Env, gamma=1.0, alpha=2e-13, policy: PolicyBase = None) -> None:
        self.env = env
        self.gamma = gamma
        self.alpha = alpha

        self.policy = policy
        if policy == None:
            raise ValueError("An approximator must be provided.")
    
    def compute_returns(self, rewards):
        """Compute the returns from the rewards."""
        G = 0
        returns = []
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def train(self, num_episodes):
        """Monte Carlo Roll Out"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            log_probs = []
            rewards = []

            while not done:
                action, log_prob = self.policy.select_action()
                next_state, reward, done, truncated, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state

            returns = self.compute_returns(rewards)
            self.policy.update(log_probs, returns, self.alpha)

            if episode % 100 == 0:
                total_return = sum(rewards)
                print(f"Episode {episode}, Total Return: {total_return}")
    