"""
    REINFORCE 
"""

import numpy as np
import gymnasium as gym

class PolicyBase:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def action_probabilities(self):
        raise NotImplementedError

    def select_action(self):
        raise NotImplementedError

class LinearApproximator(PolicyBase):
    def __init__(self, initial_weights=None):
        self.initial_weights = initial_weights
        self.reset()

    def reset(self):
        self.w = self.initial_weights if self.initial_weights is not None else np.array([2.0, -2.0])

    def action_probabilities(self):
        """Compute action probabilities using softmax."""
        exp_w = np.exp(self.w)
        return exp_w / np.sum(exp_w)

    def select_action(self, state):
        """Select action based on the probabilities."""
        probs = self.action_probabilities()
        action = np.random.choice(len(self.w), p=probs)

        # Compute the gradient of the log-probability
        dlog_pi = -probs  # This subtracts the probability distribution from the one-hot encoded action
        dlog_pi[action] += 1
        
        return action, dlog_pi

class REINFORCE:
    def __init__(self, env: gym.Env, gamma=1.0, alpha=5e-4, policy: PolicyBase = None) -> None:
        self.env = env
        self.gamma = gamma
        self.alpha = alpha

        self.policy = policy
        if policy == None:
            raise ValueError("A policy must be provided.")
    
    def reset(self):
        self.policy.reset()
    
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
        self.reset()
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            dlog_pis = []
            rewards = []

            while not done:
                action, dlog_pi = self.policy.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                dlog_pis.append(dlog_pi)
                rewards.append(reward)

                state = next_state

            returns = self.compute_returns(rewards)
            self.update(dlog_pis, returns)

    def update(self, dlog_pis, returns):
        """Update the policy weights using the monte carlo method."""
        gradients = np.zeros_like(self.policy.w)
        for t, (dlog_pi, return_) in enumerate(zip(dlog_pis, returns)):
            # Incorporate the discount factor gamma
            discounted_return = (self.gamma ** t) * return_
            gradients += dlog_pi * discounted_return
        self.policy.w += self.alpha * gradients
