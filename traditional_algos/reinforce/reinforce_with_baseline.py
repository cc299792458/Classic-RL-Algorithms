import numpy as np

from gymnasium import Env
from .reinforce import REINFORCE
from traditional_algos.reinforce.reinforce import PolicyBase

class ValueEstimation:
    def __init__(self, initial_weights=None) -> None:
        self.initial_weights = initial_weights
        self.reset()

    def reset(self):
        self.w = self.initial_weights if self.initial_weights is not None else np.array([0.0, 0.0, 0.0, 0.0])

    def predict(self, state_vector):
        """Predict the value of the given state."""
        return np.dot(self.w, state_vector)

class REINFORCEWithBaseline(REINFORCE):
    def __init__(self, env: Env, gamma=1.0, alpha_policy=2e-3, alpha_baseline=5e-3, policy: PolicyBase = None, baseline: ValueEstimation = None) -> None:
        super().__init__(env, gamma, None, policy)
        self.baseline = baseline
        self.alpha_policy = alpha_policy
        self.alpha_baseline = alpha_baseline

    def reset(self):
        self.policy.reset()
        self.baseline.reset()

    def update(self, dlog_pis, returns, state_vectors):
        """Update the policy and baseline weights using the Monte Carlo method."""
        gradients_policy = np.zeros_like(self.policy.w)
        gradients_baseline = np.zeros_like(self.baseline.w)
        for t, (dlog_pi, return_, state_vector) in enumerate(zip(dlog_pis, returns, state_vectors)):
            # Estimate baseline value
            baseline_value = self.baseline.predict(state_vector)

            # Compute delta (advantage)
            delta = return_ - baseline_value

            # Update policy weights using the delta
            gradients_policy += (self.gamma ** t) * dlog_pi * delta

            # Update baseline weights using the delta
            gradients_baseline += delta * state_vector

        # Apply policy weight update
        self.policy.w += self.alpha_policy * gradients_policy

        # Update baseline weights
        self.baseline.w += self.alpha_baseline * gradients_baseline

    def train(self, num_episodes):
        """Train the agent and log the returns and weights for each episode."""
        self.reset()
        episode_returns = []
        weights_history = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            dlog_pis = []
            rewards = []
            state_vectors = []

            # Record the weights before each episode
            weights_history.append(self.policy.w.copy())

            while not done:
                action, dlog_pi = self.policy.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                dlog_pis.append(dlog_pi)
                rewards.append(reward)
                state_vectors.append(self._to_vector(state))

                state = next_state

            returns = self.compute_returns(rewards)
            self.update(dlog_pis, returns, state_vectors)

            undiscounted_return = sum(rewards)
            episode_returns.append(undiscounted_return)

        return episode_returns, weights_history
    
    def _to_vector(self, state):
        state_vector = np.zeros(self.env.observation_space.n)
        state_vector[state] = 1.0

        return state_vector