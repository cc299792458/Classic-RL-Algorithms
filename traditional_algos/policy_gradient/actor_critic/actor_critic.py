"""
    Actor-Critic
"""

import numpy as np

from gymnasium import Env
from traditional_algos.policy_gradient.reinforce import ValueEstimation, PolicyBase

class ActorCritic:
    def __init__(self, env: Env, gamma=1.0, alpha_actor=5e-3, alpha_critic=1e-1, actor: PolicyBase = None, critic: ValueEstimation = None) -> None:
        self.env = env
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.reset()

    def reset(self):
        self.I = 1  # Used for dicounting
        self.actor.reset()
        self.critic.reset()

    def update(self, dlog_pi, state_vector, next_state_vector, reward, done):
        """Update the actor and critic weights using one-step TD learning"""
        # Predict the value of the current and next states
        value_current = self.critic.predict(state_vector)
        value_next = self.critic.predict(next_state_vector)

        # Calculate TD error
        td_error = reward + (0 if done else self.gamma * value_next) - value_current
        
        # Update actor weights using the TD error
        self.actor.w += self.I * self.alpha_actor * td_error * dlog_pi

        # Update critic weights using the TD error
        self.critic.w += self.alpha_critic * td_error * state_vector

        self.I *= self.gamma

    def train(self, num_episodes):
        """Train the agent and log the returns and weights for each episode."""
        self.reset()
        for episode in range(num_episodes):
            self.I = 1
            state, _ = self.env.reset()
            done = False

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

    def _to_vector(self, state):
        state_vector = np.zeros(self.env.observation_space.n)
        state_vector[state] = 1.0
        return state_vector