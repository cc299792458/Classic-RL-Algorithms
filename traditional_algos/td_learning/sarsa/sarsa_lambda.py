import numpy as np
import gymnasium as gym

from .sarsa import Sarsa

class SarsaLambda(Sarsa):
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.1, alpha=0.1, lambda_=0.9):
        super().__init__(env, gamma, epsilon, alpha)
        self.lambda_ = lambda_  # Eligibility trace decay rate

    def reset(self):
        super().reset()
        self.eligibility_trace = np.zeros_like(self.Q)  # Reset eligibility traces

    def update_q_function(self, state, action, reward, next_state):
        next_action = np.random.choice(np.arange(self.num_action), p=self.policy[next_state])
        
        # Update eligibility trace for current state-action pair
        self.eligibility_trace[state][action] += 1

        # Update all Q-values and decay eligibility traces
        td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        self.Q += self.alpha * td_error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_

        return next_action