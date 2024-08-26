"""
    Double Q Learning
"""

import numpy as np

from .q_learning import QLearning

class DoubleQLearning(QLearning):
    def reset(self):
        super().reset()
        self.Q1 = np.zeros_like(self.Q)  # Initialize second Q-table
        self.Q2 = np.zeros_like(self.Q)

    def update_q_function(self, state, action, reward, next_state):
        if np.random.rand() < 0.5:
            best_next_action = np.argmax(self.Q1[next_state])
            td_error = reward + self.gamma * self.Q2[next_state][best_next_action] - self.Q1[state][action]
            self.Q1[state][action] += self.alpha * td_error
        else:
            best_next_action = np.argmax(self.Q2[next_state])
            td_error = reward + self.gamma * self.Q1[next_state][best_next_action] - self.Q2[state][action]
            self.Q2[state][action] += self.alpha * td_error

        self.Q = self.Q1 + self.Q2  # Combine the two Q-tables for policy improvement