import numpy as np
import gymnasium as gym

from ..td_learning.q_learning.q_learning import QLearning

class SemiGradientQLearning(QLearning):
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.1, alpha=0.1, feature_dim=8, feature_function=None):
        self.feature_dim = feature_dim
        self.feature_function = feature_function if feature_function is not None else self.default_feature_function
        super().__init__(env, gamma, epsilon, alpha)

    def reset(self):
        super().reset()
        # Initialize to non-zero value to show the divergence   # TODO: But all of then are be proved to be converged? Why?
        self.weights = np.random.rand(self.num_action, self.feature_dim)
        # self.weights = np.ones((self.num_action, self.feature_dim))     
        # self.weights = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0],
        #                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0]])

    def default_feature_function(self, state):
        """
            For baird's counterexample
        """
        phi_s = np.zeros(8)
        if state < 6:
            phi_s[state] = 2
            phi_s[7] = 1
        else:
            phi_s[6] = 1
            phi_s[7] = 2
        return phi_s

    def q_value(self, state, action):
        features = self.feature_function(state)
        return np.dot(self.weights[action], features)

    def update_q_function(self, state, action, reward, next_state):
        features_state = self.feature_function(state)
        # features_next_state = self.feature_function(next_state)

        td_target = reward + self.gamma * np.max([self.q_value(next_state, a) for a in range(self.num_action)])
        td_error = td_target - self.q_value(state, action)

        self.weights[action] += self.alpha * td_error * features_state  # Update the weights corresponding to q

    def improve_policy(self, state):
        tolerance = 1e-8
        q_values = [self.q_value(state, a) for a in range(self.num_action)]
        max_q_value = np.max(q_values)
        best_actions = np.argwhere(np.abs(q_values - max_q_value) <= tolerance).flatten()

        self.policy[state] = np.zeros(self.num_action)
        self.policy[state][best_actions] = 1.0 / len(best_actions)
        
        if self.epsilon > 0:
            self.policy[state] = (1 - self.epsilon) * self.policy[state] + (self.epsilon / self.num_action)