import numpy as np
import gymnasium as gym

from envs.grid_world import GridWorld

class PolicyIteration:
    def __init__(self, env: gym.Env, gamma=1.0, theta=1e-6) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta  # Convergence threshold
        self.num_state = self.env.observation_space.n

        self.reset()

    def reset(self):
        """Reset the policy to a uniform random policy and reset the value function."""
        self.policy = np.ones([self.num_state, self.env.action_space.n]) / self.env.action_space.n
        self.value_function = np.zeros(self.num_state)

    def policy_evaluation(self, inplace=True, print_each_iter=False, print_result=False):
        iteration = 0
        while True:
            iteration += 1
            delta = 0
            
            # If not updating in place, create a new value function to store updates
            if not inplace:
                new_value_function = self.value_function.copy()  # Use copy to avoid modifying the original directly
            
            for state in range(self.num_state):
                if self.env.is_terminal_state(state):
                    continue  # Skip terminal states
                
                v = 0
                # Calculate the expected value for the current state under the policy
                for action, action_prob in enumerate(self.policy[state]):
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        v += action_prob * prob * (reward + self.gamma * self.value_function[next_state])
                
                delta = max(delta, np.abs(v - self.value_function[state]))
                
                if not inplace:
                    new_value_function[state] = v
                else:
                    self.value_function[state] = v
            
            if not inplace:
                self.value_function = new_value_function.copy()

            if print_each_iter:
                print(f"---------- Iteration {iteration} ----------")
                self.print_value_function()
                print("\n")
            
            if delta < self.theta:
                if print_result:
                    print(f"---------- Value Function ----------")
                    self.print_value_function()
                break

    def policy_improvement(self, print_flag=False):
        policy_stable = True
        tolerance = 1e-8  # Define a small tolerance for floating-point comparisons

        for state in range(self.num_state):
            if self.env.is_terminal_state(state):
                continue  # Skip terminal states

            q = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    q[action] += prob * (reward + self.gamma * self.value_function[next_state])
                    
            max_q_value = np.max(q)
            best_actions = np.argwhere(np.abs(q - max_q_value) <= tolerance).flatten()

            new_policy = np.zeros(self.env.action_space.n)
            new_policy[best_actions] = 1.0 / len(best_actions)

            if not np.allclose(self.policy[state], new_policy, atol=tolerance):
                policy_stable = False
                self.policy[state] = new_policy
            
        if print_flag:
            print(f"---------- Policy ----------")
            self.print_policy()
            print("\n")

        return policy_stable

    def iterate(self, eval_print_flag=False, impro_print_flag=False):
        self.reset()
        is_optimal = False
        iteration = 0
        while not is_optimal:
            iteration += 1
            if eval_print_flag or impro_print_flag:
                print(f"---------- Iteration {iteration} ----------")
            self.policy_evaluation(print_result=eval_print_flag)
            is_optimal = self.policy_improvement(print_flag=impro_print_flag)

    def print_value_function(self):
        if hasattr(self.env, '_print_value_function'):
            self.env._print_value_function(self.value_function)
        else:
            print(self.value_function)

    def print_policy(self):
        if hasattr(self.env, '_print_policy'):
            self.env._print_policy(self.policy)
        else:
            print(self.policy)