# **Classic-RL-Algorithms**

This repository provides a comprehensive collection of classic reinforcement learning algorithms. It features simple implementations of **traditional methods** such as **dynamic programming, monte carlo methods, and td-learning**, including both **tabular approaches and those using function approximation.** All classic algorithms here have been tested using environments from [Sutton and Barto's book: "Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book-2nd.html). Additionally, this repository will include **algorithms implemented with deep neural networks**, covering both **Q-function-based methods and policy gradient methods**. Some of these implementations have borrowed code from the homworks of [Berkeley's CS 285 course on Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/). 

What's more, this repository includes notes for each algorithm to share my **findings from the experiments**. For beginners, these notes provide insights into why certain algorithms are used, their advantages and disadvantages, and potential issues you might encounter during implementation. Additionally, in the **Blog** folder, you can find some of my opinions on reinforcement learning. 

This repository is still in development, with the traditional algorithms section largely complete. Currently, I am working on implementing the deep reinforcement learning algorithms. In the future, I plan to add more algorithms, including those related to model-based RL, visual RL, and real-world RL.

You are welcome to provide suggestions or raise an issue if you have any questions.

**Reinforcement learning is fascinating, and I wish this repository will be helpful to others who share the same interest!**

## The list of traditional ones is:
- [Epsilon Greedy](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/epsilon_greedy)
- [Dynamic Programming](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/dynamic_programming)
  - [Policy Iteration](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/dynamic_programming/policy_iteration) 
  - [Value Iteration](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/dynamic_programming/value_iteration)
- [Monte Carlo Methods](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/monte_carlo) 
  - [Monte Carlo](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/monte_carlo/monte_carlo.py)
- [Temporal Difference Learning](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/td_learning) 
  - [Sarsa](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/td_learning/sarsa): [Sarsa](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/td_learning/sarsa/sarsa.py), [Expected-Sarsa](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/td_learning/sarsa/sarsa.py), [N-Step Sarsa](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/td_learning/sarsa/n_step_sarsa.py), [Sarsa Lambda](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/td_learning/sarsa/sarsa_lambda.py)
  - [Q-Learning](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/td_learning/q_learning): [Q-Learning](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/td_learning/q_learning/q_learning.py), [Double Q-Learning](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/td_learning/q_learning/double_q_learning.py)
- [Gradient-Based Value Methods](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/gradient_based_value_methods)
  - [Gradient Monte Carlo](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/gradient_based_value_methods/gradient_monte_carlo.py)
  - [Semi-Gradient Q-Learning](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/gradient_based_value_methods/semi_gradient_q_learning.py)
- [Policy Gradient Methods](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/policy_gradient):
  - [REINFORCE](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/policy_gradient/reinforce): [REINFORCE](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/policy_gradient/reinforce/reinforce.py), [REINFORCE with Baseline](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/policy_gradient/reinforce/reinforce_with_baseline.py) 
  - [Actor-Critic](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/policy_gradient/actor_critic)
- [Planning and Learning Methods](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/planing_and_learning_methods)
  - [Dyna-Q](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/planing_and_learning_methods/dyna_q.py), [Dyna-Q+](https://github.com/cc299792458/Classic-RL-Algorithms/blob/main/traditional_algos/planing_and_learning_methods/dyna_q_plus.py)
    
