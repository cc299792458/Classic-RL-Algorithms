**Sarsa** is the most intuitive TD-learning method. The algorithm is named after the sequence of events (State, Action, Reward, next State, next Action) it updates at each step. What we should note is that the next Action is sampled by the policy itself, that's why sarsa is an on-policy algorithm.

Expected Sarsa is a variation of Sarsa. Unlike Sarsa, it does not depend on the specific action chosen by the policy, making expected sarsa an off-policy algorithm.

TODO: Compare the computational efficiency between sarsa and expected sarsa.