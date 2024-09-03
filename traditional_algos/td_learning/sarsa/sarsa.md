**Sarsa** is the most intuitive TD-learning method. The algorithm is named after the sequence of events (State, Action, Reward, next State, next Action) it updates at each step. What we should note is that the next Action is sampled by the policy itself, that's why sarsa is an on-policy algorithm.

Expected Sarsa is a variation of Sarsa. Unlike Sarsa, it does not depend on a single sampled action to update the Q-value. Instead, it calculates the expected Q-value based on the current policy, which reduces variance and generally makes the learning process more stable.

TODO: Compare the computational efficiency between sarsa and expected sarsa.