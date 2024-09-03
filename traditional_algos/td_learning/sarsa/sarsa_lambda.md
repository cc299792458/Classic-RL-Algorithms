Sarsa($\lambda$) takes the concept of N-Step Sarsa to its extreme by utilizing returns from all possible n-step Sarsa updates. This approach adds a new dimension, $\lambda$, which allows performance to be optimized by selecting an appropriate middle value. Additionally, it should be note that sarsa($\lambda$) is an on-policy method that doesn’t wait for N steps to update. Instead, it performs an initial update with available information and then refines it as more information becomes available, ultimately leading to a more complete and accurate update.

TODO: Compare the performance between Sarsa $\lambda$ and N-Step Sarsa.