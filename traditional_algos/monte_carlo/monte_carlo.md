**Monte Carlo Methods** use sampling to estimate value functions and determine policies when the environment's transition model is not available.

It's important to note that, without the transition model, we must estimate the Q-functions directly instead of the V-functions. This consideration also applies when discussing TD-learning methods.

To accurately estimate action values, it's crucial to maintain exploration; otherwise, some state-action pairs may never be visited.

TODO: Add every-visit monte carlo and test it on grid world or black jack.
TODO: Add off-policy monte carlo method using importance sampling.(both origainal and weighted versions)