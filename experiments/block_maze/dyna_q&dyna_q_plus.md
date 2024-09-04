**Use Dyna Q and Dyna Q+ to solve Blocked Maze**

In this experiment, it can be observed that Dyna Q+ consistently outperforms Dyna Q when $\kappa$ is set within a broad range. Interestingly, setting $\kappa$ to values such as 1e-0 and 1e-4 achieved better results, while the intermediate values yielded slightly weaker performance. The specific reason for this phenomenon has not yet been fully investigated.

Additionally, I initially made an error by sampling from all states instead of just the encountered states during the planning part, which caused Dyna Q+ to perform worse than Dyna Q. The underlying reason for this issue is still under exploration. A brief analysis suggests that in the early stages, choosing only the visited states is similar to on-policy sampling, whereas sampling from all states is akin to exhaustive search, with the latter being far less efficient than the former. However, in the later stages, the difference between the two approaches should be minimal. Therefore, this might not be the full explanation for the observed issue.

An interesting situation in the Blocked Maze environment is that the agent may find itself at the position of a new wall just as the walls are about to be updated. To handle this, I implemented a delay in updating the walls until the agent moves out of the new wall's position.

TODO: Exercise 8.4 in the RL Book.