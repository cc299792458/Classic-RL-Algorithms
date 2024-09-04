**Use Dyna Q and Dyna Q+ to solve Shortcut Maze**

Compared to the experiment in the book, I present a simplified version here (with a much smaller grid map). From this experiment, it can be observed that Dyna Q+ indeed handles improvements in the environment better than Dyna Q. However, certain conditions are still required:

- First, an appropriate $\kappa$ value. Based on the experiment, it shouldn't be too large or too small, and it seems that $\kappa = 0.1$ is a suitable value here. 

- Second, an appropriate epsilon-greedy strategy is needed. I set $\epsilon$ to 0.3 (and I have also tried $\epsilon$ = 0.1), which is a sufficiently large epsilon to ensure exploration.

In addition to these conditions, sufficient time is still required to discover new optimal paths.

However, intuitively, wouldn't a larger $\kappa$ provide more opportunities for exploration? I haven't yet thoroughly explored this question.