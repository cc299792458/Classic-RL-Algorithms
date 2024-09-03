To determine a policy without model, we need q functions rather than v functions.

To estimate the action values, one key thing we need to consider is maintaining exploration. Otherwise, some action-state pairs will be never visited.

Without epsilon-greedy, the policy can find the optimal policy but can't thoroughly explore all the actions, so sometimes it can't hanlde multi-optimal actions' situation.
