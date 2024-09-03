**Use Policy Iteration to solve a 4 by 4 Grid World**

From the Grid World experiment, we can observe that using in-place Policy Iteration is faster than the non-in-place version.

**Use Value Iteration to solve a 4 by 4 Grid World**

Value Iteration is more efficient than Policy Iteration. In this experiment, it solved the problem in approximately 1/30th of the time required by Policy Iteration.

By the way, the names of these algorithms are quite fitting: Value Iteration directly iterates on the value function, while Policy Iteration alternates between iterating on the policy and evaluating it.

**TODO:** Value Iteration and Policy Iteration represent two extremes, but in practice, there can be intermediate approaches, such as performing a few policy evaluation updates. An experiment could be conducted to explore the effects of these intermediate methods.