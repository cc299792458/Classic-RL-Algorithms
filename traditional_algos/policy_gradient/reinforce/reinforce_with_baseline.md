**REINFORCE with Baseline** is an extension of the REINFORCE algorithm that introduces a baseline to reduce the variance of policy gradient estimates. The baseline is typically the value function, which estimates the expected return from a given state. By subtracting the baseline from the cumulative rewards, the algorithm maintains an unbiased estimate of the gradient while reducing variance. This adjustment leads to more stable and efficient learning, as the policy updates focus more on actions that lead to better-than-expected outcomes, rather than raw rewards.