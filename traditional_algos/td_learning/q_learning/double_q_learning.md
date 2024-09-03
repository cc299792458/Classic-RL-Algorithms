**Double Q-Learning** is a variant of Q-Learning designed to reduce overestimation bias in Q-value estimates. It does this by maintaining two separate Q-value tables and updating them alternately. Intuitively, when you have a single Q-function, the selected action always has the highest Q-value, which is likely to be overestimated. However, with two Q-functions, since they undergo different updates, the action selected by one Q-function may not have a high Q-value in the other, effectively mitigating overestimation.