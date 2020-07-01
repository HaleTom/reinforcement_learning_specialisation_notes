# Sample-based Learning Methods

# Week 4: Temporal Difference Learning Methods for Control

### Sarsa

![wk4-sarsa-update.png](wk4-sarsa-update.png)

Just like TD state value evaluation (the recall bit above) required the next state to be known so that it's estimate could be looked up, with Sarsa, the next (state, action) pair must be known for it's value to be looked up.

![wk4-sarsa-windy-gridworld.png](wk4-sarsa-windy-gridworld.png)

Early episodes take many more timesteps to complete than later ones.  Around 7000 steps, the $\epsilon$-greedy policy stops improving.

The policy won't be optimal because it will continue to explore.

Monte Carlo wouldn't be a good fit - many policies don't lead to termination (eg constantly selecting "left").

Sarsa would (somehow!?) learn such policies are bad during the episode, and switch to another one during the episode.

Sarsa is a sample-based algorithm to solve the Bellman equation for action-values.

![wk4-sarsa-algorithm.png](wk4-sarsa-algorithm.png)

### Q-learning

The 2nd Sarsa equation above is actually the Bellman action-value equation. As it has a weighted sum over next-state-actions, it's a bit more like Expected Sarsa than Sarsa.

Q-learning was developed in 1989 and was one of the first online RL algorithms.

Q-learning is a sample-based algorithm to solve the Bellman *optimality* equation for action-values.

Unlike Sarsa, it doesn't need the next action taken - it selects the best possible next action.

![wk4-revisiting-bellman-equations.png](wk4-revisiting-bellman-equations.png)

Sarsa is a sample-based version of policy iteration that which uses Bellman equations for action-values, and is dependent on a fixed policy.

In contrast, because the *optimality* Bellman equation is used in Q-learning, $q_*$ is learned directly, eliminating the need for cycling between policy policy improvement / evaluation steps.

Q-learnng is a sample-based version of value iteration which iteratively applies the Bellman optimality equation, which always improves the action-value function (unless already optimal).

Just like value iteration will converge on the optimal solution, Q-learning will also converge to optimality as long as it continues to explore and samples all regions of the state-action space.

### Q-learning in the windy gridworld

Q-Learning directly learns the optimal policy's action-value function. Perhaps it is more stable - the update target is based on the max of the next action-values, so it only changes when one action is better than another.

Sarsa uses the next action (even if not optimal), as part of its update target.

![wk4-windy-gridworld-parameter-study.png](wk4-windy-gridworld-parameter-study.png)

With a smaller $\alpha = 0.1$, both perform equally (the gradients are parallel, episodes are comleted at the same rate).

### How is Q-learning off-policy?

How can Q-learning be off-policy without using importance sampling?

![wk4-q-learning-comparison-to-sarsa.png](wk4-q-learning-comparison-to-sarsa.png)

Sarsa is on-policy - it bootstraps based on the next action-value, which is determined by the current policy.  Evaluation is on the behaviour policy.

Q-learning is off-policy - it bootstraps on the best next action value, which may be different to the one of the current policy.  Evaluation is on an estimate of the optimal target policy.  It learns about the best action it could take, rather than the action actually taken (which we don't need to wait to know).

Whenever seeing a RL algorithm, a natural question to ask is:  "What are the target and behaviour policies?"

![wk4-no-importance-sampling.png](wk4-no-importance-sampling.png)

Q-learning's target policy is always greedy w.r.t. it's current values.  It's behaviour policy can be anything that continues to visit all state-action pairs during learning (eg, $\epsilon$-greedy).

If Q-learning is off-policy, why don't we see importance sampling ratios?  It's because the agent is estimating action values with a known policy.  It doesn't need importance sampling ratios to correct for the difference in action selection.  The agent can use $Q(S_{t+1}, a')$ and $\pi(a'|S_{t+1})$ to calculate an expected return.  Q-learning uses this technique to learn off-policy.  All non-maximal actions have probability $0$, so the expected return of a state is the same as the maximal action-value from that state.

![wk4-sarsa-vs-Q-episode-rewards.png](wk4-sarsa-vs-Q-episode-rewards.png)

Q-learning doesn't alternate between evaluation and improvements GPI steps, but rather learns the optimal policy directly.

There are some subtleties that make this less desirable in some specific situations.

Q-learning learns an optimal policy.  The optimal policy walks next to the cliff, but an exploratory action can give a hefty -100 reward.

Sarsa learns about its current policy, and takes into account $\epsilon$-greedy action selection, and thus learns a longer but more reliable path further from the cliff.  Sarsa is able to reach the goal more reliably.

Learning off- vs on-policy can make for differences in control, depending on the task.  For online learning, on-policy Sarsa would do better in this case.

### Expected Sarsa

![wk4-expected-sarsa-formula.png](wk4-expected-sarsa-formula.png)

The Bellman equation for an action-value is a sum over next possible states, and the possible next actions in those states.

Sarsa estimates the Bellman expectation by sampling the next state from the environment, and the next action from it's policy.  But the agent already knows its policy, so why bother sampling a next action?  Why not compute the expected value directly?

Expected Sarsa computes a weighted sum of action-values to get the expected value of the next action.

The Expected Sarsa algorihm is the same as Sarsa, with the exception of the update target using the expected estimate of the next action-value rather than a sampled action-value.

There is a huge advantage in calculating the update target: Expected Sarsa has a more stable update target than Sarsa.

In this example, the immediate reward is always $1$.  Both types of Sarsa start out with the true action values of the next state.

Sarsa's update could be in the wrong direction, but eventually, in expectation across multiple updates, the direction is correct.

In contrast, Expected Sarsa's update targets are exactly correct, and don't change the estimated values away from the true values.

Expected Sarsa's update targets have a much lower variance than Sarsa's.

Expected Sarsa's decreased variance comes with a downside: as the number of actions increases, the time taken to compute the expected value increases.  This average needs to be computed every time step.

### Expected Sarsa in the Cliff World

![wk4-expected-vs-sarsa-parameter-study-50000x.png](wk4-expected-vs-sarsa-parameter-study-50000x.png)

Above, $\epsilon = 0.1$ was used in all cases.  100 episodes, averaged over 50,000 independent runs.

Expected Sarsa can use larger $\alpha$ values more effectively because it explicitly averages over the randomness due to its own policy.

The environment is deterministic, so Expected Sarsa's updates are deterministic for a given state and action.

Sarsa's updates can vary significantly depending on the next action.

![wk4-expected-vs-sarsa-parameter-study-100,000x.png](wk4-expected-vs-sarsa-parameter-study-100,000x.png)

Above, after 100,000 episodes both algorithms have learned everytthing that they're going to learn.

Expected Sarsa's long-term behaviour is unaffected by $\alpha$, because in this example updates are deterministic.  The step size only determines how quickly the estimates approach their target values.

As $\alpha$ decreases, Sarsa's long-run performance approaches Expected Sarsa's.

Summary: Expected Sarsa learns more quickly and is more robust to larger step sizes than Sarsa.

Sarsa and Expected Sarsa both approximate the same Bellman action-value equation.

### Expected Sarsa's off-policy learning

Expected Sarsa and Q-learning both use the expectation over their target policies in their update targets, meaning that importance sampling is not required.

![wk4-expected-sarsa-off-policy.png](wk4-expected-sarsa-off-policy.png)

The expectation over actions is calculated independently of the action actually selected in the next state, which can be taken based on a behaviour policy different to $\pi$.

### How Expected Sarsa's generalises Q-learning

If the target policy $\pi$ is greedy, then only the highest value action(s) are considered in the expectation, and the $\displaystyle \sum_{a'}$ will be the $\displaystyle \max_{a'} Q(a', S_{t+1})$, giving the same result as the Q-learning formula.

Q-learning is a special case of Expected Sarsa.

### Summary

![wk4-TD-control-agorithm-comparison.png](wk4-TD-control-agorithm-comparison.png)

The 2nd row lists the update targets.

Sarsa learns a sample-based version of the Bellman action-value equation to learn $q_\pi$.

Expected Sarsa uses the same equation as Sarsa, but the sample is based on an expectation over the next action-values.

Q-learning uses the action-value Bellman optimality equation to learn $q_*$.

Expected Sarsa is both an on-policy and off-policy algorithm and a generalised version of Q-learning.

![wk4-subtleties-with-off-policy-control.png](wk4-subtleties-with-off-policy-control.png)

Sarsa can do better than Q-learning when performance is measured online.  This is because on-policy control methods account for their own exploration.

Expected Sarsa performed better than Sarsa across all step sizes measured because it mitigates the variance due to its own policy by taking the expectation over the next action.

# TODO

* Write out all formulae to test real understanding.

[//]: # (This may be the most platform independent comment)
