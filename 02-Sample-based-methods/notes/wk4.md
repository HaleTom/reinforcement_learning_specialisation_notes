# Week 4 - Planing, Learning and Control

# Lesson 1: What is a model?

Planning is using a model to improve a policy.

We simulate experience, then update the value function as if those experiences actually occurred.  The improved value estimates allow us to make more informed decisions via policy.

With simulated experience, fewer interactions with the world are required to generate the same policy (assuming the model continues to converge toward the real world).

We will unify the best of bot with the Dyna model.

### Describe what a model is and how they can be used 

Models are used to store information about how the world works. It should produce an approximation of $p$.

Models allow us to predict an outcome of an action without having to actually take the action.

Planning is the process of using a model to improve a policy. We can simulate actions and update the value function without actually taking the action.

We can then make more informed decisions based on the simulation-updated policy with fewer actual samples taken from the real world (greater sample efficiency).

### Classify models as distribution models or sample models

#### Sample model
These produce an actual outcome drawn from some underlying probability distribution.  Eg, sampling what side a coin faces upward after a toss.  Q-learning is a sample-based learning method.

Computationally inexpensive - samples can be easily produced based on a set of rules for producing them.

The probability of each outcome need not be known.

#### Distribution model
These completely specify the likelihood of every possible outcome. Eg the complete $p$ of the MDP in Dynamic Programming.  All the information is there, no decision needs to be made, rather there is simply a search for the best action.

These models contain more information, but can be difficult to specify and become very large.  Every possible outcome must be enumerated with its probability, which can become combinatorially huge.

Distribution models can be used as sample models by drawing a samples based on the distribution of each outcome, but distribution models contain more information than is needed just to generate samples.


## Describe the advantages and disadvantages of sample models and distribution models

Consider rolling 12 dice:

### Sample models

It's easy to generate a random int in [1...6] twelve times.

Joint probabilities are not necessary.

* Require less memory
* Can only estimate an expected outcome by averaging many samples

### Distribution models
Calculating joint probabilities of combinations is much more work: consider all possible outcomes of a set of dice and then the probability of each.  For 12 dice, there are over 2 billion combinations to consider.

With the greater information, we can calculate an expected outcome or quantify the variability in outcomes.

* Can calculate an exact expected outcome by summing over all outcomes weighted by their probabilities.
* Can be used to assess risk accurately

## Explain why sample models can be represented more compactly than distribution models

# Lesson 2: Planning 

Explain how planning is used to improve policies 
Describe random-sample one-step tabular Q-planning 
Lesson 3: Dyna as a formalism for planning 

Recognize that direct RL updates use experience from the environment to improve a policy or value function 
Recognize that planning updates use experience from a model to improve a policy or value function 
Describe how both direct RL and planning updates can be combined through the Dyna architecture 
Describe the Tabular Dyna-Q algorithm 
Identify the direct-RL and planning updates in Tabular Dyna-Q 
Identify the model learning and search control components of Tabular Dyna-Q 
Describe how learning from both direct and simulated experience impacts performance 
Describe how simulated experience can be useful when the model is accurate 
Lesson 4: Dealing with inaccurate models 

Identify ways in which models can be inaccurate 
Explain the effects of planning with an inaccurate model 
Describe how Dyna can plan successfully with a partially inaccurate model 
Explain how model inaccuracies produce another exploration-exploitation trade-off 
Describe how Dyna-Q+ proposes a way to address this trade-off
Lesson 5: Course wrap-up




================= older notes ===================
# Sample-based Learning Methods

# Week 4: Temporal Difference Learning Methods for Control

xyzzy Read summary of Ch7 (skipped - make reading todo.md), then Preview Ch 8 (8.1-8.3 are the assigned reading)
xyzzy read to top p 189.


### Sarsa

![wk4-sarsa-update.png](wk4-sarsa-update.png)

Just like TD state *value* evaluation (`Recall` in the slide) required the next state to be known so that it's estimate could be looked up, with Sarsa, the next (state, action) pair must be known for it's value to be looked up.

![wk4-sarsa-windy-gridworld.png](wk4-sarsa-windy-gridworld.png)

If wind strength is 1, it will move up an additional cell more on the next state compared to zero-wind.

The plot shows the total number of episodes completed after each time step.  Results are averaged over 100 runs.

Epsilon greedy action selection was used, with $r = -1, \gamma\ = 1$ (motivates escape as fast as possible).  Initial values are optimistic, encouraging systematic exploration.

Early episodes take many more timesteps to complete than later ones.  Around 7000 steps, the $\epsilon$-greedy policy stops improving.

Monte Carlo wouldn't be a good fit - many policies don't lead to termination (eg constantly selecting "left").  Sarsa would learn such policies are bad during the episode, and switch to another one during the episode.

Sarsa is a sample-based algorithm to solve the Bellman equation for action-values.

The $\epsilon$-greedy policy won't be optimal because it will continue to explore.

![wk4-sarsa-algorithm.png](wk4-sarsa-algorithm.png)

### Q-learning

Q-learning was developed in 1989 and was one of the first online RL algorithms.

Q-learning is a sample-based algorithm to solve the Bellman *optimality* equation for action-values.

Unlike Sarsa, it doesn't need the next action taken - it selects the best possible next action.

Below, note the only difference from SARSA: the $max$ in the update line.  SARSA update uses the actual next action value, whereas Q-learning uses the best next action value.

![wk4-Q-learning-off-policy.png](wk4-Q-learning-off-policy.png)


![wk4-revisiting-bellman-equations.png](wk4-revisiting-bellman-equations.png)

The 2nd Sarsa equation above is actually the Bellman action-value equation. As it has a weighted sum over next-state-actions, it's a bit more like Expected Sarsa (read on!) than Sarsa.

Sarsa is a sample-based version of policy iteration that which uses Bellman equations for action-values, and is dependent on a fixed policy.

In contrast, because the *optimality* Bellman equation is used in Q-learning, $q_*$ is learned directly, eliminating the need for cycling between policy policy improvement / evaluation steps.

Q-learning is a sample-based version of value iteration which iteratively applies the Bellman optimality equation, which always improves the action-value function (unless already optimal).

Just like value iteration will converge on the optimal solution, Q-learning will also converge to optimality as long as it continues to explore and samples all regions of the state-action space.

### Q-learning in the windy gridworld

Q-Learning directly learns the optimal policy's action-value function. Perhaps it is more stable - the update target is based on the max of the next action-values, so it only changes when one action is better than another.

Sarsa uses the next action (even if not optimal), as part of its update target.

![wk4-windy-gridworld-parameter-study.png](wk4-windy-gridworld-parameter-study.png)

With a $\alpha = 0.5$, both perform similarly.
With a smaller $\alpha = 0.1$, both perform equally eventually (the gradients are parallel, episodes are completed at the same rate), but SARSA take a lot longer to catch up on rate of episode completion..

### How is Q-learning off-policy?

How can Q-learning be off-policy without using importance sampling?

![wk4-q-learning-comparison-to-sarsa.png](wk4-q-learning-comparison-to-sarsa.png)

Sarsa is on-policy - it bootstraps based on the actual next action-value, which is determined by the current policy.  Evaluation is on the behaviour policy.

Q-learning is off-policy - it bootstraps on the estimated best next action value, which may be different to the one of the current policy.  Exploration still happens based on an $\epsilon$-greedy policy for the current action. Evaluation is based on an estimate of the optimal (target) policy for the next state -- it learns from the estimated best action it could take, rather than the action actually taken in the next state (which we don't need to wait to know).

Whenever seeing a RL algorithm, a natural question to ask is:  "What are the target and behaviour policies?"

Below, the right branch is taken deterministically:

![wk4-no-importance-sampling.png](wk4-no-importance-sampling.png)

Q-learning's target policy is always greedy w.r.t. its current values.  It's behaviour policy can be anything that continues to visit all state-action pairs during learning (eg, $\epsilon$-greedy).

If Q-learning is off-policy, why don't we see importance sampling ratios?  It's because the agent is estimating action values with a known policy.  It doesn't need importance sampling ratios to correct for the difference in action selection.  The agent can use $\displaystyle \max_{a'} \ Q(S_{t+1}, a')$ to calculate an expected return.  Q-learning uses this technique to learn off-policy.  All non-maximal actions have probability $0$, so the expected return of a state is the same as the maximal action-value from that state.

![wk4-sarsa-vs-Q-episode-rewards.png](wk4-sarsa-vs-Q-episode-rewards.png)

Q-learning doesn't alternate between evaluation and improvements GPI steps, but rather learns the optimal policy directly.

There are some subtleties that make this less desirable in some specific situations.

Q-learning learns an optimal policy.  The optimal policy walks next to the cliff, but an $\epsilon$-greedy exploratory action will give a hefty -100 reward.

Sarsa learns about its current policy, which takes into account $\epsilon$-greedy action selection, and thus learns a longer but more reliable path further from the cliff.  Sarsa is able to reach the goal more reliably with a better average return.

Learning off- vs on-policy can make for differences in control, depending on the task.  For online learning, on-policy Sarsa would do better in this case.

### Expected Sarsa

![wk4-expected-sarsa-formula.png](wk4-expected-sarsa-formula.png)

Rewritten:

![wk4-expected-sarsa-formula-substituted.png](wk4-expected-sarsa-formula-substituted.png)

The Bellman equation for an action-value is a sum over next possible states, and the possible next actions in those states.

Sarsa estimates the Bellman expectation by sampling the next state from the environment, and the next action from it's policy.  But the policy dictates what action would be taken, so why bother sampling a next action?  Why not compute the expected value directly?

Expected Sarsa computes a weighted sum of next action-values to get the expected value of the next action.

The Expected Sarsa algorithm is the same as Sarsa, with the exception of the TD update target which uses the expected estimate of the next action-value rather than a sampled action-value.

In a deterministic environment, Expected Sarsa's updates will be deterministic for a given state and action, but Sarsa's will depend on the action taken in the next state (which may be a dud exploratory).

There is a huge advantage in calculating the update target: Expected Sarsa has a more stable update target than Sarsa.

Sarsa's TD update target may be in the wrong direction due to a particular sample of next action. But eventually, in expectation across multiple updates, the direction is correct.

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

Above, after 100,000 episodes both algorithms have learned everything that they're going to learn.

Expected Sarsa's long-term behaviour is unaffected by $\alpha$, because in this example updates are deterministic.  The step size only determines how quickly the estimates approach their target values.

![wk4-Sarsa-vs-Expected-vs-Q.png](wk3-Sarsa-vs-Expected-vs-Q.png)

My question on this: [Why does Expected Sarsa always perform better than Q-learning?](https://www.coursera.org/learn/sample-based-learning-methods/discussions/weeks/3/threads/7l9LmThvEeuwbhKYWKHY2Q)

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

# TODO work out week numbers and .md notes numbers.  In coursera, I've completed all of week4 (minus the assignment which I can't)..  **** Also do summary reading from textbooik.

[//]: # (This may be the most platform independent comment)
