# Week 4: Dynamic Programming

## Learning objectives

### Lesson 1: Policy Evaluation (Prediction)

Policy evaluation is the task of determining the value function for a given policy.

Control is the task of finding a policy which gives as much return as possible (ie, maximises the value function).

Control is the ultimate goal of RL.  Policy evaluation is usually a necessary first step - it provides a metric for improvement.

Dynamic Programming solves both policy evaluation and control problems.

#### Understand the distinction between policy evaluation and control

Dynamic Programming uses the Bellman equations to define converging algorithms for policy evaluation and control.

![wk4-policy-evaluation.png](wk4-policy-evaluation.png)

Policy evaluation is the task of determining $v_\pi$ for a given policy $\pi$.

![wk4-bellman-solving-theory-vs-practice.png](wk4-bellman-solving-theory-vs-practice.png)

The Bellman state equation reduces the problem of finding $v_\pi$ to a system of linear equations, one for each state.

"Strictly better" when comparing policies means that the area under the state-value plot is greater:
* The policy is "as good as or better" (meaning across all states), and:
* There is at least one state where the value is greater and not equal.

The goal of the control task is to modify a policy to produce one which is strictly better.

When this is no longer possible, it means there is no policy strictly better than the current policy, meaning the current policy is an optimal policy.

#### Explain the setting in which dynamic programming can be applied, as well as its limitations

![wk4-dynamic-programming-uses-p.png](wk4-dynamic-programming-uses-p.png)

Classical Dynamic Programming doesn't involve environment interaction (trial-and-error learning), it assumes complete knowledge of the MDP via $p$.

It uses the 4 Bellman equations with $p$ to iteratively work out value functions and optimal policies.

Most other RL methods can be seen as an approximation to Dynamic programming without $p$.  The difference is probably most striking in the temporal-difference based dyna-planning algorithm covered in course 2.

Dynamic programming solves both policy evaluation and control if we have access to the model's dynamics, $p$.

In the iterative policy improvement case, there is an initial or last-iteration ($v_\pi$) and ($v_\pi$ and $\pi$) respectively fed into the top two processes.

#### Outline the iterative policy evaluation algorithm for estimating state values under a given policy

Dynamic programming algorithms come from turning the Bellman equations into update rules.

![wk4-iterative-policy-evaluation-rule.png](wk4-iterative-policy-evaluation-rule.png)

Instead of an equation that holds for the true value function, we have a procedure that we can apply until the equation holds.

A sweep is applying the above update $\forall s \in \mathcal S$.

Begin with an arbitrary initialisation for the value function, called $v_0$.  Each sweep of the update rule produces a better and better approximation of $v_\pi$.

When the update converges (the state-value function no longer changes),  then we have evaluated $v_\pi$ for the current policy $\pi$.  This is because $v_\pi$ is the unique solution to the Bellman equation.  The only way the update could not make a change is if $v_k$ already obeys the Bellman equation.

When $v_{k+1} = v_k$ (equality not assignment) for all states, then replacing both terms with $v_\pi$ gives us the Bellman equation, and therefore the state-value function for policy $\pi$.

#### Apply Iterative policy evaluation to compute value functions

Use two arrays:
1. $V$ stores the current state values
2. $V'$ stores the updated state values
3. Update $V$ with $V'$ after a full sweep.

Two arrays are used so that the new values can be computed from the old, without changing the old values in the process.

It's also possible to use a single array, in which case some updates will use new values.  This is still guaranteed to converge, and will in fact usually converge faster, as it gets to use the newer values sooner.

![wk4-gridworld-01.png](wk4-gridworld-01.png)

Terminal states are top left or bottom right squares, but formally both are the same state.  The value of the terminal state is defined to be $0$.

Each transition has reward $-1$. Any transition which would take the agent off the grid returns it to the same position.

If using the 2 array update, the initial $V'$ values are irrelevant as they will all be updated based on $V$ before use.

![wk4-iterative-policy-evaluation-pseudocode.png](wk4-iterative-policy-evaluation-pseudocode.png)

After convergence, it looks like:

![wk4-gridworld-converged.png](wk4-gridworld-converged.png)

### Lesson 2: Policy Iteration (Control)

#### Understand the policy improvement theorem

The Policy Improvement Theorem tells us that a greedified policy is a strict improvement (unless it was already optimal).

![wk4-policy-improvement-formulae.png](wk4-policy-improvement-formulae.png)

In the `?` or $\pi'$ case, we select the action which is greedy based on the current policy $\pi$, (because we don't yet know $\pi_*$).  We use $v_\pi$ as we don't yet know $v_*$.

The new policy must be different to $\pi$, else...

If greedification doesn't change $\pi$, then $\pi$ was already optimal with respect to its value function, i.e. it was already an optimal policy.

#### Use a value function for a policy to produce a better policy for a given MDP

![wk4-policy-improvement-theorem.png](wk4-policy-improvement-theorem.png)

The new policy must be a strict improvement to unless the policy was already optimal.

#### Outline the policy iteration algorithm for finding the optimal policy

![wk4-policy-iteration-process.png](wk4-policy-iteration-process.png)

Above is the process for Policy Iteration.

Each policy is guaranteed to be an improvement on the last (unless the last was already optimal).

If each policy generated is deterministic, given that there are a finite number of deterministic policies, to the iterative improvement must eventually find an optimal policy.

Instead of calculating the new policy's value function from scratch, we instead start with the value function from the old policy, which is presumably similar to the new policy's, and aids in convergence.

#### Understand “the dance of policy and value”

![wk4-policy-improvement-intuition.png](wk4-policy-improvement-intuition.png)

The goals interact because the two lines are not parallel. An improvement in one causes an improvement in the other.

1. Evaluate $\pi_1$ $ \rightarrow v_{\pi_1}$
2. Use $v_{\pi_1}$ to get $\pi_2$.  (Now $\pi_2$ is greedy w.r.t $v_{\pi_1}$, but $v_{\pi_1}$ no longer reflects $\pi_2$.)
3. Evaluate $\pi_2$ $ \rightarrow v_{\pi_2}$
4. Use $v_{\pi_2}$ to get $\pi_3$.  (Now $\pi_3$ is greedy w.r.t $v_{\pi_2}$, but $v_{\pi_2}$ no longer reflects $\pi_3$.)

This continues until we reach a policy which is greedy with respect to its own value function, namely an optimal policy.

![wk4-iterative-policy-evaluation-cuts-search-space.png](wk4-iterative-policy-evaluation-cuts-search-space.png)

Because each step of improvement moves in the direction of "better", the search space for policies is progressively narrowed.

#### Apply policy iteration to compute optimal policies and optimal value functions

Alternate between evaluation and improvement until the policy doesn't improve.

### Lesson 3: Generalized Policy Iteration

We don't need to strictly alternate between evaluation and improvement. Additionally, we can still retain optimality guarantees.

#### Understand the framework of generalized policy iteration

![wk4-generalised-policy-iteration-intuition.png](wk4-generalised-policy-iteration-intuition.png)

We don't need to iterate all the way to an accurate state-value function, or optimal policy.

Almost all reinforcement learning methods are well described as GPI, i.e. all have:
* Identifiable policies and value functions
* Policy always being improved with respect to the value function
* Value function always being driven toward the value function for the policy

Both processes (evaluation, improvement) stabilize only when a policy has been found that is greedy with respect to its own evaluation function. This implies that the Bellman optimality equation (4.1) holds, and thus that the policy and the value function are optimal.

#### Outline value iteration, an important example of generalized policy iteration

Value Iteration combines (greedy) policy evaluation and improvement into a single step.  The algorithm doesn't reference any particular policy, hence the name "value iteration".

In Value Iteration, we make passes over all the states, running only one step of (greedy) policy evaluation per state, and using the greedy value to update the state-value.

![wk4-value-iteration-algorithm.png](wk4-value-iteration-algorithm.png)

Instead of updating based on a fixed policy, we update based on the greedy policy, ie, the action that maximises the current value estimate.

#### Differentiate between synchronous and asynchronous dynamic programming methods

Synchronous methods sweep the entire state-space in each iteration.  This can be problematic if the state space is large.

Asynchronous methods may update a single state many times before another is updated once.

For convergence, asynchronous methods must continue to update (even if rarely) the values of all states.

Selective updates allow asynchronous methods to propagate more quickly, e.g. by updating the states "nearby" those that have recently changed value.  This is useful when the state space is very large.

Asynchronous algorithms also make it easier to intermix computation with real-time interaction. To solve a given MDP, we can run an iterative DP algorithm at the same time that an agent is actually experiencing the MDP. The agent’s experience can be used to determine the states to which the DP algorithm applies its updates. At the same time, the latest value and policy information from the DP algorithm can guide the agent’s decision making.

To complete even one sweep of a synchronous method requires computation and memory for every state. For some problems, even this much memory and computation is impractical, yet the problem is still potentially solvable because relatively few states occur along optimal solution trajectories. Asynchronous methods and other variations of GPI can be applied in such cases and may find good or optimal policies much faster than synchronous methods can.

#### Describe Monte Carlo as an alternative method for learning a value function

![wk4-monte-carlo-sampling-example.png](wk4-monte-carlo-sampling-example.png)

We could take many different returns under $\pi$ and find their mean, which will eventually converge to the state-value.  We may need to average many returns before the estimate converges, and this would need to be done for every single state.

#### Describe brute force search as an alternative method for searching for an optimal policy

There are a finite number of deterministic policies, and there always exists (at least) one optimal policy.

If the policy space is small enough, then we could evaluate every policy one at a time, and pick the best one with the highest value.

A deterministic policy has one action choice per state, so the total number of deterministic policies is exponential in the number of states:  $|A|^{|\mathcal S|}$.

#### Understand the advantage of Dynamic programming and “bootstrapping” over Monte Carlo and brute-force

A special property of DP methods is that all of them update estimates of the values of states based on estimates of the values of successor states. That is, they update estimates on the basis of other estimates. We call this general idea bootstrapping.  This can be much more efficient that estimating each state's value independently.

Dynamic programming's advantage is that we don't need to treat the evaluation of each state as a separate problem.  We can use the other state-value estimates that we've already worked hard to compute.

Policy Improvement Theorem guarantees that policy iteration will find a sequence of better and better policies, a significant improvement over an exhaustive search.

A DP method takes a number of computational operations that is less than some polynomial function of $|\mathcal S|$ and $|\mathcal A|$.  This is exponentially faster than brute force, i.e. $\mathcal O (|A|^{|\mathcal S|})$.

With each step of policy iteration, the values tend to change less and less.

Monte Carlo methods don't require a model, and also don't bootstrap.

Linear programming methods can also be used to solve MDPs, and in some cases their worst-case convergence guarantees are better than those of DP methods.  But linear programming methods become impractical at a much smaller number of states than do DP methods (by a factor of about 100). For the largest problems, only DP methods are feasible.

### Curse of Dimensionality

The state space grows exponentially as the number of relevant features or state variables increases.

This is not an issue with Dynamic Programming, but rather an inherent complexity of real-world problems.

Various techniques for mitigating this curse exist, and we will continue to deal with it for the remainder of this specialisation.

### Warren Powell: Approximate dynamic programming for fleet management

We want to calculate the marginal value $\bar v$ of a single driver.  This is a linear programming problem, solvable by Gurobi and cplux.

Vector valued action spaces.

For each driver, we drop them out of the system and calculate the system's new value.  The difference in values between the original and driver $a$ dropped value is $\hat v(a)$.

The packages above give the $\hat v$ values for free.

### Quiz learnings

A value function is based on a policy.  If a policy is greedy w.r.t it's own value function, then it is an optimal policy (the value function reflects the policy's always-greediness).

### Reading notes

All the updates done in DP algorithms are called *expected updates* because they are based on an expectation over all possible next states rather than on a sample next state.

Rather than using two arrays for policy evaluation, we usually have the in-place version in mind when we think of DP algorithms.

Value iteration simply turns the Bellman optimality equation (which doesn't reference $\pi$) into an update rule for $v(s) \  \forall s \in \mathcal S$.  In the formula, it uses the $\max_a$ rather than an action's expected value (under $\pi$) used in policy evaluation.  In value iteration, the evaluation doesn't converge to approximate a bellman equation - it only updates $v'(s)$ once for each state.

[//]: # (This may be the most platform independent comment)
