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

Policy evaluation is the task of determing $v_\pi$ for a given policy $\pi$.

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

#### Outline the iterative policy evaluation algorithm for estimating state values under a given policy

Dynamic programming algorithms come from turning the Bellman equations into update rules.

![wk4-iterative-policy-evaluation-rule.png](wk4-iterative-policy-evaluation-rule.png)

Instead of an equation that holds for the true value funciton, we have a procedure that we can apply until the equation holds.

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

An action which would take the agent off the grid returns it to the same position.

If using the 2 array update, the $V'$ values are irrelevant as they will all be updated based on $V$ before use.

![wk4-iterative-policy-evaluation-pseudocode.png](wk4-iterative-policy-evaluation-pseudocode.png)

After convergence, it looks like:

![wk4-gridworld-converged.png](wk4-gridworld-converged.png)

### Lesson 2: Policy Iteration (Control)

#### Understand the policy improvement theorem

The Policy Improvement Theorem tells us that a greedified policy is a strict improvement (unless it was already optimal).

![wk4-policy-improvement-formulae.png](wk4-policy-improvement-formulae.png)

In the `?` or $\pi'$ case, we select the action which is greedy based on the current policy $\pi$, (because we don't yet know $\pi_*$).  We use $v_\pi$ as we don't yet know $v_*$.

The new policy must be different to $\pi$, else...

If greedy-fication doesn't change $\pi$, then $\pi$ was already optimal with respect to its value function, i.e. it was already an optimal policy.

#### Use a value function for a policy to produce a better policy for a given MDP

![wk4-policy-improvement-theorem.png](wk4-policy-improvement-theorem.png)

The new policy must be a strict improvement to unless the policy was already optimal.

#### Outline the policy iteration algorithm for finding the optimal policy

![wk4-policy-iteration-process.png](wk4-policy-iteration-process.png)

Above is the process for Policy Iteration.

Each policy is guaranteed to be an improvement on the last (unless the last was already optimal).

If each policy generated is deterministic, given that there are a finite number of deterministic policies, to the iterative improvement must eventually find an optimal policy.

#### Understand “the dance of policy and value”

![wk4-policy-improvement-intuition.png](wk4-policy-improvement-intuition.png)

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

#### Understand the framework of generalized policy iteration

We don't need to strictly alternate between evaluation and improvement. Additionally, we can still retain optimality guarantees.

#### Outline value iteration, an important example of generalized policy iteration

![wk4-generalised-policy-iteration-intuition.png](wk4-generalised-policy-iteration-intuition.png)

We don't need to iterate all the way to an accurate state-value function, or optimal policy.

#### Differentiate between synchronous and asynchronous dynamic programming methods

#### Describe brute force search as an alternative method for searching for an optimal policy

#### Describe Monte Carlo as an alternative method for learning a value function

#### Understand the advantage of Dynamic programming and “bootstrapping” over these alternative strategies for finding the optimal policy

## Value functions


# Quiz
1. \>=
2. F
3. T
4. T
5. F
6. GPI only
7. All
8. F
9. T
10. Async
11. Model
12. -14
13. -15
14. NOT -21.   Incorrect. Try solving the equation 3/4 (x - 1) + 1/4 (-21) = x3/4(x−1)+1/4(−21)=x.


$$ \begin{align}
\end{align}$$

[//]: # (This may be the most platform independent comment)

# Deleteme:
[Textbook webpage](http://incompleteideas.net/sutton/book/the-book.html)

Notes
* [Zubieta's handwritten course notes](https://drive.google.com/file/d/1-QgHag8tGLf5rflYVQixIqhjdW8a-Hdt/view)
* [FrancescoSaverioZuppichini](https://github.com/FrancescoSaverioZuppichini/Reinforcement-Learning-Cheat-Sheet) Reinforcement Learning Cheat Sheet
* [yashbonde](https://yashbonde.github.io/musings.html) - Chapters 2-6, incl exercises
* [micahcarroll](https://micahcarroll.github.io/learning/2018/05/17/sutton-and-barto-rl.html) - Chapters 2 and 13
* [j-kan](https://observablehq.com/@j-kan/reinforcement-learning-notes) - Chapter 3 onwards
* [indoml](https://indoml.com/2018/02/14/study-notes-reinforcement-learning-an-introduction/#lstd) Most chapters, images generated from latex
* [nathandesdouits](https://github.com/nathandesdouits/reinforcement-learning-notes) 1st Ed. Chapter 2 & 3 with numpy code

Textbook solutions

* [iamhectorotero - Chapter 1 to 3](https://github.com/iamhectorotero/rlai-exercises)
* [LyWangPX - Chapter 3 onwards](https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)
* [Weatherwax's 2008 solutions](http://fumblog.um.ac.ir/gallery/839/weatherwax_sutton_solutions_manual.pdf)

Possibly this:
https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e

