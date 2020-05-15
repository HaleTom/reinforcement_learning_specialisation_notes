# Fundamentals of Reinforcement Learning
Week 1, Univerity of Alberta

[Textbook webpage](http://incompleteideas.net/sutton/book/the-book.html)

Notes
* [Zubieta's handwritten course notes](https://drive.google.com/file/d/1-QgHag8tGLf5rflYVQixIqhjdW8a-Hdt/view)
* [FrancescoSaverioZuppichini](https://github.com/FrancescoSaverioZuppichini/Reinforcement-Learning-Cheat-Sheet) Reinforcement Learning Cheat Sheet
* [micahcarroll](https://micahcarroll.github.io/learning/2018/05/17/sutton-and-barto-rl.html) - Chapters 2 and 13
* [j-kan](https://observablehq.com/@j-kan/reinforcement-learning-notes) - Chapter 3 onwards
* [indoml](https://indoml.com/2018/02/14/study-notes-reinforcement-learning-an-introduction/#lstd) Most chapters, images generated from latex
* [nathandesdouits](https://github.com/nathandesdouits/reinforcement-learning-notes) 1st Ed. Chapter 2 & 3 with numpy code

Possibly this:
https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e

## Course Introduction

Supervised learning: Labelled examples determine correctness of answer.

Reinforcement learning: Reward gives the agent some idea of how good or bad recent actions were.  Reward says what good behaviour looks like, not exactly how to solve the problem.

Unsupervised learning:  Extracting the underlying structure from data, the representation of the data.  Can assist in RL.

RL is about learning while interacting with an ever-changing world, refining behaviour as they go.

Allowing for changing goals and integrating recent experience is important.

The defining feature and difficulty in RL is learning "online" (real-time) rather than just learning from data.

Weekly new algorithm improvements - breakneck pace.

The fundamentals of RL date back to Pavlov's drooling dogs.

At the heart of any RL system are ideas from one or two decades before now. DQN combines neural networks, Q learning, and experience replay

### Instructors

Adam is well known for his work on predictive knowledge for reinforcement learning.

Martha's algorithm contributions to reinforcement learning are too many to list here. She has developed several new off-policy learning algorithms, new approaches to policy gradient, and dozens of impressive contributions to representation learning

RL is a generic approach to automated decision making.  It will likely really take off in industrial control systems.

### Specialisation roadmap

Course will closely follow Sutton & Barto's "Reinforcement Learning: An Introduction"

The RL book and this specialization adhere to a simple principle, introduce each idea in the simplest setting it arises

## Lesson 1: The K-Armed Bandit Problem

The agent learns it's own training data by interacting with the world, through trial and error.

Motivating Problem: Decision making under uncertainty with K-armed bandits.

In the k-armed bandit problem, we have a decision-maker or agent, who chooses between k different actions, and receives a reward based on the action it chooses.  Choosing an action yields an unknown reward.

### Sequential decision making with evaluative feedback

In the stationary version, each of our actions has an _expected reward_ given that that action is selected, called the _action value_.

$$\begin{align}
q_\star(a) & \doteq \mathbb{E}[R_t \mid A_t = a] \quad  \forall a \in \{1, …, k\} \\
  & = \sum_{r} p(r \mid a) \ r
\end{align}
$$

The conditional expectation is the sum of all possible rewards.

In the continuous case, switch the summation to an integral.

The goal of the agent is to maximise the expected reward by picking the corresponding action.  

![wk1-calculating-q-star.png](wk1-calculating-q-star.png)

Above, 1, 2, & 3 are the different treatments given, and the numbers are improvements in blood pressure.

### Learning action values

$q \star (a)$ is not known to the agent, so we must estimate it.

One way is via the Sample-Average method:

$$ Q_t(a) \doteq \frac{\text{sum of rewards when action `a` taken before time } t}{\text{number of times action `a` was taken prior to } t} $$

Sample-average is set to $0$ if the action hasn't yet been taken.

The greedy action is the one that currently has the largest estimated value.  The agent is exploiting its current knowledge for immediate return.

Alternatively, the agent could perform a non-greedy action, sacrificing immediate reward hoping to gain more information about the value of the other actions, and find a overall better action.

### Estimating action values incrementally

The incremental update rule can be written recursively:

$$\begin{align}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^n R_i \\
& = \frac{1}{n} \Bigg(R_n + \sum_i^{n-1} R_i\Bigg) \\
& = \frac{1}{n} \Bigg(R_n + (n-1) \frac{1}{(n-1)}\sum_i^{n-1} R_i\Bigg) \\
&= \frac{1}{n} \Big(R_n + (n-1) Q_{n}\Big) \\
&= Q_n + \frac{1}{n} \Big[R_n - Q_{n}\Big]
\end{align}$$









================

$r(s,a) = \mathbb{E}\Big[ R_t \mid S_{t-1}=s, A_{t-1}=a \Big] =
  \sum_{r \in \mathcal{R}} r \cdot p(r | s,a) =
  \sum_{r \in \mathcal{R}} r \cdot \sum_{s' \in \mathcal{S}} p(s', r | s,a)
$
