# Sample-based Learning Methods

# Week 3: Temporal Difference Learning Methods for Prediction

TD learning is the most central and novel idea in RL.

The special cases of TD methods introduced in chapter 6 should rightly be called one-step, tabular, model-free TD methods.

We can classify TD control methods according to whether they deal with this complication by using an on-policy or off-policy approach:

* Sarsa is an on-policy method.
* Q-learning is an off-policy method.  Expected Sarsa is also an off-policy method as presented here.

## Introduction to Temporal Difference Learning

### Temporal Difference Learning

![wk3-incremental-update-formula.png](wk3-incremental-update-formula.png)

The 2nd formula can be used to for an online update to the previous value of a state to form a Monte Carlo estimate without saving a list of returns to average.

Note the LHS of the red box is drawn over the `[` (right square bracket)

But $G_t$ is the return from a full episode, meaning that we can't (yet) learn incrementally inside an episode.

![wk3-recursive-state-value-formula.png](wk3-recursive-state-value-formula.png)

Above we replace $G_{t+1}$ with a recursively-defined value from $v$.

The value of the next state is a stand-in for the value of the return until the end of the episode.  We don't need to wait until the end of the episode, but we do need to wait until the next step.

### Temporal Difference Error

![wk3-TD-update.png](wk3-TD-update.png)

The highlighted terms are called the *TD error*, or $\delta_t$.  The first two of these are called the *TD target*.

TD updates its estimate of one state towards its own estimate of the next state.  As the estimated value of the next state improves, so does our TD target.

### TD(0) algorithm

![wk3-1-step-TD.png](wk3-1-step-TD.png)

Assume we are are looking from the perspective of state $S_{t+1}$. We've stored the state of the previous time step to make our update to it (after we have observed the reward associated with the action taken within it).  We discount the current state's value by $\gamma$, add the observed reward and then treat that sum as the observed value of the previous state.

In DP, we used the $p$ and $\pi$ to update based on all possible future states (and their transition returns).  Here we only use the next observed state and observed return.

![wk3-tabular-TD(0)-algorithm.png](wk3-tabular-TD0-algorithm.png)

### Richard S. Sutton - Temporal Learning

TD learning is specialised for prediction learning, making it the most important thing for AI in the century so far.

Methods that scale with computation are the future of AI.

Supervised learning and model-free RL methods are only weakly scalable.

Prediction learning is the most thoroughly *scalable* model-free learning.

Training sets and an objective are not required, just waiting for the outcome.

TD learning is a method specialised for learning to predict.

* Widely used in RL to predict future reward (value functions)
* Used in Sarsa, Q-learning, TD($\lambda$), Deep Q-learning, TD-Gammon, actor-critic methods, Samuel's Checker Player
  * (but not AlphaGo, helicopter autopilots, pure policy-based methods...)
* Seems to be how brain reward signals work
* Can predict any signal, not just reward

A series of predictions are made (after every time step), and we gradually find out what the correct answers were for the predictions when what actually happens is observed.

In prediction learning, as a parallel to supervised learning, the supervisor / labeller (telling the correct answer) is the environment. So prediction learning is the unsupervised supervised learning.

![wk3-Richard-Sutton.png](wk3-Richard-Sutton.png)

TD learning is learning from another, later, learned prediction.
* i.e., learning a guess from a guess
* The TD error is the difference between two predictions, the *temporal difference*.
* Otherwise TD is the same as supervised learning, back-propagating the error.

TD is only relevant for multi-step prediction learning (with information possibly revealed on each step).

Supervised learning is not prediction learning in that the label tells the correct prediction, rather than wait-and-see.

If the multi-step is a single step, then prediction reduces to the traditional supervised learning problem.

It's not possible to compose multi-step predictions from single-step predictions in practice: long-term predictions are exponentially complex and amplify small errors in one-step predictions.

We can see signals in the brain that correspond to the TD error.  Dopamine is the carrier in mammals, octopamine in bees.

Temporal difference learning is an important topic in both AI and in neuroscience and psychology (where it models and predicting behaviour).

### Advantages of TD learning

Like DP, TD can bootstrap.  Like MC, TD can learn directly from experience.

#### Understand the benefits of learning online with TD

Unlike with MC, we don't need to wait until the end of an episode - we can update the value for the previous state as soon as we transition and observe the reward and next state.

Unlike DP, a model of the environment is is not required

Unlike both, the updates are online and fully incremental. TD updates as soon as it receives new information, making it useful in real-time settings.

TD asymptotically converges to the correct predictions, and usually does so faster than MC methods.

### Empirical differences

With MC, the final return propagates back all the way to the beginning (discounted by $\gamma$).

With TD, the update is only based on the difference between the estimated previous and discounted next-state values, and the return.

![wk3-RMS-error-TD-vs-MC.png](wk3-RMS-error-TD-vs-MC.png)

TD learned quicker and achieved a better final error.

We could use a decaying $\alpha$ to achieve even better results.

### Barto and Sutton: More history of RL

Harry Klopf created a contract (not grant) at UMass to study if his theory of The Hedonistic Neuron made any sense.  Barto was hired as a post-doc. Michael Harvard/Harlem? said Harry required them to bring Rich Sutton on board.  Andy says he is risk adverse, and Sutton is risk-seeking.  Sutton says Barto is a contrare, wanting to go in the opposite direction to others. Barto says he went orthogonally in a sparsely covered field so he didn't need to keep up with the literature.  Barto left his tenure track position to be a post-doc, which Sutton says was risk taking :)

In the 60s, the first wave of ML turned into supervised learning.  In the 50s, there was interest in reward.  Barto says that even very astute people still confuse error correction and trial and error learning.

In the 80s, the advances in RL got overshadowed by the advances in supervised learning.  Also people thought that RL was too similar to behaviourism and threw the baby out with the bathwater.

Sutton hints at much behaviourist learning being lost, and a superpower coming from his understanding of the behaviourists, including the idea of TD learning.

Barto talks about intrinsic motivation in terms of research, rather than the nomadic researchers that flit from one hot topic to another.  Utility was not the driving force of the research, even though the utility turned out to be massive.

# ==========================


Lesson 2: Advantages of TD

Module 3: Temporal Difference Learning Methods for Control

Lesson 1: TD for Control

Explain how generalized policy iteration can be used with TD to find improved policies

Describe the Sarsa Control algorithm

Understand how the Sarsa control algorithm operates in an example MDP

Analyse the performance of a learning algorithm

Lesson 2: Off-policy TD Control: Q-learning

Describe the Q-learning algorithm

Explain the relationship between q-learning and the Bellman optimality equations.

Apply q-learning to an MDP to find the optimal policy

Understand how Q-learning performs in an example MDP

Understand the differences between Q-learning and Sarsa

Understand how Q-learning can be off-policy without using importance sampling

Describe how the on-policy nature of SARSA and the off-policy nature of Q-learning affect their relative performance

Lesson 3: Expected Sarsa

Describe the Expected Sarsa algorithm

Describe Expected Sarsa’s behaviour in an example MDP

Understand how Expected Sarsa compares to Sarsa control

Understand how Expected Sarsa can do off-policy learning without using importance sampling

Explain how Expected Sarsa generalizes Q-learning

Module 4: Planning, Learning & Acting

Lesson 1: What is a model?

Describe what a model is and how they can be used

Classify models as distribution models or sample models

Identify when to use a distribution model or sample model

Describe the advantages and disadvantages of sample models and distribution models

Explain why sample models can be represented more compactly than distribution models

Lesson 2: Planning

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

# XXXXXXXXXXX

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

* [yashbonde](https://yashbonde.github.io/musings.html) - Chapters 2-6, incl exercises
* [iamhectorotero - Chapter 1 to 3](https://github.com/iamhectorotero/rlai-exercises)
* [LyWangPX - Chapter 3 onwards](https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)
* [Weatherwax's 2008 solutions](http://fumblog.um.ac.ir/gallery/839/weatherwax_sutton_solutions_manual.pdf)

Possibly this:
https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e

[//]: # (This may be the most platform independent comment)
