# Sample-based Learning Methods

# Week 4 - Planing, Learning and Control

## Lesson 1: What is a model?

Planning is using a model to improve a policy.

We simulate experience, then update the value function as if those experiences actually occurred.  The improved value estimates allow us to make more informed decisions via policy.

With simulated experience, fewer interactions with the world are required to generate the same policy (assuming the model continues to converge toward the real world).

We will unify the best of both model-based (DP, heuristic search) and model-free (Monte Carlo, TD) methods with the Dyna model.

Primarily, model-based methods rely on planning, whereas model-free methods rely on learning. At the heart of both is the computation of value functions.

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
## Explain why sample models can be represented more compactly than distribution models

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


# Lesson 2: Planning

"Planning with model experience" is learning without needing to interact with the world.

## Explain how planning is used to improve policies

Planning takes a model and policy and produces a better policy.

Take samples from the model, then update the value function as if those events actually happened.  Updating the policy greedily based on the updated value function will improve it also.

Q-planning takes the experience from the model instead of the environment.

Required:
* Model simulating transition dynamics - $p$
* Strategy for sampling relevant state-action pairs

State-action pairs can be sampled uniform randomly, then the model queried for the next state and reward. s

![wk5-random-sample-one-step-tabular-Q-learning](wk5-random-sample-one-step-tabular-Q-learning.png)



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


Quiz
1  dist as sample, both a next possible
2  ABD  no sim to model
3  SASR, env determistic
4  ABCD
5  f
6  CD  env change, stochastic
7  B non-zero back
8  AC both increase explor
9  C one many
10 AC dyna , value iter
