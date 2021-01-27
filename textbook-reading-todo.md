# [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) TODO

## Chapter 6 - 

### 6.7 Maximization Bias and Double Learning

## Chapter 7 - n-step Bootstrapping
Coursera skipped this chapter entirely.

Describes the spectrum between MC methods at one end (no bootstrapping), and TD-0 for better fit to problems.

All n-step methods involve a delay of n time steps before updating, as only then are all the required future events known.

> Another way of looking at the benefits of n-step methods is that they free you from
the tyranny of the time step. With one-step TD methods the same time step determines
how often the action can be changed and the time interval over which bootstrapping
is done. In many applications one wants to be able to update the action very fast to
take into account anything that has changed, but bootstrapping works best if it is over a
length of time in which a significant and recognizable state change has occurred.

Computational issues are fixed in Ch 12.

## Chapter 8 - Planning and Learning with Tabular Methods

Integration of model-based and model-free methods.

Coursera only covered 8.1 - 8.3.

