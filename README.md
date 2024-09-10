# RL-Explorations

This repository contains implementations, notes, and experiments related to Reinforcement Learning (RL). It includes both foundational concepts and various toy projects exploring RL applications across different domains.

Of course, the first implementation is just a quick naive one of the multi-armed bandit problem based on Sutton's book, which contains action-value methods and the $\epsilon$-greedy strategy.  

Foundational concepts are based off Sutton's Reinforcement Learning book.

## The simulation

The repository contains three python modules which together simulate the dynamics of the multi-armed bandit problem. 

## Mathematical estimates

This sections aims to explore the simulation data in order to present some mathematical estimates and explanations behind some numbers the simulation presents.

## Action-Value methods

### Average reward method

Methods called Action-Value methods are the ones which are based off some estimate in order to act (greedy, $\epsilon$-greedy) and thus maximize some kind of cumulative reward.

The simplest method relies on **averaging** the rewards received.

$$Q_{t}(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$$

in another words, the estimated value of action a at time step $t$ is then:

$$Q_n = \frac{R_1+R_2+\cdots+R_{n-1}}{n-1}$$

where $Q_{t}(a)$ denotes the **estimated value of action $a$ at time step $t$** and $R_{i}$ denotes the reward received after the ith selection of a certain action.

**To continue**

### Gradient Bandit Algorithms

Gradient algorithms are different. They are based off the fact that we can learn a numerical preference and after that set it in a way that we are more likely or not to chose it.

**To continue**






