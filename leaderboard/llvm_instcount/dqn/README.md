# DQN

**tldr;**
A deep q-network that is trained to learn sequences of transformation passes on programs from the test set.

**Authors:**
Patrick Hesse

**Results:**
1. [Episode length 10, 4000 training episodes](results-instcountnorm-H10-N4000.csv).

**Publication:**

**CompilerGym version:**
0.1.9

**Open source?**
Yes. [Source Code](https://github.com/phesse001/compiler-gym-dqn/blob/5a7dc2eec2f144bdabf640266b1667b3da470c79/eval.py).

**Did you modify the CompilerGym source code?**
No.

**What parameters does the approach have?**
* Episode length. *H*
* Learning rate. *λ*
* Discount fatcor. *γ*
* Actions that are considered by the algorithm. *a*
* Features used *f*
* Number of episodes used during learning. *N*
* Ratio of random actions to greedy actions. *ε*
* Rate of decrease of ε. *d*
* Final value of ε. *E*
* Size of memory buffer to store (action, state, reward, new_state) tuple. *s*
* Frequency of target network update. *t*
* The number of time-steps without reward before considering episode done (patience). *p*
* The minimum number of memorys in replay buffer before learning. *l*
* The size of a batch of observations fed through the network *b*
* The number of nodes in a fully connected layer *n*

**What range of values were considered for the above parameters?**
Originally, I tried a much larger set of hyperparameters, something like:
* H=40, λ=0.001, γ=0.99, entire action space, f=InstCountNorm, N=100000, ε=1.0, d=5e-6, E=0.05, s=100000, t=10000, p=5, l=32, b=32, n=512.
But the model was much more unstable, oscillating between ok and bad policies. After some trial and error I eventually decided to scale down the problem by using a subset of the action space with actions that are known to help with code-size reduction and ended up using this set of hyperparameters:
* H=10, λ=0.001, γ=0.9, 15 selected actions, f=InstCountNorm, N=4000, ε=1.0, d=5e-5, E=0.05, s=100000, t=500, p=5, l=32, b=32, n=128.

**Is the policy deterministic?**
The policy itself is deterministic after its trained. However the initialization of network parameters is non-deterministic, so the behavior is different when trained again.

## Description

Deep Q-learning is a standard reinforcement learning algorithm that uses a neural
network to approximate Q-value iteration. As the agent interacts with it's environment,
transitions of state, action, reward, new state, and done are stored in a buffer and
sampled randomly to feed to the Q-network for learning. The Q-network predicts the
expected cumulative reward of taking an action in a state and updates the network
parameters with respect to the huber loss between the predicted Q-values and the
more stable target predicted Q-values.

This algorithm learns from data collected online by the agent, but the data are stored
in a replay buffer and sampled randomly to remove sequential correlations.

The decision-making is done off-policy, meaning that it's actions are dictated not
only by the policy but also by some randomness to encourage exploration of the
environment.

### Experimental Setup

|        | Hardware Specification                        |
| ------ | --------------------------------------------- |
| OS     | Ubuntu 20.04                                  |
| CPU    | Ryzen 5 3600 CPU @ 3.60GHz (6× core)          |
| Memory | 16 GiB                                        |

### Experimental Methodology

```sh
# this will train the network parameters, which we will load later for evaluation
# since this is not for generalization, we will average the train time over the 23 benchmarks and add it to the geomean time
$ time python train.py --episodes=4000 --episode_length=10 --fc_dim=128 --patience=4
$ python eval.py --epsilon=0 --episode_length=10 --fc_dim=128 --patience=4
```
