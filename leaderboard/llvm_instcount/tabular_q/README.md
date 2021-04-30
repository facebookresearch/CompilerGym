# Tabular Q

**tldr;**

A tabular, online Q-learning algorithm that is trained on each program in the test set.

**Authors:**
JD

**Results:**
1. [Episode length 5, 2000 training episodes](results-H5-N2000.csv).
2. [Episode length 10, 5000 training episodes](results-H10-N5000.csv)


**Publication:**


**CompilerGym version:**
0.1.7

**Open source?**
Yes, MIT licensed. [Source Code](tabular_q_eval.py).

**Did you modify the CompilerGym source code?**
No.

**What parameters does the approach have?**
* Episode length during the Q-table creation *H*.
* Learning rate. *λ*
* Discount fatcor. *γ*
* Actions that are considered by the algorithm. *a*
* Features that are used from the Autophase feature set. *f*
* Number of episodes used during Q-table learning. *N*

**What range of values were considered for the above parameters?**
* H=5, λ=0.1, γ=1.0, 15 selected actions, 3 selected features, N=2000 (short).
* H=10, λ=0.1, γ=1.0, 15 selected actions, 3 selected features, N=5000 (long).

**Is the policy deterministic?**
The policy itself is deterministic after its trained. However the training
process is non-deterministic, so the behavior is different when trained again.

## Description

Tabular Q learning is a standard reinforcement learning technique that computes the
expected accumulated reward from any state action pair, and store them in a table.
Through interaction with the environment, the algorithm improves the estimation by
using step-wise reward and existing entries of the q table.

The implementation is online, thus for every step taken in the environment, the reward
is immediately used to improve the current Q-table.

### Experimental Setup

|        | Hardware Specification                        |
| ------ | --------------------------------------------- |
| OS     | Ubuntu 20.04                                  |
| CPU    | Intel Xeon Gold 6230 CPU @ 2.10GHz (80× core) |
| Memory | 754.5 GiB                                     |

### Experimental Methodology

```sh
# short
$ python tabular_q_eval.py --episodes=2000 --episode_length=5 --learning_rate=0.1 --discount=1 --log_every=0
# long
$ python tabular_q_eval.py --episodes=5000 --episode_length=10 --learning_rate=0.1 --discount=1 --log_every=0
```
