# Epsilon-Greedy Search

**tldr;**
A greedy search policy that at each step will choose to select an action
randomly with probability ε.

**Authors:**
Facebook AI Research

**Results:**
[results_e0.csv](results_e0.csv),
[results_e25.csv](results_e25.csv).

**Publication:**
<!-- TODO(cummins): Add CompilerGym citation when ready. -->

**CompilerGym version:**
0.1.3

**Open source?**
Yes, MIT licensed. [Source Code](e_greedy.py).

**Did you modify the CompilerGym source code?**
No.

**What parameters does the approach have?**
Probability of selecting a random action *ε*.

**What range of values were considered for the above parameters?**
ε=0 (deterministic) and ε=0.25. Both results are reported.

**Is the policy deterministic?**
Yes if ε=0, else no.

## Description

At each step the agent selects either a greedy policy or an exploration policy.
If greedy, every action is evaluated and the action with the greatest reward
is selected. If exploration, an action is selected randomly. The episode
terminates when the maximum reward attainable by any action is <= 0.


### Experimental Setup

|        | Hardware Specification                        |
| ------ | --------------------------------------------- |
| OS     | Ubuntu 20.04                                  |
| CPU    | Intel Xeon Gold 6230 CPU @ 2.10GHz (80× core) |
| Memory | 754.5 GiB                                     |
