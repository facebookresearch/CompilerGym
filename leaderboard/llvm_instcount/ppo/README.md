<!-- To submit a leaderboard entry please fill in this document follow the
instructions in the CONTRIBUTING.md document to file a pull request. -->
# Proximal Policy Optimization with Guided Random Search

**tldr;**
Proximal Policy Optimization (PPO) followed by guided search using the action 
probabilities of the PPO-Model

**Authors:** Nicolas Fröhlich, Robin Schmöcker, Yannik Mahlau
<!-- A comma separated list of authors. -->


**Publication:** Not Available
<!-- A link to a publication, if applicable. -->


**Results:** Geometric Mean: 1.070, [results](/results.csv)
<!-- Add one or more links to CSV files containing the raw results. -->


**CompilerGym version:** 0.2.1
<!-- You may print the version of CompilerGym that is installed from the command
line by running:

    python -c 'import compiler_gym; print(compiler_gym.__version__)'
-->



**Is the approach Open Source?:** Yes
<!-- Whether you have released the source code of your approach, yes/no. If
yes, please state the license. -->
The source code is available as Open-Source: 
https://github.com/xtremey/ppo_compiler_gym

**Did you modify the CompilerGym source code?:** No (apart from state space wrappers)
<!-- Whether you made any substantive changes to the CompilerGym source code,
e.g. to optimize the implementation or change the environment dynamics. yes/no.
If yes, please briefly summarize the modifications. -->

**What parameters does the approach have?**
<!-- A description of any tuning parameters. -->
| Hyperparameter            	| Value   	|
|---------------------------	|---------	|
| Number of Epochs          	| 80      	|
| Epsilon Clip              	| 0.1     	|
| Mean Square Error Factor  	| 0.5     	|
| Entropy Factor            	| 0.01    	|
| Learning Rate             	| 0.0005  	|
| Trajectories until Update 	| 20      	|
| Hidden Layer Size         	| 128     	|
| Activation Function       	| TanH    	|
| Number of Layers          	| 4       	|
| Shared Parameter Layers   	| First 3 	|
| Optimizer                 	| Adam    	|

**What range of values were considered for the above parameters?**
<!-- Briefly describe the ranges of values that were considered for each
parameter, and the metrics and dataset used to select from the values. -->
We experimented a little bit with the hyperparameters of PPO, but the results did not 
change drastically. Therefore, we did not perform any Hyperparameter Optimization.

**Is the policy deterministic?:** No
<!-- Whether the (state, action) policy is deterministic, yes/no. -->

## Description

<!-- A brief summary of the approach. Please try to be sufficiently descriptive
such that someone could replicate your approach. Insert links to external sites,
publications, images, or other pages where relevant. -->

Our Model uses the Proximal Policy Optimization (PPO) Architecture: 
https://arxiv.org/abs/1707.06347

We used a wrapper to extend the state space such that the number of remaining steps is 
an additional entry in the state space (as a number, not one hot encoded). During 
training we limited the number of steps per episode to 200.

In a second step we use the action probabilities of the model to perform a guided 
random search (also for 200 steps). We limited the search time to one minute for each 
environment.

In a third step we optimized the best trajectory found during random search by taking 500 
additional steps using the models action probabilities. This did not yield improvement
for all environments, but sometimes improved solution a little with basically no
computational cost. Therefore, the maximum possible length of a trajectory is 700. 
However, most trajectories are much shorter.

We excluded the Ghostscript benchmark during training since it took a lot of computation 
and presented itself as a bottleneck. Additionally, we excluded the random search and additional 
steps for this benchmark since it did not yield any improvement and drastically increased the mean
walltime.


### Credit
Credit to nikhilbarhate99 
(https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py).
Parts of the rollout buffer and the update method are taken from this repo.