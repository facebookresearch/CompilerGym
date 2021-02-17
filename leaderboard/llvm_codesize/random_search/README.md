# Random Search

**tldr;**
A pure random policy that records the best result found within a fixed time
budget.

**Authors:**
Facebook AI Research

**Results:**
[results_t60_p125.csv](results_t60_p125.csv),
[results_t1800_p125.csv](results_t1800_p125.csv).

**Publication:**
<!-- TODO(cummins): Add CompilerGym citation when ready. -->

**CompilerGym version:**
0.1.3

**Open source?**
Yes, MIT licensed. [Source Code](random_search.py).

**Did you modify the CompilerGym source code?**
No.

**What parameters does the approach have?**
Search time *t*, patience *p*.

**What range of values were considered for the above parameters?**
Search time is fixed and is indicated by the leaderboard entry name, e.g.
"Random search (t=30)" means a random search for 30 seconds. Eight values for
patience were considered, *n/4*, *n/2*, *3n/4*, *n*, *5n/4*, *3n/2*, *7n/8*, and
*2n*, where *n* is the size of the action space. The patience value was selected
using the `blas-v0` dataset for validation, see appendix below.

**Is the policy deterministic?**
No.

## Description

This approach uses a simple random agent on the action space of the target
program. This is equivalent to running
`python -m compiler_gym.bin.random_search` on the programs in the test set.

The random search operates by selecting actions randomly until a fixed number of
steps (the "patience" of the search) have been evaluated without an improvement
to reward. The search stops after a predetermined amount of search time has
elapsed.

Pseudo-code for this search is:

```c++
float search_with_patience(CompilerEnv env, int patience, int search_time) {
    float best_reward = -INFINITY;
    int end_time = time() + search_time;
    while (time() < end_time) {
        env.reset();
        int p = patience;
        bool done = false;
        do {
            env.step(env.action_space.sample());
            if (env.reward() > best_reward) {
                p = patience;  // Reset patience every time progress is made
                best_reward = env.reward();
            }
        } while (--p && time() < end_time && !env.done())
        // search terminates when patience or search time is exhausted, or
        // terminal state is reached
    }
    return best_reward;
}
```

To reproduce the search, run the [random_search.py](random_search.py) script.


### Tuning the patience parameter

The `--patience_ratio` value was selected by running a random 200 searches on
programs from the `blas-v0` dataset and selecting the value that produced the
best average reward:

```sh
#!/usr/bin/env bash
set -euo pipefail

SEARCH_TIME=30
for patience_ratio in 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 ; do
    echo -e "\nEvaluating --patience_ratio=$patience_ratio"
    logfile="blas_t${SEARCH_TIME}_${patience_ratio}.csv"
    python random_search.py --max_benchmarks=20 --n=10 --dataset=blas-v0 \
        --search_time="${SEARCH_TIME}" --patience_ratio="${PATIENCE_RATIO}" \
        --logfile="$LOGFILE"
    python -m compiler_gym.bin.validate --env=llvm-ic-v0 \
        --reward_aggregation=geomean < "$LOGFILE" | tail -n4
done
```

The patience value that returned the best geomean reward can then be used as the
value for the search on the test set. For example:

```
python random_search.py --search_time=30 --patience_ratio=1.25 --logfile=random_search_t30_p125.csv
```


### Experimental Setup

|        | Hardware Specification                        |
| ------ | --------------------------------------------- |
| OS     | Ubuntu 20.04                                  |
| CPU    | Intel Xeon Gold 6230 CPU @ 2.10GHz (80× core) |
| Memory | 754.5 GiB                                     |
