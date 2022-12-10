"""Perform a GA of the action space of a CompilerGym environment.

Use the RandomWalk python script to generate the initial candidates, then
use a GA to select the best one.

To make this somewhat tractable, this assumes that the environemnt
takes -1 as an action, and that -1 resets the environment.
TODO -- Support environments that don't do that.
"""

import random
import math

import humanize
from random_walk import run_random_walk
from compiler_gym.datasets import benchmark
from absl import app, flags

from typing import Set, List

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ActionType
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.timer import Timer

from compiler_gym.random_search import random_search

import numpy as np

# Dict for keeping track of various debugging/information counters.
ga_stats = { }

def reset_ga_stats():
    global ga_stats

    ga_stats['valid_candidates_during_compute_score'] = 0
    ga_stats['invalid_candidates_during_compute_score'] = 0
    ga_stats['incomplete_candidates_during_compute_score'] = 0
    ga_stats['candidates_randomly_skipped'] = 0
    ga_stats['candidates_scored'] = 0

reset_ga_stats()
print("Initialized Counters")

if __name__ == "__main__":
    flags.DEFINE_boolean(
        "variable_length_sequences",
        False,
        "Use a crossover algorithm that supports generation of variable length sequences"
    )
    flags.DEFINE_boolean(
        "print_counters",
        False,
        "Print Internal Compile Conters"
    )
    flags.DEFINE_integer(
        "iters",
        12,
        "Min numbrt og iterations"
    )
    flags.DEFINE_integer(
        "generation_size",
        32,
        "number of candidates to track"
    )
    flags.DEFINE_integer(
        "initialization_steps",
        100,
        "number of steps to initialize the initial elements."
    )
    flags.DEFINE_integer(
        "max_cands",
        1000,
        "max number of candidates to add in crossover."
    )
    flags.DEFINE_float(
        "length_preservation_factor",
        0.9,
        "how much to discount different length crossovers. (formual is N^(length difference))"
    )
    flags.DEFINE_float(
        "reduction_factor",
        0.1,
        "what fraction to reduce the number of generated candidates by (0.1 is 10 percent of generated candidates are carried forward to evaluation)"
    )
    flags.DEFINE_boolean(
        "refill",
        False,
        "refill the candidates list using randomly generated candidates if it is too small after each generation"
    )
    flags.DEFINE_integer(
        "expected_candidates",
        1000,
        "How many candidates to take during crossover (in expectation) (only for fixed length --- see reduction factor for variable length)"
    )
    FLAGS = flags.FLAGS

class Candidate:
    def __init__(self, actions):
        self.actions = actions
        self.score = None
        self.failed = True

    def copy(self):
        new_cand = Candidate(self.actions[:])
        new_cand.score = self.score
        new_cand.failed = self.failed

        return new_cand

    def __str__(self):
        return "Actions: " + str(self.actions) + ", score " + str(self.score) + " (failed: " + str(self.failed) + ")"

    def score(self):
        return self.score

    def compute_score(self, env, reward_space_name):
        ga_stats['candidates_scored'] += 1
        env.reset()
        assert len(self.actions) > 0

        print ("Computing score for actions " + str(self.actions))
        reward = -100000 # Largest reward is best
        failed = True # Empty schedules marked as failing (Is this a good idea?)
        try:
            done = False
            for action in self.actions:
                failed = False
                observation, reward, done, info = env.step(action)
                # TODO -- Should we check if we finsihed early?
            if done:
                ga_stats['valid_candidates_during_compute_score'] += 1
                failed = False
            else:
                ga_stats['incomplete_candidates_during_compute_score'] += 1
                failed = True
        except:
            # TODO -- can we do better?
            failed = True
            ga_stats['invalid_candidates_during_compute_score'] += 1

        if failed:
            reward = -10000000
        self.score = reward
        self.failed = failed
        return reward, failed

def mutate(cands: Set[List[ActionType]]):
    new_set = set()
    for c in cands:
        if random.randint(0, 1) == 0:
            # TODO -- pick a better probability of that?
            max_action = max(c.actions) # TODO --- select from all actions not just from max seen.
            new_c = c.copy()
            new_c.actions[random.randint(0, len(c.actions) - 1)] = random.randint(0, max_action)
            new_set.add(new_c)
        new_set.add(c)

    return new_set

# This is a much simpler method that produces a much smaller set for
# fixed-lenght sequences.  The crossover_variable_length method
# has a tendency to explode under long action sequences.
def crossover_fixed_length(cands: Set[List[ActionType]]):
    new_cands = set()
    for c in cands:
        cand_count = len(c.actions) * len(cands) * len(cands)
        fraction_taken = float(FLAGS.expected_candidates) / float(cand_count)

        for c2 in cands:
            for i in range(len(c.actions)):
                should_add = random.random() < fraction_taken
                if not should_add:
                    continue

                new_cand = Candidate(c.actions[:i] + c2.actions[i:])
                new_cands.add(new_cand)

    print("Generated ", len(new_cands), "candidates")
    return new_cands

def crossover_variable_length(cands: Set[List[ActionType]]):
    # Is there a better way to do this?
    # For the CGRA env, these are not fixed length.

    # Try to keep the number generated down a bit:
    naive_number = 0
    for c in cands:
        for c2 in cands:
            naive_number += len(c.actions) * len(c2.actions)

    print ("Naively would add " + str(naive_number) + " candidates")
    new_candidates = set()
    for cand in cands:
        for i in range(len(cand.actions)):
            cand_head = cand.actions[:i]

            # Before we bother iterating, sandwich the range here.
            length_diff = len(cand.actions) - len(cand_head)
            # So the mean length added should be that lenght diff
            # formula is pow(factor, abs(act_lengh - length_diff))
            # Threshold at like 10%
            bound_value = 0.10
            bounds = int(math.log(bound_value, FLAGS.length_preservation_factor))
            for other_cand in cands:
                for j in range(max(length_diff - bounds, 0), min(length_diff + bounds, len(other_cand.actions))):
                    # Don't add everything with certainty. Exponential backoff on size.
                    probability = pow(FLAGS.length_preservation_factor, abs(len(cand.actions) - (len(cand_head) + j)))
                    if (random.random() < probability) and (random.random() < FLAGS.reduction_factor):
                        other_cand_tail = other_cand.actions[j:]

                        new_candidates.add(Candidate(cand_head + other_cand_tail))
                    else:
                        ga_stats['candidates_randomly_skipped'] += 1
    print("Generated " + str(len(new_candidates)) + " new candidates")

    return new_candidates.union(cands)

def get_best(cands, count=1):
    filtered_cands = list(filter(lambda f: (not f.failed), cands))
    print("Got ", len(filtered_cands), "filtered cands from ", len(cands), "original cands")
    sorted_cands = sorted(filtered_cands, key=lambda e: -e.score)
    assert count > 0
    best_score = sorted_cands[0].score

    result = set()
    for cand in sorted_cands[:count]:
        result.add(cand)

    return result, best_score

def compute_individual_fitness(inps):
    cand, env = inps
    cand.compute_score(env, FLAGS.reward)
    return cand

def compute_set_fitness(inps):
    cands, env = inps
    env.reset()
    for c in cands:
        compute_individual_fitness((c, env))

# For splitting up array a for multi core
def split_array(a, n):
    new_arrs = []
    for i in range(n):
        new_arrs.append([])

    ind = 0
    for elem in a:
        new_arrs[ind].append(elem)
        ind += 1
        ind = ind % n

    return new_arrs
    
def compute_fitness(cands, benchmark):
    with env_from_flags(benchmark=benchmark) as env:
        env.reset()
        for cand in cands:
            compute_individual_fitness((cand, env))

    return cands
    

def run_ga(benchmark: benchmark.Benchmark, step_count: int, initial_candidates: Set[List[ActionType]]) -> None:
    # Create an optimizer

    iter_number = 0
    candidate_count = len(initial_candidates)

    current_candidates = initial_candidates

    while iter_number < step_count:
        if (len(current_candidates)) < candidate_count and FLAGS.refill:
            current_candidates += generate_random_candidates(candidate_count - len(current_candidates), benchmark)

        print ("Starting iteration", iter_number, "with", len(current_candidates), "candidates")
        mutations = mutate(current_candidates.copy())
        if FLAGS.variable_length_sequences:
            crossed = crossover_variable_length(mutations)
        else:
            crossed = crossover_fixed_length(mutations)
        fitness = compute_fitness(crossed, benchmark)
        print ("Iter: " + str(iter_number) + " with generation size " + str(len(fitness)))

        current_candidates, best = get_best(fitness, count=candidate_count)
        print ("After iteration " + str(iter_number) + " best score is " + str(best))

        iter_number += 1

    # Get the best one.
    return get_best(current_candidates, count=1)

def generate_random_candidates(number, this_benchmark):
    cands = []
    with env_from_flags(benchmark=this_benchmark) as env:
        env.reset()
        for i in range(number):
            print("Init Candidate ", i)
            env.reset()
            cands.append(Candidate(run_random_walk(env, FLAGS.initialization_steps)))

    return cands

def main(argv):
    print("Starting GA")
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"

    this_benchmark = benchmark_from_flags()
    initial_candidates = set(generate_random_candidates(FLAGS.generation_size, this_benchmark))

    result, best_score = run_ga(this_benchmark, FLAGS.iters, initial_candidates)
    for elt in result:
        # This loop should only go once.
        print ("Result is: ", elt)

        with env_from_flags(benchmark=this_benchmark) as env:
            env.reset()
            for action in elt.actions:
                env.step(action)
            
            print ("Result env was", env, "best score was", best_score)

    if FLAGS.print_counters:
        for field in ga_stats:
            print (field, " = ", ga_stats[field])

if __name__ == "__main__":
    app.run(main)
