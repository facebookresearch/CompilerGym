from unittest.result import failfast
from compiler_gym.wrappers.datasets import CycleOverBenchmarks
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.wrappers import TimeLimit
import compiler_gym
import model
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import ray
import matplotlib.pyplot as plt
from itertools import islice
import argparse

class RelativePlacementModel(model.Model):
    def __init__(self):
        super().__init__()

    def get_action(observations):
        return super().get_action()

def make_env() -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make(
        "relative-cgra-v0",
        observation_space="RLMapObservations",
        reward_space="II",
        action_space="move",
        benchmark='dfg_10/0' # I think this gets overwritten in the running loop.
    )
    env = TimeLimit(env, max_episode_steps=5)

    return env

def plot_results(rewards):
    plt.bar(range(len(rewards)), rewards)
    plt.ylabel("Reward (higher better)")
    plt.savefig('rewards.png')

def run_agent_on_benchmarks(bmarks):
    with make_env() as env:
        rewards = []
        for i, benchmark in enumerate(bmarks, start=1):
            observation, done = env.reset(benchmark=benchmark), False
            reward = 0
            while not done:
                action = int(agent.compute_action(observation))
                print(type(action))
                observation, reward, done, _ = env.step(action)
            # Just append the last reward, because that is the II. (or a large -ve noting
            # failure)
            rewards.append(reward)
            print ("Exectuted ", i, "th benchmark of", len(bmarks))
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a recreation of the RLMap tool.")
    parser.add_argument('--train', dest='train', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=None)
    args = parser.parse_args()

    with make_env() as env:
        bench = env.datasets['dfg_10']
        train_benchmarks = list(islice(bench.benchmarks(), 650))
        train_benchmarks, val_benchmarks, test_benchmarks = train_benchmarks[:500], train_benchmarks[500:550], train_benchmarks[550:650]

        print("Number of benchmarks for training: ", len(train_benchmarks))
        print("Number of benchmarks for vlaidation: ", len(val_benchmarks))
        print("Number of benchmarks for testing:", len(test_benchmarks))

    def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
        del args
        return CycleOverBenchmarks(make_env(), train_benchmarks)

    tune.register_env("RLMap", make_training_env)

    if args.train:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        analysis = tune.run(
            PPOTrainer,
            checkpoint_at_end=True,
            stop={
                "episodes_total": 500
            },
            config={
                "seed": 0xCC,
                "num_workers": 1,
                "env": "RLMap",
                "rollout_fragment_length": 5,
                "train_batch_size": 5,
                "sgd_minibatch_size": 5,
            },
        )
        best_checkpoint = analysis.get_best_checkpoint(
            metric="episode_reward_mean",
            mode="max",
            trial=analysis.trials[0]
        )
        print("Best checkpoint is '", best_checkpoint, "'")
    if args.test:
        checkpoint = args.test
        agent = PPOTrainer(
            env='RLMap',
            config={
                "num_workers": 1,
                "seed": 0xCC,
                "explore": False
            }
        )

        agent.restore(checkpoint)
        val_rewards = run_agent_on_benchmarks(val_benchmarks)

        plot_results(val_rewards)
    else:
        print("Not testing (use --test to also test)")
