# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a CompilerGym environment with text interface controls.

.. code-block::

    $ python -m compiler_gym.bin.manual_env --env=<env> [--benchmark=<name>] [--observation=<space>] [--reward=<space>]

The benchmark to use can be specified using :code:`--benchmark=<name>`. If not
provided, you be presented with a list of benchmarks to choose from on launch.
Select :code:`random` to select a benchmark randomly.
"""
import sys
from typing import Optional

from absl import app, flags
import cmd
import readline
import random

import compiler_gym.util.flags.ls_benchmark  # Flag definition.
from compiler_gym.datasets.dataset import require
from compiler_gym.envs import CompilerEnv
from compiler_gym.util import user_input
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.tabulate import tabulate
from compiler_gym.util.timer import Timer

FLAGS = flags.FLAGS


class ActionHistoryElement:
    """The compiler gym shell records a list of actions taken. This class represent those elements."""
    def __init__(self,
        action_name,
        action_index,
        eager_observation,
        eager_reward,
        done,
        info
    ):
        """Arguments are the returns from env.step"""
        self.action_name = action_name
        self.action_index = action_index
        self.eager_observation = eager_observation
        self.eager_reward = eager_reward
        self.done = done
        self.info = info

    def has_no_effect(self):
        """Determine if the service thinks this action had no effect"""
        return self.info.get("action_had_no_effect")

    def has_effect(self):
        """Determine if the service thinks this action had an effect"""
        return not self.has_no_effect()


class CompilerGymShell(cmd.Cmd):
    """Run an environment manually.
    The manual environment allows the user to step through the environment,
    selection observations, rewards, and actions to run as they see fit. This is
    useful for debugging.
    """
    
    init = "Welcome to the CompilerGym manual environment!"

    def __init__(self, env: CompilerEnv):
        """Initialise with an environment.
        :param env: The environment to run.
        """
        super().__init__()

        self.env = env

        self.init_benchmarks()

        # Get the benchmarks
        self.benchmarks = sorted(self.env.benchmarks)
        # Strip default benchmark:// protocol.
        for i, benchmark in enumerate(self.benchmarks):
            if benchmark.startswith("benchmark://"):
                self.benchmarks[i] = benchmark[len("benchmark://") :]

        # Get the observations
        self.observations = sorted(self.env.observation.spaces.keys())
        # Get the rewards
        self.rewards = sorted(self.env.reward.spaces.keys())

        # Set up the stack.
        self.stack = []

        self.set_prompt()

    def preloop(self):
        self.old_completer_delims = readline.get_completer_delims()
        readline.set_completer_delims(" \t\n")

    def postloop(self):
        readline.set_completer_delims(self.old_completer_delims)
        # Clear the stack
        self.stack.clear()
        self.env.close();
        self.env = None

    def init_benchmarks(self):
        """Initialise the set of benchmarks"""
        # Get the benchmarks
        self.benchmarks = sorted(self.env.benchmarks)
        # Strip default benchmark:// protocol.
        for i, benchmark in enumerate(self.benchmarks):
            if benchmark.startswith("benchmark://"):
                self.benchmarks[i] = benchmark[len("benchmark://") :]


    def set_prompt(self):
        """Set the prompt - shows the benchmark name"""
        if self.env.benchmark:
            bname = self.env.benchmark
            if bname.startswith("benchmark://"):
                bname = bname[len("benchmark://") :]
        else:
            bname = "NO-BENCHMARK"
        self.prompt = f"compilergym:{bname}> "

    def simple_complete(self, text, options):
        """Return a list of options that match the text prefix"""
        if text:
            return [opt for opt in options if opt.startswith(text)]
        else:
            return options


    def get_datasets(self):
        """Get the list of available datasets"""
        return sorted([k for k in self.env.available_datasets])

    def do_list_datasets(self, arg):
        """List all of the available datasets"""
        print(", ".join(self.get_datasets()))

    def complete_require_dataset(self, text, line, begidx, endidx):
        """Complete the require_benchmark argument"""
        return self.simple_complete(text, self.get_datasets())

    def do_require_dataset(self, arg):
        """Require dataset
            The argument is the name of the dataset to require.
        """
        if self.get_datasets().count(arg):
            with Timer() as timer:
                require(self.env, arg)
            self.init_benchmarks()
            # FIXME CHRIS, why can't I get it to update the list of benchmarks?
            # I have to restart
            print(f"Downloaded dataset {arg} in {timer}")
            print("Application must be restarted to make changes visible.")
        else:
            print("Unknown dataset, '" + arg + "'")
            print("Available datasets are listed with command, list_available_datasets")


    def do_list_benchmarks(self, arg):
        """List all of the available benchmarks"""
        if not self.benchmarks:
            doc_root_url = "https://facebookresearch.github.io/CompilerGym/"
            install_url = doc_root_url + "getting_started.html#installing-benchmarks"
            print("No benchmarks available. See " + install_url)
            print("Datasets can be installed with command, require_dataset")
        else:
            print(", ".join(self.benchmarks))

    def complete_set_benchmark(self, text, line, begidx, endidx):
        """Complete the set_benchmark argument"""
        return self.simple_complete(text, self.benchmarks)

    def do_set_benchmark(self, arg):
        """Set the current benchmark.
            set_benchmark <name> - set the benchmark
            The name should come from the list of benchmarks printed by the command list_benchmarks.
            Tab completion will be used if available.
            This command will delete the action history.
            Use '-' for a random benchmark.
        """
        if arg == "-":
            arg = random.choice(self.benchmarks)
            print(f"set_benchmark {arg}")

        if self.benchmarks.count(arg):
            self.stack.clear()

            # Set the current benchmark
            with Timer() as timer:
                eager_observation = self.env.reset(benchmark=arg)

            print(f"Reset {self.env.benchmark} environment in {timer}")
            if self.env.observation_space and eager_observation is not None:
                print(f"Observation: {self.env.observation_space.to_string(eager_observation)}")

            self.set_prompt()

        else:
            print("Unknown benchmark, '" + arg + "'")
            print("Bencmarks are listed with command, list_benchmarks")


    def get_actions(self):
        """Get the list of actions"""
        return self.env.action_space.names

    def do_list_actions(self, arg):
        """List all of the available actions"""
        actions = self.get_actions()
        print(", ".join(actions))

    def complete_action(self, text, line, begidx, endidx):
        """Complete the action argument"""
        return self.simple_complete(text, self.get_actions())

    def do_action(self, arg):
        """Take a single action step.
            action <name> - take the named action
            The name should come from the list of actions printed by the command list_actions.
            Tab completion will be used if available.
            Use '-' for a random action.
        """
        if not self.env.benchmark:
            print("No benchmark set, please call the set_benchmark command")
            return

        if self.stack and self.stack[-1].done:
            print("No action possible, last action ended by the environment with error:", self.stack[-1].info["error_details"])
            print("Consider commands, back or reset")
            return

        # Determine which action to apply
        actions = self.get_actions()
        if arg == "-":  # Random
            index = self.env.action_space.sample()
            arg = actions[index]
            print(f"action {arg}")
        elif actions.count(arg):
            index = actions.index(arg)
        else:
            print("Unknown action, '" + arg + "'")
            print("Actions are listed with command, list_actions")
            print("Use '-' for a random action")
            return

        # Do the step
        with Timer() as timer:
            eager_observation, eager_reward, done, info = self.env.step(index)

        # Print the eager observation, if available.
        if self.env.observation_space and eager_observation is not None:
            print(f"Observation: {self.env.observation_space.to_string(eager_observation)}")

        # Print the eager reward and the diff, if available.
        if self.env.reward_space and eager_reward is not None:
            print(f"Reward: {eager_reward:.6f}")

        # Append the history element
        hist = ActionHistoryElement(
            self.env.action_space.names[index],
            index,
            eager_observation,
            eager_reward,
            done,
            info
        )
        self.stack.append(hist)

        print(
            f"Action {self.env.action_space.names[index]} in {timer}.",
            " No effect." if hist.has_no_effect() else "",
            flush=True,
        )
        if done:
            print("Episode ended by environment: ", info["error_details"])
            print("No further actions will be possible")

    def rerun_stack(self, check_rewards = True):
        """Rerun all the actions on the stack.
        """
        self.env.reset()
        old_stack = self.stack
        self.stack = []
        for i, old_hist in enumerate(old_stack):
            eager_observation, eager_reward, done, info = self.env.step(old_hist.action_index)
            hist = ActionHistoryElement(
                old_hist.action_name,
                old_hist.action_index,
                eager_observation,
                eager_reward,
                done,
                info
            )
            self.stack.append(hist)

            if check_rewards and eager_reward != old_hist.eager_reward:
                print(f"Warning previous eager reward at {i}: {hist.action_name} was {hist.eager_reward:.6f} now {eager_reward:.6f}")

    def do_hill_climb(self, arg):
        """Do some steps of hill climbing.
            A random action is taken, but only accepted if it has a positive reward.
            An argument, if given, should be the number of steps to take.
            The search will try to improve the default reward. Please call set_default_reward if needed.
        """
        if not self.env.reward_space:
            print("No default reward set. Call set_default_reward")
            return

        try:
            num_steps = max(1, int(arg))
        except ValueError:
            num_steps = 1

        with Timer() as timer:
            for i in range(num_steps):
                index = self.env.action_space.sample()
                action = self.env.action_space.names[index]

                eager_observation, eager_reward, done, info = self.env.step(index)

                # FIXME Chris, not sure what to do if the rewards aren't eager
                accept = not done and (eager_reward is not None) and (eager_reward > 0)
                if accept:
                    # Append the history element
                    hist = ActionHistoryElement(
                        action,
                        index,
                        eager_observation,
                        eager_reward,
                        done,
                        info
                    )
                    self.stack.append(hist)
                else:
                    # Basically undo
                    self.rerun_stack()

                # FIXME Chris, I'm not sure about the reward diffs. Sometimes it looks like the reward is automatically diffed.
                print(f"Step: {i+1} Action: {action} Reward: {eager_reward:.6f} Accept: {accept}")
                if done:
                    print("Episode ended by environment: ", info["error_details"])
        print(f"Hill climbed {num_steps} steps in {timer}")

    def get_action_rewards(self):
        """Get all the rewards for the possible actions at this point"""
        items = []
        for index, action in enumerate(self.env.action_space.names):
            self.rerun_stack()
            eager_observation, eager_reward, done, info = self.env.step(index)
            hist = ActionHistoryElement(
                action,
                index,
                eager_observation,
                eager_reward,
                done,
                info
            )
            items.append(hist)
            print(f"Action: {action} Reward: {eager_reward:.6f}")

        self.rerun_stack()
        items.sort(key = lambda h: h.eager_reward, reverse = True)
        return items

    def do_try_all_actions(self, args):
        """Tries all actions from this position and reports the results in sorted order by reward"""
        with Timer() as timer:
            items = self.get_action_rewards()
        print(f"Got actions in {timer}")

        def row(item):
            return (item.action_name, item.has_effect(), item.done, f"{item.eager_reward:.6f}")
        rows = [row(item) for item in items]
        headers = ["Action", "Effect", "Done", "Eager Reward"]
        print(tabulate(rows, headers=headers, tablefmt="presto"))


    def do_greedy(self, arg):
        """Do some greedy steps.
            All actions are tried and the one with the biggest positive reward is accepted.
            An argument, if given, should be the number of steps to take.
            The search will try to improve the default reward. Please call set_default_reward if needed.
        """
        if not self.env.reward_space:
            print("No default reward set. Call set_default_reward")
            return

        try:
            num_steps = max(1, int(arg))
        except ValueError:
            num_steps = 1

        with Timer() as timer:
            for i in range(num_steps):
                best = self.get_action_rewards()[0]
                if (not best.done) and (best.eager_reward is not None) and (best.eager_reward > 0):
                    self.env.step(best.action_index)
                    self.stack.append(best)
                    print(f"Step: {i+1} Selected action: {best.action_name} Reward: {best.eager_reward:.6f}")
                else:
                    print(f"Step: {i+1} Selected no action")
                    break

        print(f"Greedy {num_steps} steps in {timer}")


    def do_list_observations(self, arg):
        """List the available observations"""
        print(", ".join(self.observations))

    def complete_observation(self, text, line, begidx, endidx):
        """Complete the observation argument"""
        return self.simple_complete(text, self.observations)

    def do_observation(self, arg):
        """Show an observation value
            observation <name> - show the named observation
            The name should come from the list of observations printed by the command list_observations.
            Tab completion will be used if available.
        """
        if not self.env.benchmark:
            print("No benchmark set, please call the set_benchmark command")
            return

        if self.observations.count(arg):
            with Timer() as timer:
                value = self.env.observation[arg]
            print(self.env.observation.spaces[arg].to_string(value))
            print(f"Observation {arg} in {timer}")
        else:
            print("Unknown observation, '" + arg + "'")
            print("Observations are listed with command, list_observations")

    def complete_set_default_observation(self, text, line, begidx, endidx):
        """Complete the set_default_observation argument"""
        return self.simple_complete(text, self.observations)

    def do_set_default_observation(self, arg):
        """Set the default observation space
            set_default_observation <name> - set the named observation
            The name should come from the list of observations printed by the command list_observations.
            Tab completion will be used if available.
            With no argument it will set to None.
            This command will rerun the actions on the stack.
        """
        if not self.env.benchmark:
            print("No benchmark set, please call the set_benchmark command")
            return

        arg = arg.strip()
        if not arg or self.observations.count(arg):
            with Timer() as timer:
                self.env.observation_space = arg if arg else None
                self.rerun_stack(check_rewards = False)
            print(f"Observation {arg} in {timer}")
        else:
            print("Unknown observation, '" + (arg if arg else "None") + "'")
            print("Observations are listed with command, list_observations")

    def do_list_rewards(self, arg):
        """List the available rewards"""
        print(", ".join(self.rewards))

    def complete_reward(self, text, line, begidx, endidx):
        """Complete the reward argument"""
        return self.simple_complete(text, self.rewards)

    def do_reward(self, arg):
        """Show an reward value
            reward <name> - show the named reward
            The name should come from the list of rewards printed by the command list_rewards.
            Tab completion will be used if available.
        """
        if not self.env.benchmark:
            print("No benchmark set, please call the set_benchmark command")
            return

        if self.rewards.count(arg):
            with Timer(f"Reward {arg}"):
                print(f"{self.env.reward[arg]:.6f}")
        else:
            print("Unknown reward, '" + arg + "'")
            print("Rewards are listed with command, list_rewards")

    def complete_set_default_reward(self, text, line, begidx, endidx):
        """Complete the set_default_reward argument"""
        return self.simple_complete(text, self.rewards)

    def do_set_default_reward(self, arg):
        """Set the default reward space
            set_default_reward <name> - set the named reward
            The name should come from the list of rewards printed by the command list_rewards.
            Tab completion will be used if available.
            With no argument it will set to None.
            This command will rerun the actions on the stack.
        """
        if not self.env.benchmark:
            print("No benchmark set, please call the set_benchmark command")
            return

        arg = arg.strip()
        if not arg or self.rewards.count(arg):
            with Timer(f"Reward {arg}"):
                self.env.reward_space = arg if arg else None
                self.rerun_stack(check_rewards = False)
        else:
            print("Unknown reward, '" + (arg if arg else "None") + "'")
            print("Rewards are listed with command, list_rewards")


    def do_commandline(self, arg):
        """Show the command line equivalent of the actions taken so far"""
        print("$", self.env.commandline(), flush=True)

    def do_stack(self, arg):
        """Show the environments on the stack. The current environment is the first shown."""
        rows = []
        total = 0
        for i, hist in enumerate(self.stack):
            name = hist.action_name
            effect = hist.has_effect()
            done = hist.done
            reward = f"{hist.eager_reward:.6f}" if hist.eager_reward is not None else "-"
            total += hist.eager_reward or 0
            row = (i+1, name, effect, done, reward, f"{total:.6f}")
            rows.append(row)
        rows.reverse()
        rows.append((0, "<init>", False, False, 0, 0))

        headers = ["Depth", "Action", "Effect", "Done", "Reward", "Cumulative Reward"]
        print(tabulate(rows, headers=headers, tablefmt="presto"))

    def do_simplify_stack(self, arg):
        """Simplify the stack
            There may be many actions on the stack which have no effect or created a negative reward.
            This command makes a basic attempt to remove them. It reruns the stack, using only the
            commands which appeared to have a effect and positive reward. If the reward is None
            e.g. if there was no default reward set, then it will only check if there was some effect.
            Note that the new rewards are not checked, so there may be odd effects caused by an action
            being removed that previously had a negative reward being necessary for a later action to
            have a positive reward. This means you might see non-positive rewards on the stack afterwards.
        """
        self.env.reset()
        old_stack = self.stack
        self.stack = []
        for i, old_hist in enumerate(old_stack):
            if old_hist.has_effect() and (old_hist.eager_reward is None or old_hist.eager_reward > 0):
                eager_observation, eager_reward, done, info = self.env.step(old_hist.action_index)
                hist = ActionHistoryElement(
                    old_hist.action_name,
                    old_hist.action_index,
                    eager_observation,
                    eager_reward,
                    done,
                    info
                )
                self.stack.append(hist)

                if eager_reward != old_hist.eager_reward:
                    print(f"Warning previous eager reward at {i}: {hist.action_name} was {old_hist.eager_reward:.6f} now {eager_reward:.6f}")


    def do_reset(self, arg):
        """Clear the stack of any actions and reset"""
        self.stack.clear()
        with Timer() as timer:
            self.env.reset()
        print(f"Reset in {timer}")

    def do_back(self, arg):
        """Undo the last action, if any"""
        if self.stack:
            top = self.stack.pop()
            with Timer() as timer:
                self.rerun_stack()
            print(f"Undid {top.action_name} in {timer}")
        else:
            print("No actions to undo")

    def do_exit(self, arg):
        """Exit"""
        print("Exiting")
        return True

    def do_breakpoint(self, arg):
        """Enter the debugger.
            If you suddenly want to do something funky with self.env, or the self.stack, this is your way in!
        """
        breakpoint()

    def default(self, line):
        """Override default to quit on end of file"""
        if line == "EOF":
            return self.do_exit(line)

        return super().default(line)


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    if FLAGS.ls_benchmark:
        benchmark = benchmark_from_flags()
        env = env_from_flags(benchmark)
        print("\n".join(sorted(env.benchmarks)))
        env.close()
        return

    with Timer("Initialized environment"):
        # FIXME Chris, I don't seem to actually get a benchmark
        benchmark = benchmark_from_flags()
        env = env_from_flags(benchmark)

    shell = CompilerGymShell(env)
    shell.cmdloop()


if __name__ == "__main__":
    try:
        main(sys.argv)
    except app.UsageError as err:
        print("Usage Error: " + str(err))
