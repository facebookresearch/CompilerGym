# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Note that the tutorial is extracted from the doc string with the repeated ^
# signs. So, please keep them as they are.
"""Run a CompilerGym environment with text interface controls.

.. code-block::

    $ python -m compiler_gym.bin.manual_env --env=<env> [--benchmark=<name>] [--observation=<space>] [--reward=<space>]

The benchmark to use can be specified using :code:`--benchmark=<name>`.

**************************
CompilerGym Shell Tutorial
**************************

This program gives a basic shell through which many of commands from CompilerGym
can be executed. CompilerGym provides a simple Python interface to various
compiler functions, enabling programs to be compiled in different ways and to
make queries about those programs. The goal is to have a simple system for
machine learning in compilers.

Setting a Benchmark, Reward and Observation
-------------------------------------------
The CompilerGym operates on a program or benchmark. If not set on the command
line, the benchmark can be specified in the shell with:

.. code-block::

    compiler_gym:cbench-v1/qsort> set_benchmark <benchmark-name>

When a benchmark is set, the prompt will update with the name of the benchmark.
Supposing that is "bench", then the prompt would be:

.. code-block::

    compiler_gym:bench>

The list of available benchmarks can be shown with, though this is limited to
the first 200 benchmarks:

.. code-block::

    compiler_gym:bench> list_benchmarks

You can also see what datasets are available with this command:

.. code-block::

    compiler_gym:cbench-v1/qsort> list_datasets

The default reward and observation can be similarly set with:

.. code-block::

    compiler_gym:bench> set_default_reward <reward-name>
    compiler_gym:bench> set_default_observation <observation-name>

And lists of the choices are available with:

.. code-block::

    compiler_gym:bench> list_rewards
    compiler_gym:bench> list_observations

The default rewards and observations will be reported every time an action is
taken. So, if, for example, you want to see how the instruction count of the
benchmark program is affected by your actions, set the default reward to
"IrInstructionCount". Then the change in instruction count for each action will
be reported.

Additionally, some of the search techniques require the default reward to be
set, since they will try to optimise that reward.

Actions and the Action Stack
----------------------------
In CompilerGym an action corresponds to invoking an compiler operation
(currently an LLVM opt pass) on the intermediate representation of the program.
Each action acts on the result of the previous action and so on.

So, for example, to apply first the 'tail call elimination' pass, then the 'loop
unrolling' pass we call two actions:

.. code-block::

    compiler_gym:bench> action -tailcallelim
    compiler_gym:bench> action -loop-unroll

Each action will report its default reward. Note that multiple actions can be
placed on a single line, so that the above is equivalent to:

.. code-block::

    compiler_gym:bench> action -tailcallelim -loop-unroll

You can choose a random action, by using just a '-' as the action name:

.. code-block::

    compiler_gym:bench> action -

Since an empty line on the shell repeats the last action, you can execute many
random actions by typing that line first then holding down return.

The actions are recorded in a stack, with the latest action on the top of the
stack. You can view the action stack with stack command:

.. code-block::

    compiler_gym:bench> stack

This will show for each action if it had an effect (as computed by the
underlying compiler), whether this terminated compiler, and what the per action
and cumulative rewards are.

The last action can be undone by:

.. code-block::

    compiler_gym:bench> undo

All actions in the stack can be undone at once by:

.. code-block::

    compiler_gym:bench> reset

You can find out what the effect of each action would be by calling this
command:

.. code-block::

    compiler_gym:bench> try_all_actions

This will show a table with the reward for each action, sorted by best first.

If you have a large stack of actions, many of which are not profitable, you can
simplify the stack with this command:

.. code-block::

    compiler_gym:bench> simplify_stack

This will redo the entire stack, keeping only those actions which previously
gave good rewards. (Note this doesn't mean that the simplified stack will only
have positive rewards, some negative actions may be necessary set up for a later
positive reward.)

Current Status
--------------
For the current state of the program - after whatever actions have been called
on it - you can make several queries.

The first is to get a reward. This might not be the same as the current default
reward:

.. code-block::

    compiler_gym:bench> reward <reward-name>

You can see various observations with:

.. code-block::

    compiler_gym:bench> observation <observation-name>

Finally, you can print the equivalent command line for achieving the same
behaviour as the actions through the standard system shell:

.. code-block::

    compiler_gym:bench> commandline

Searching
---------
Some very basic search capabilities are supported, directly in the shell. Each
of them just looks for another action to add.

First, is the random search through this command:

.. code-block::

    compiler_gym:bench> action -

Multiple steps can be taken by holding down the return key.

A hill-climbing search tries an action, but will only accept it if it yields a
positive reward:

.. code-block::

    compiler_gym:bench> hill_climb <num-steps>

A simple greedy search tries all possible actions and takes the one with the
highest reward, stopping when no action has a positive reward:

.. code-block::

    compiler_gym:bench> greedy <num-steps>

Miscellaneous
-------------
One useful command is:

.. code-block::

    compiler_gym:bench> breakpoint

Which drops into the python debugger. This is very useful if you want to see
what is going on internally. There is a 'self.env' object that represents the
environment that is definitely worth exploring.

And finally:

.. code-block::

    compiler_gym:bench> exit

Drops out of the shell. :code:`Ctrl-D` should have the same effect.
"""
import cmd
import random
import readline
import sys
from itertools import islice

from absl import app, flags

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.tabulate import tabulate
from compiler_gym.util.timer import Timer

FLAGS = flags.FLAGS


# Extract the tutorial from the doc string
tutorial = "**************************".join(
    __doc__.split("**************************")[1:]
)


class ActionHistoryElement:
    """The compiler gym shell records a list of actions taken. This class represent those elements."""

    def __init__(self, action_name, action_index, observation, reward, done, info):
        """Arguments are the returns from env.step"""
        self.action_name = action_name
        self.action_index = action_index
        self.observation = observation
        self.reward = reward
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

    intro = """Welcome to the CompilerGym Shell!
---------------------------------
Type help or ? for more information.
The 'tutorial' command will give a step by step guide."""

    def __init__(self, env: CompilerEnv):
        """Initialise with an environment.
        :param env: The environment to run.
        """
        super().__init__()

        self.env = env

        # Get the benchmarks
        self.benchmarks = []
        for dataset in self.env.datasets:
            self.benchmarks += islice(dataset.benchmark_uris(), 50)
        self.benchmarks.sort()

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

    def __del__(self):
        """Tidy up in case postloop() is not called."""
        if self.env:
            self.env.close()
            self.env = None

    def do_tutorial(self, arg):
        """Print the turorial"""
        print(tutorial)

    def preloop(self):
        self.old_completer_delims = readline.get_completer_delims()
        readline.set_completer_delims(" \t\n")

    def postloop(self):
        readline.set_completer_delims(self.old_completer_delims)
        # Clear the stack
        self.stack.clear()
        self.env.close()
        self.env = None

    def set_prompt(self):
        """Set the prompt - shows the benchmark name"""
        benchmark_name = self.env.benchmark.uri
        if benchmark_name.startswith("benchmark://"):
            benchmark_name = benchmark_name[len("benchmark://") :]
        prompt = f"compiler_gym:{benchmark_name}>"
        self.prompt = f"\n{emph(prompt)} "

    def simple_complete(self, text, options):
        """Return a list of options that match the text prefix"""
        if text:
            return [opt for opt in options if opt.startswith(text)]
        else:
            return options

    def get_datasets(self):
        """Get the list of datasets"""
        return sorted([k.name for k in self.env.datasets.datasets()])

    def do_list_datasets(self, arg):
        """List all of the datasets"""
        print(", ".join(self.get_datasets()))

    def do_list_benchmarks(self, arg):
        """List the benchmarks"""
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
            arg = self.env.datasets.random_benchmark().uri
            print(f"set_benchmark {arg}")

        try:
            benchmark = self.env.datasets.benchmark(arg)
            self.stack.clear()

            # Set the current benchmark
            with Timer() as timer:
                observation = self.env.reset(benchmark=benchmark)
            print(f"Reset {self.env.benchmark} environment in {timer}")

            if self.env.observation_space and observation is not None:
                print(
                    f"Observation: {self.env.observation_space_spec.to_string(observation)}"
                )

            self.set_prompt()
        except LookupError:
            print("Unknown benchmark, '" + arg + "'")
            print("Benchmarks are listed with command, list_benchmarks")

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
        if self.stack and self.stack[-1].done:
            print(
                "No action possible, last action ended by the environment with error:",
                self.stack[-1].info["error_details"],
            )
            print("Consider commands, back or reset")
            return

        # Determine which action to apply
        actions = self.get_actions()
        # Allow for multiple actions at once
        args = arg.split()
        if not args:
            print("No action given")
            print("Actions are listed with command, list_actions")
            print("Use '-' for a random action")
            return
        # Check each action before executing
        for arg in args:
            if arg != "-" and actions.count(arg) == 0:
                print("Unknown action, '" + arg + "'")
                print("Actions are listed with command, list_actions")
                print("Use '-' for a random action")
                return
        # Replace random actions
        for i in range(len(args)):
            if args[i] == "-":
                args[i] = actions[random.randrange(self.env.action_space.n)]

        # Now do the actions
        cum_reward = 0
        actions_taken = []
        with Timer() as timer:
            for a in args:
                print(f"Action {a}")
                index = actions.index(a)

                observation, reward, done, info = self.env.step(index)

                # Print the observation, if available.
                if self.env.observation_space and observation is not None:
                    print(
                        f"Observation: {self.env.observation_space_spec.to_string(observation)}"
                    )

                # Print the reward, if available.
                if self.env.reward_space and reward is not None:
                    print(f"Reward: {reward:.6f}")
                    cum_reward += reward

                # Append the history element
                hist = ActionHistoryElement(
                    self.env.action_space.names[index],
                    index,
                    observation,
                    reward,
                    done,
                    info,
                )
                self.stack.append(hist)

                if hist.has_no_effect():
                    print("No effect")

                actions_taken.append(a)
                if hist.done:
                    print("Episode ended by environment: ", info["error_details"])
                    print("No further actions will be possible")
                    break
        print(
            f"Actions {' '.join(actions_taken)} in {timer} with reward {cum_reward}.",
            flush=True,
        )

    def rerun_stack(self, check_rewards=True):
        """Rerun all the actions on the stack."""
        self.env.reset()
        old_stack = self.stack
        self.stack = []
        for i, old_hist in enumerate(old_stack):
            observation, reward, done, info = self.env.step(old_hist.action_index)
            hist = ActionHistoryElement(
                old_hist.action_name,
                old_hist.action_index,
                observation,
                reward,
                done,
                info,
            )
            self.stack.append(hist)

            if check_rewards and reward != old_hist.reward:
                print(
                    f"Warning previous reward at {i}: {hist.action_name} was {hist.reward:.6f} now {reward:.6f}"
                )

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

        num_accepted = 0
        cum_reward = 0
        with Timer() as timer:
            for i in range(num_steps):
                index = random.randrange(self.env.action_space.n)
                action = self.env.action_space.names[index]

                observation, reward, done, info = self.env.step(index)

                accept = not done and (reward is not None) and (reward > 0)
                if accept:
                    # Append the history element
                    hist = ActionHistoryElement(
                        action, index, observation, reward, done, info
                    )
                    self.stack.append(hist)
                    num_accepted += 1
                    cum_reward += reward
                else:
                    # Basically undo
                    self.rerun_stack()

                print(
                    f"Step: {i+1} Action: {action} Reward: {reward:.6f} Accept: {accept}"
                )
                if done:
                    print("Episode ended by environment: ", info["error_details"])
        print(
            f"Hill climb complete in {timer}. Accepted {num_accepted} of {num_steps} steps for total reward of {cum_reward}."
        )

    def get_action_rewards(self):
        """Get all the rewards for the possible actions at this point"""
        items = []
        for index, action in enumerate(self.env.action_space.names):
            self.rerun_stack()
            observation, reward, done, info = self.env.step(index)
            hist = ActionHistoryElement(action, index, observation, reward, done, info)
            items.append(hist)
            print(f"Action: {action} Reward: {reward:.6f}")

        self.rerun_stack()
        items.sort(key=lambda h: h.reward, reverse=True)
        return items

    def do_try_all_actions(self, args):
        """Tries all actions from this position and reports the results in sorted order by reward"""
        if not self.env.reward_space:
            print("No default reward set. Call set_default_reward")
            return

        with Timer("Got actions"):
            items = self.get_action_rewards()

        def row(item):
            return (
                item.action_name,
                item.has_effect(),
                item.done,
                f"{item.reward:.6f}",
            )

        rows = [row(item) for item in items]
        headers = ["Action", "Effect", "Done", "Reward"]
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
                if (not best.done) and (best.reward is not None) and (best.reward > 0):
                    self.env.step(best.action_index)
                    self.stack.append(best)
                    print(
                        f"Step: {i+1} Selected action: {best.action_name} Reward: {best.reward:.6f}"
                    )
                else:
                    print(f"Step: {i+1} Selected no action.")
                    if i + 1 < num_steps:
                        print("Greedy search stopping early.")
                    break

        print(f"Greedy {i+1} steps in {timer}")

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
        if arg == "" and self.env.observation_space:
            arg = self.env.observation_space_spec.id

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
        arg = arg.strip()
        if not arg or self.observations.count(arg):
            with Timer() as timer:
                self.env.observation_space = arg if arg else None
                self.rerun_stack(check_rewards=False)
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
        if arg == "" and self.env.reward_space:
            arg = self.env.reward_space.id

        if self.rewards.count(arg):
            with Timer(f"Reward {arg}"):
                print(f"{self.env.reward[arg]:.6f}")
        else:
            print(f"Unknown reward, '{arg}'")
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
        arg = arg.strip()
        if not arg or self.rewards.count(arg):
            with Timer(f"Reward {arg}"):
                self.env.reward_space = arg if arg else None
                self.rerun_stack(check_rewards=False)
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
            reward = f"{hist.reward:.6f}" if hist.reward is not None else "-"
            total += hist.reward or 0
            row = (i + 1, name, effect, done, reward, f"{total:.6f}")
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
            if old_hist.has_effect() and (
                old_hist.reward is None or old_hist.reward > 0
            ):
                observation, reward, done, info = self.env.step(old_hist.action_index)
                hist = ActionHistoryElement(
                    old_hist.action_name,
                    old_hist.action_index,
                    observation,
                    reward,
                    done,
                    info,
                )
                self.stack.append(hist)

                if reward != old_hist.reward:
                    print(
                        f"Warning previous reward at {i}: {hist.action_name} was {old_hist.reward:.6f} now {reward:.6f}"
                    )

    def do_reset(self, arg):
        """Clear the stack of any actions and reset"""
        self.stack.clear()
        with Timer("Reset"):
            self.env.reset()
        self.set_prompt()

    def do_back(self, arg):
        """Undo the last action, if any"""
        if self.stack:
            top = self.stack.pop()
            with Timer(f"Undid {top.action_name}"):
                self.rerun_stack()
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

    with Timer("Initialized environment"):
        benchmark = benchmark_from_flags()
        env = env_from_flags(benchmark)

    shell = CompilerGymShell(env)
    shell.cmdloop()


if __name__ == "__main__":
    try:
        main(sys.argv)
    except app.UsageError as err:
        print("Usage Error: " + str(err))
