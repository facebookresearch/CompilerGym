# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a parallelized exhaustive search of an action space.

All possible combinations of actions up to a finite limit are
evaluated, but partial sequences of actions that end up in the same
state are deduplicated, sometimes dramatically reducing the size of
the search space. Can also be configured to do a beam search.

Example usage:

    $ python explore.py --env=llvm-ic-v0 --benchmark=cbench-v1/dijkstra \
       --episode_length=10 --actions=-simplifycfg,-instcombine,-mem2reg,-newgvn

Use --help to list the configurable options.
"""
import hashlib
import math
from enum import IntEnum
from heapq import nlargest
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Lock
from time import time

import humanize
from absl import app, flags

from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

flags.DEFINE_list(
    "actions",
    [],
    "A list of action names to enumerate. If not provided, all actions are used.",
)
flags.DEFINE_integer("episode_length", 5, "The number of steps in each episode.")
flags.DEFINE_integer(
    "nproc", cpu_count(), "The number of parallel worker threads to run."
)
flags.DEFINE_integer(
    "topn",
    0,
    "If positive, explore only the top n states for each sequence length. "
    "This is in effect the width of a beam search.",
)
flags.DEFINE_integer(
    "show_topn", 3, "Show this many top sequences " "at each sequence length."
)

FLAGS = flags.FLAGS


class CustomEnv:
    """A wrapper for an LLVM env that takes a subset of the actions.

    Taking a subset in the env avoids the easy error to make to pass in
    i as an action instead of actions[i] where actions is the subset.
    """

    def __init__(self):
        self._env = env_from_flags(benchmark_from_flags())
        try:
            # Project onto the subset of transformations that have
            # been specified to be used.
            if not FLAGS.actions:
                self._action_indices = list(range(len(self._env.action_space.names)))
            else:
                self._action_indices = [
                    self._env.action_space.flags.index(a) for a in FLAGS.actions
                ]
            self._action_names = [
                self._env.action_space.names[a] for a in self._action_indices
            ]

        finally:
            # The program will not terminate until the environment is
            # closed, not even if there is an exception.
            self._env.close()

    def action_names(self, actions):
        return [self._action_names[a] for a in actions]

    def step(self, action):
        return self._env.step(self._action_indices[action])

    def reset(self):
        self._env.reset()

    def close(self):
        self._env.close()

    def action_count(self):
        return len(self._action_indices)

    def actions(self):
        return range(self.action_count())

    @property
    def observation(self):
        return self._env.observation


# Used to determine if two rewards are equal up to a small
# tolerance. Cannot use math.isclose with default parameters as it
# sets abs_tol to 0, which means that a zero reward will compare
# unequal with e.g. 1e-100, leading to bugs.
def rewards_close(a, b):
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-10)


NO_EDGE = -1


class Node:
    def __init__(self, reward_sum, edge_count):
        self.reward_sum = reward_sum
        self.edges = [NO_EDGE] * edge_count
        self.back_edge = None


# Represents env states as nodes and actions as edges.
class StateGraph:
    def __init__(self, edges_per_node):
        self._edges_per_node = edges_per_node
        self._nodes = []
        self._fingerprint_to_index = dict()

    def add_or_find_node(self, fingerprint, reward_sum):
        if fingerprint in self._fingerprint_to_index:
            node_index = self._fingerprint_to_index[fingerprint]
            assert rewards_close(
                self._nodes[node_index].reward_sum, reward_sum
            ), f"{self._nodes[node_index].reward_sum} != {reward_sum}"
            return (node_index, False)
        node_index = self.node_count()
        self._fingerprint_to_index[fingerprint] = node_index
        node = Node(reward_sum, self._edges_per_node)
        self._nodes.append(node)
        return (node_index, True)

    def add_edge(self, from_node_index, edge_index, to_node_index):
        assert edge_index in range(self._edges_per_node)
        assert from_node_index in range(self.node_count())
        assert to_node_index in range(self.node_count())
        assert self.get_edge(from_node_index, edge_index) == NO_EDGE

        from_node = self._nodes[from_node_index]
        from_node.edges[edge_index] = to_node_index

        to_node = self._nodes[to_node_index]
        if to_node.back_edge is None:
            to_node.back_edge = (from_node_index, edge_index)

    def get_edge(self, from_node_index, edge_index):
        assert edge_index < self._edges_per_node
        assert from_node_index < self.node_count()
        return self._nodes[from_node_index].edges[edge_index]

    # Returns a path back to node 0. For this to work, edges have to
    # be added in a order so that the subgraph consisting of the first
    # in-coming edge to each node defines a tree with node 0 as the
    # root.
    def node_path(self, node_index):
        assert node_index < self.node_count()

        path = []
        while node_index != 0:
            back_edge = self._nodes[node_index].back_edge
            assert back_edge is not None
            (prior_node_index, edge_index) = back_edge
            node_index = prior_node_index
            path.append(edge_index)
        path.reverse()
        return path

    def reward_sum(self, node_index):
        return self._nodes[node_index].reward_sum

    def node_count(self):
        return len(self._nodes)


def env_to_fingerprint(env):
    # TODO: There is some sort of state in the env that is not
    # captured by this. Figure out what it is and fix it. Also
    # consider adding a fingerprint observation to env.
    if False:
        # BitcodeFile is slower, so using Ir instead.
        path = env.observation["BitcodeFile"]
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = env.observation["Ir"].encode()

    return hashlib.sha256(data).digest()


def compute_edges(env, sequence):
    edges = []
    for action in env.actions():
        env.reset()
        reward_sum = 0.0
        for action in sequence + [action]:
            _, reward, _, _ = env.step(action)
            reward_sum += reward

        edges.append((env_to_fingerprint(env), reward_sum))
    return edges


class NodeTypeStats:
    """Keeps statistics on the exploration."""

    class EdgeType(IntEnum):
        unpruned = 0
        self_pruned = 1
        cross_pruned = 2
        back_pruned = 3
        dropped = 4

    def __init__(self, action_count):
        self._action_count = action_count
        self._depth = 0
        self._depth_start_time_in_seconds = time()

        # Nodes added at this depth.
        self._depth_stats = [0] * len(self.EdgeType)

        # Nodes added across all depths.
        self._all_stats = [0] * len(self.EdgeType)

        # The full number of nodes that is theoretically in the graph
        # at this depth if no nodes had been pruned anywhere.
        self._full_depth_stats = [0] * len(self.EdgeType)

        # The full number of nodes across depths if no nodes had been
        # pruned anywhere.
        self._full_all_stats = [0] * len(self.EdgeType)

    def start_depth_and_print(self, episode_length):
        self._depth += 1
        print(
            f"*** Processing depth {self._depth} of {episode_length} with",
            f"{self._depth_stats[self.EdgeType.unpruned]} states and",
            f"{self._action_count} actions.\n",
        )

        self._depth_start_time_in_seconds = time()
        self._full_depth_stats[self.EdgeType.unpruned] = 0
        for e in self.EdgeType:
            self._depth_stats[e] = 0
            if e != self.EdgeType.unpruned:
                # The pruned nodes at the prior depth would have
                # turned into this many more nodes at the next depth.
                self._full_depth_stats[e] *= self._action_count
                self._full_all_stats[e] += self._full_depth_stats[e]

            # At a certain point these large numbers just clutter up
            # the display.
            if self._full_all_stats[e] > 1e9:
                self._full_all_stats[e] = float("inf")
            if self._full_depth_stats[e] > 1e9:
                self._full_depth_stats[e] = float("inf")

    def note_edge(self, edge_type):
        self._adjust_edges(edge_type, 1)

    def drop_unpruned_edge(self):
        self._adjust_edges(self.EdgeType.unpruned, -1)
        self._adjust_edges(self.EdgeType.dropped, 1)

    def _adjust_edges(self, edge_type, adjustment):
        self._depth_stats[edge_type] += adjustment
        self._all_stats[edge_type] += adjustment
        self._full_depth_stats[edge_type] += adjustment
        self._full_all_stats[edge_type] += adjustment

    def end_depth_and_print(self, env, graph, best_node):
        align = 16

        def number_list(stats):
            return "".join(
                [humanize.intcomma(n).rjust(align) for n in stats + [sum(stats)]]
            )

        legend = [e.name for e in self.EdgeType] + ["sum"]
        print(
            "                        ",
            "".join([header.rjust(align) for header in legend]),
        )
        print("        added this depth", number_list(self._depth_stats))
        print("   full nodes this depth", number_list(self._full_depth_stats))
        print("     added across depths", number_list(self._all_stats))
        print("full added across depths", number_list(self._full_all_stats))

        # If this does not match then something was over or under
        # counted. Based on x^0 + x^1 ... + x^n = (x^(n+1) - 1) / (x -
        # 1), which is the number of nodes in a complete tree where
        # every interior node has x children. If the numbers are too
        # large then there may not be equality due to rounding, so do
        # not check this in that case.
        full_all_sum = sum(self._full_all_stats)
        assert full_all_sum > 1e9 or full_all_sum == (
            pow(env.action_count(), self._depth + 1) - 1
        ) / (env.action_count() - 1)

        depth_time_in_seconds = time() - self._depth_start_time_in_seconds
        print()
        print(f"Time taken for depth: {depth_time_in_seconds:0.2f} s")

        if FLAGS.show_topn >= 1:
            print(f"Top {FLAGS.show_topn} sequence(s):")
            for n in nlargest(
                FLAGS.show_topn,
                range(graph.node_count()),
                key=lambda n: graph.reward_sum(n),
            ):
                print(
                    f"  {graph.reward_sum(n):0.4f} ",
                    ", ".join(env.action_names(graph.node_path(n))),
                )

        print("\n")


# Compute an action graph and use it to find the optimal sequence
# within episode_length actions. Uses as many threads as there are
# elements in envs.
def compute_action_graph(envs, episode_length):
    assert len(envs) >= 1
    env_queue = Queue()
    for env in envs:
        env_queue.put(env)
    pool = ThreadPool(len(envs))

    stats = NodeTypeStats(action_count=env.action_count())
    graph = StateGraph(edges_per_node=env.action_count())

    # Add the empty sequence of actions as the starting state.
    envs[0].reset()
    best_node, _ = graph.add_or_find_node(env_to_fingerprint(envs[0]), 0.0)
    stats.note_edge(NodeTypeStats.EdgeType.unpruned)

    # A node is defined by a sequence of actions that end up in that
    # node. Nodes are deduplicated based on a hash (fingerprint) of
    # their state, so that if two sequences of actions end up with the
    # same state than they will also converge on the same node in the
    # graph.
    #
    # The outer loop goes through sequences by the depth/length of the
    # sequence, first all sequences of one element, then all sequences
    # of two elements and so on. This partition of the nodes creates
    # multiple kinds of edges:
    #
    #  Back edges.  Edges pointing to the same or lower depth. These
    #  edges represent sequences that are equivalent to a shorter
    #  sequence. These edges are pruned as no new nodes can be
    #  discovered from them and they cannot participate in a minimal
    #  best sequence as they are not minimal. Self edges are excluded
    #  from this definition.
    #
    #  Self edges.  Loops, i.e. edges that go from a node to
    #  itself. This represents actions that do not change the
    #  state. These are pruned for the same reason as back edges and
    #  have their own category as they are a very common case.
    #
    #  Cross edges.  These are edges that go forward to the next depth
    #  but there is already another edge that goes to the same
    #  node. The edge itself is not pruned from the graph, as it can
    #  be part of a minimal optimal sequence, but since the
    #  destination node already exists there is no new node introduced
    #  by a cross edge, so you could consider that the hypothetical
    #  distinct node that this edge might have created is pruned
    #  through deduplication.
    #
    #  Unpruned edges.  These are edges that go forward to the next
    #  depth and there is not yet any other edge that goes to that
    #  node. This kind of edge causes a new node to be created that
    #  will be expanded at the next depth.
    #
    #  Dropped.  These are otherwise unpruned edges that end up
    #  getting dropped due to a limit on how many states to explore
    #  per depth.
    #
    # If there are N nodes, then they are indexed as [0, N) in order
    # of insertion. New nodes are added to the graph when an unpruned
    # edge is found that points to them. A node is expanded when its
    # edges are computed and added to the graph, potentially causing
    # new nodes to be added.
    #
    # The nodes are partitioned into 3 ranges:
    #
    #  [0; depth_start)  These nodes are already expanded and done with.
    #
    #  [depth_start; next_depth_start)  These are the nodes at the
    #  current depth that will be expanded to create nodes at the next
    #  depth.
    #
    #  [next_depth_start, N)  These are the nodes that have been added
    #  at this iteration of the loop to be expanded at the next
    #  iteration of the loop.
    dropped = set()
    next_depth_start = 0
    for depth in range(episode_length):
        stats.start_depth_and_print(episode_length)
        depth_start = next_depth_start
        next_depth_start = graph.node_count()

        if depth_start == next_depth_start:
            print("There are no more states to process, stopping early.")
            break

        lock = Lock()

        def expand_node(node_index):
            with lock:
                if node_index in dropped:
                    return node_index, ()
                path = graph.node_path(node_index)

            # ThreadPool.map doesn't support giving each thread its
            # own env, so we use a queue instead. Each thread gets
            # some env and has exclusive use of it while it has it.
            local_env = env_queue.get()
            edges = compute_edges(local_env, path)
            env_queue.put(local_env)

            return node_index, edges

        undropped = [
            n for n in range(depth_start, next_depth_start) if n not in dropped
        ]
        computed_edges = pool.map(expand_node, undropped)

        # This could easily be done also with a lock as above, saving
        # the memory for computed_edges, and when done that way, the
        # lock is not at all contended. However, there is currently an
        # issue with non-determinism with multithreading and so it's
        # preferable for right now to make the node ordering
        # deterministic, so as to not add to the non-determinism, even
        # though the node ordering shouldn't matter.
        for node_index, edges in computed_edges:
            for i, (fingerprint, reward_sum) in zip(range(len(edges)), edges):
                target_node_index, inserted = graph.add_or_find_node(
                    fingerprint, reward_sum
                )

                if target_node_index == node_index:  # self edge
                    assert not inserted
                    stats.note_edge(NodeTypeStats.EdgeType.self_pruned)
                    continue
                if target_node_index < next_depth_start:  # back edge
                    assert not inserted
                    stats.note_edge(NodeTypeStats.EdgeType.back_pruned)
                    continue

                if not inserted:  # cross edge
                    stats.note_edge(NodeTypeStats.EdgeType.cross_pruned)
                else:  # unpruned - node was added
                    stats.note_edge(NodeTypeStats.EdgeType.unpruned)

                graph.add_edge(node_index, i, target_node_index)

                best_reward = graph.reward_sum(best_node)
                if reward_sum > best_reward and not rewards_close(
                    best_reward, reward_sum
                ):
                    best_node = target_node_index

        if FLAGS.topn > 0:
            top_nodes = list(range(next_depth_start, graph.node_count()))
            top_nodes.sort(key=lambda n: graph.reward_sum(n), reverse=True)
            for n in top_nodes[FLAGS.topn :]:
                dropped.add(n)
                stats.drop_unpruned_edge()

        stats.end_depth_and_print(envs[0], graph, best_node)


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    print(f"Running with {FLAGS.nproc} threads.")
    assert FLAGS.nproc >= 1
    try:
        envs = []
        for _ in range(FLAGS.nproc):
            envs.append(CustomEnv())
        compute_action_graph(envs, episode_length=FLAGS.episode_length)
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    app.run(main)
