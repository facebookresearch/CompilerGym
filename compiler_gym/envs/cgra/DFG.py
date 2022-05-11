import json
from pathlib import Path
import random

from importlib_metadata import entry_points
from compiler_gym.service.proto import (
Benchmark
)
from typing import Optional, List
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
import compiler_gym.third_party.llvm as llvm
from compiler_gym.envs.cgra.Operations import Operation, operation_from_name

class Edge(object):
    def __init__(self, type):
        self.type = type

class Node(object):
    def __init__(self, name, operation):
        self.name = name
        self.operation = operation

    def __str__(self):
        return "Node with name " + self.name + " and op " + str(self.operation)

class DFG(object):
    def __init__(self, working_directory: Optional[Path] = None, benchmark: Optional[Benchmark] = None, from_json: Optional[Path] = None, from_text: Optional[str] = None):
        # Copied from here: https://github.com/facebookresearch/CompilerGym/blob/development/examples/loop_optimizations_service/service_py/loops_opt_service.py
        # self.inst2vec = _INST2VEC_ENCODER

        if from_json is not None:
            self.load_dfg_from_json(from_json)
        elif from_text is not None:
            self.load_dfg_from_text(from_text)
        elif benchmark is not None:
            # Only re-create the JSON file if we aren't providing an existing one.
            # The existing ones are mostly a debugging functionality.
            with open(self.working_directory / "benchmark.c", "wb") as f:
                f.write(benchmark.program.contents)

            # We use CGRA-Mapper to produce a DFG in JSON.
            run_command(
                ["cgra-mapper", self.src_path, self.dfg_path]
            )

            # Now, load in the DFG.
            self.load_dfg_from_json(self.dfg_path)

    def __str__(self):
        res = "nodes are: " + str(self.nodes) + " and edges are " + str(self.adj)
        return res

    def load_dfg_from_json(self, path):
        import json
        with open(path, 'r') as p:
            # This isnt' text, but I think the json.loads
            # that this calls just works?
            self.load_dfg_from_text(p)

    def load_dfg_from_text(self, text):
        import json
        f = json.loads(text)
        self.nodes = {}
        self.node_names = []
        self.edges = []
        self.adj = {}
        self.entry_points = f['entry_points']

        # build the nodes first.
        for node in f['nodes']:
            self.nodes[node['name']] = (Node(node['name'], operation_from_name(node['operation'])))
            self.adj[node['name']] = []
            self.node_names.append(node['name'])

        for edge in f['edges']:
            self.edges.append(Edge(edge['type']))

        # Build the adj matrix:
        for edge in f['edges']:
            fnode = edge['from']
            tnode = edge['to']

            self.adj[fnode].append(tnode)
    
    # Bit slow this one --- the adjacency matrix is backwards for it :'(
    def get_preds(self, node):
        preds = []
        for n in self.adj:
            if node.name in self.adj[n]:
                preds.append(self.nodes[n])

        return preds

    def get_succs(self, node):
        succs = []
        for n in self.adj[node.name]:
            succs.append(self.nodes[n])
        return succs

    # TODO -- fix this, because for a graph with multiple entry nodes,
    # this doesn't actually give the right answer :)
    # (should do in most cases)
    def bfs(self):
        to_explore = self.entry_points[:]
        print ("Doing BFS, entry points are ")
        print(self.entry_points)
        seen = set()

        while len(to_explore) > 0:
            head = to_explore[0]
            to_explore = to_explore[1:]
            if head in seen:
                continue
            seen.add(head)
            yield self.nodes[head]

            # Get the following nodes.
            following_nodes = self.adj[head]
            to_explore += following_nodes

# Generate a test DFG using the operations in
# 'operations'.
def generate_DFG(operations: List[Operation], size, seed=0):
    random.seed(seed)
    # Start with some 0-input ops:
    start_ops = random.randint(1, min(size, 3))

    # Jump-start this --- in reality, these can be
    # phi nodes coming from previous tiers of the loop,
    # or variables coming from outside the loop.
    start_options = []
    print("Generating DFG with ", start_ops, " starting nodes")
    for op in operations:
        if op.inputs == 0:
            start_options.append(op)

    node_number = 0
    edge_number = 0

    entry_points = []
    nodes = {}
    node_names = []
    nodes_list = []
    edges = []
    adj = {}

    # Keep track of variables that we should probably use somewhere.
    unused_outputs = []
    for i in range(start_ops):
        name = "node" + str(node_number)
        node_names.append(name)
        n = Node(name, random.choice(start_options))
        node_number += 1

        nodes[name] = n
        nodes_list.append(n)
        entry_points.append(name)
        unused_outputs.append(n)
        adj[name] = []

    while len(nodes) < size:
        # Generate a new node.
        operation = random.choice(operations)
        name = "node" + str(node_number)
        node_names.append(name)
        node_number += 1

        # Get inputs for this:
        inputs = []
        while len(inputs) < operation.inputs:
            # Select random nodes: baised towards the unused ones.
            if random.randint(0, 10) > 6 and len(unused_outputs) > 0:
                inputs.append(unused_outputs[0])
                unused_outputs = unused_outputs[1:]
            else:
                inputs.append(random.choice(nodes_list))
        # If the node has no arguments, then we should add it
        # as an entry point.  --- todo --- should we just skip
        # this avoid creating graphs with too many constant loads?
        if operation.inputs == 0:
            entry_points.append(name)

        # now create the edges.
        for inp in inputs:
            edge = Edge('data')
            # Not too sure why this doens't have the start/end points.
            # Think it's a dead datafield.
            edges.append(edge)

            adj[inp.name].append(name)

        this_node = Node(name, operation)
        nodes[name] = this_node
        nodes_list.append(this_node)
        unused_outputs.append(this_node)
        adj[name] = []

    res = DFG()
    res.adj = adj
    res.nodes = nodes
    res.entry_points = entry_points
    res.edges = edges
    res.node_names = node_names
    print(res.nodes)

    return res