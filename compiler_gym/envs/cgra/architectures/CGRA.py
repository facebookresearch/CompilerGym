# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from compiler_gym.envs.cgra.compile_settings import CGRACompileSettings

"""
This is the most abstract representation of a CGRA ----
a list of nodes, with an interconnect archtiecture.

This can be inherited from to make things easier.
"""
class CGRA(object):
    def __init__(self, nodes, noc):
        self.nodes = nodes
        self.noc = noc
        self.dim = len(self.nodes)

    def __str__(self):
        return "CGRA: (" + (str(self.nodes)) + " nodes)"

    def is_supported(self, node_index, op):
        # TODO(jcw) -- support heterogeneity
        return True

    def cells_as_list(self):
        return self.nodes[:]

    def get_neighbour(self, direction, location_from):
        return self.noc.get_neighbour(direction, location_from)

class DataPath(object):
    # Keeps track of a path through a NOC.
    # Keeping track of the source node is important to allow
    # the same bit of data to share the same path.
    def __init__(self, source_node, start_cycle, path):
        self.path = path
        self.start_cycle = start_cycle
        self.source_node = source_node

    def __len__(self):
        return len(self.path)

    def __str__(self):
        return str(self.path) + ": Starting at " + str(self.start_cycle) + ", Carrying results of " + str(self.source_node)

# Abstract class for a NOC (network on chip).
class NOC(object):
    def __init__(self):
        pass

    def get_neighbour(self, direction, location):
        assert False

    # Work out the shortest path from from_n to to_n in
    # the current NoC.
    def shortest_path(self, node, from_n, to_n) -> DataPath:
        return self.shortest_available_path(0, node, from_n, to_n, None)

    def shortest_available_path(self, start_time, node, from_n, to_n, schedule) -> DataPath:
        assert False # Abstract Class

# A class representing a NOC (netowrk on chip).
class DictNOC(NOC):
    def __init__(self, nodes, neighbours: Dict[str, List[str]]):
        super().__init__()
        # A list of all the nodes.
        self.nodes = nodes
        # A directed list of one-hop connections between nodes.
        self.neighbours = neighbours

    # Returns the neighbour within a 3D space.  I don't really
    # know how best to set this up in reality ---- espc if a node
    # doesn't really have e.g. an 'up' neighbour, but only a 'up and left at
    # the same time neighbour'.  The key constraint currently implemented
    # here is that only six directions are supported (up, down, north, south, east
    # west)
    def get_neighbour(self, direction, location):
        ns = self.neighbours[location]
        index = None
        # TODO(jcw) --- we need a better way of storing these
        # so it isn't implicit in the connection --- this implies
        # that everything that has a 'south' connection must
        # also have a north connection.
        if direction == 'north':
            index = 0
        elif direction == 'south':
            index = 1
        elif direction == 'east':
            index = 2
        elif direction == 'west':
            index = 3
        elif direction == 'up':
            index = 4
        elif direction == 'down':
            index = 5

        if index is None:
            print("Unknown index ", direction)
            assert False

        if index < len(ns):
            print("Returning a node ", len(ns))
            return ns[index]
        else:
            return None

    # Returns a DataPath object.
    def shortest_available_path(self, start_time, source_dfg_node, from_n, to_n, schedule) -> DataPath:
        # So we should obviously do this better.
        # Just a hack-y BFS search.
        seen = set()
        # Keep track of node and path so far.
        # Invariant: this is sorted by shortest
        # path.
        to_see = [(from_n, [])]

        while len(to_see) > 0:
            n, path_to = to_see[0]
            to_see = to_see[1:]
            
            if n == to_n:
                # Found the path.  By invariant, this is the shortest
                # path.
                return DataPath(source_dfg_node, start_time, path_to)

            nexts = self.neighbours[n]
            for node in nexts:
                if node in seen:
                    pass
                else:
                    curr_time = start_time + len(path_to)
                    if schedule is not None:
                        if schedule.is_occupied(source_dfg_node, curr_time, (n, node)):
                            # Can't use this as a path if it's currently
                            # occupied.
                            # TODO(jcw) --- Add support for buffered delays.
                            # continue
                            if CGRACompileSettings['IntroduceRequiredDelays']:
                                continue
                            else:
                                if CGRACompileSettings['DebugShortestPath']:
                                    print("Shortest Path failued due to occupied slot")
                                return None
                    # This is BFS, so everything must bewithin
                    # one hop of the current search.  Therefore
                    # this is the longest one, and can go at the back.
                    to_see.append((node, path_to + [(source_dfg_node, n, node)]))
        # No path between nodes.
        if CGRACompileSettings['DebugShortestPath']:
            print("Shortest Path failued due to no path found")
        return None
