# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from re import I
import pickle

from typing import Optional, Tuple, List, Dict, Set, Union
from pathlib import Path
from compiler_gym.envs.cgra.compile_settings import CGRACompileSettings
from compiler_gym.views import ObservationSpaceSpec
from compiler_gym.spaces import Reward
from compiler_gym.envs.llvm.llvm_rewards import CostFunctionReward
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv
from compiler_gym.util.gym_type_hints import ObservationType, OptionalArgumentValue
from compiler_gym.service import CompilationSession
from compiler_gym.envs.cgra.architectures.CGRA import CGRA, NOC, DictNOC, DataPath
from compiler_gym.util.commands import run_command
from compiler_gym.service.proto import (
ActionSpace,
Benchmark,
DoubleRange,
Event,
Int64Box,
Int64Range,
Int64Tensor,
NamedDiscreteSpace,
ObservationSpace,
Space,
StringSpace
)
import compiler_gym.third_party.llvm as llvm
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
from compiler_gym.envs.cgra.DFG import DFG, Node, Edge, generate_DFG
from compiler_gym.envs.cgra.Operations import *

from compiler_gym.service.proto.compiler_gym_service_pb2 import Int64SequenceSpace
#from compiler_gym.service.runtime import create_and_run_compiler_gym_service

def load_CGRA(self, file):
    # TODO(jcw) -- properly load CGRA
    return CGRA(5, 5)

def load_NOC(self, file):
    # TODO(jcw) -- properly load NOC (network on chip).

    # Initialize to a straight-line NOC
    return DictNOC([(x, x + 1) for x in range(5)])

# This is just a wrapper around an actual object that
# is a schedule --- see the Schedule object for something
# that can be interacted with.
class InternalSchedule(object):
    def __init__(self, cgra, dfg):
        self.dfg = dfg # For tensorization.
        self.cgra = cgra
        self.operations = self.initialize_schedule()

    # Returns a complicated 4-tuple.
    # (Timeslot, Location, Path Requirements, BufferRequirements)
    # Path Requirements is a List[DataPath]
    # BufferRequirements is List[(Start Time, End Time)] with
    # the buffering location implicitly at the Location.
    def get_valid_slots(self, dependencies, latency, noc_schedule, buffer_schedule):
        # First, get the finished location of each of the dependencies.
        min_time = 0
        dep_locations = []
        for dep in dependencies:
            time, location = self.get_location(dep)
            # The operation must be scheduled --- it's a dependnecy! (support for loops needed.)
            if time is None:
                print ("Operation dependency", dep, " has not been scheduled (cross-dependencies not supported).")
                assert False
            latency = self.get_latency(time, location)

            min_time = max(min_time, time + latency)
            dep_locations.append((dep, time, location, latency))

        # Starting from the min time, iterate over every unoccupied
        # tile and see if it's reachable.  If it is both unoccupied
        # and reachable by all the deps, then we can use it.
        t = min_time
        # Keep track of the number of tries we've had without
        # finding a valid slot.  We fail if this gets too high---
        # the point is just to make this easier to debug, although
        # perhaps some algorithms could make better use of this.
        tries_since_slot_found = 0
        while True:
            if tries_since_slot_found > 1000:
                print("It has been more than 1000 slots looked at since we found a valid slot --- likely in an infinite loop.")
                assert False
            while t >= len(self.operations):
                # Make sure we aren't past the end of the schedule.
                self.operations.append(self.add_timestep())

            for loc in range(len(self.operations[t])):
                # Check first if this is occupied:
                if CGRACompileSettings['DebugGetValidSlots']:
                    print("Searchign location ", loc, "at time", t)
                is_free = self.slots_are_free(loc, t, t + latency)
                if not is_free:
                    if CGRACompileSettings['DebugGetValidSlots']:
                        print("Location was not free.")
                    # The latency compute step is expensive, so skip
                    # if possible.
                    continue
                
                # Now, check to see if all of the operands can reach
                # this --- note that we can't have them sharing routing
                # resources, so we have  to keep track of what we are using
                # here.
                used_resources = []
                arrival_times = []
                # We operate on a cloned copy becase we haven't actually
                # scheduled the op yet --- just trying to look
                # for valid locations!
                noc_schedule_clone = noc_schedule.clone()
                buffer_schedule_clone = buffer_schedule.clone()
                earliest_execution_time = t

                failed = False
                # Keep track fo the resrouces that are getting reserved
                # within these clones so they can be
                # returned and updated in a scheduling state.
                paths = []
                buffer_slots = []

                for (dep, dep_time, dep_loc, dep_latency) in dep_locations:
                    if CGRACompileSettings['DebugGetValidSlots']:
                        print("Checking dependency from ", dep_time, "and location", dep_loc)
                    if CGRACompileSettings['BufferingMode'] == 'before_transmit':
                        pass
                    else:
                        pass

                    finish_time = dep_time + dep_latency
                    path = self.cgra.noc.shortest_available_path(finish_time, dep, dep_loc, loc, noc_schedule_clone)
                    if path is None:
                        if CGRACompileSettings['DebugGetValidSlots']:
                            print("Path was not free.")
                        # Couldn't schedule the node here!
                        failed = True
                        break
                    else:
                        arrival_times.append(finish_time + len(path))
                        # Reserve the routing resources in the NOC clone.
                        noc_schedule_clone.occupy_path(path)
                        paths.append(path) # Keep track of the paths this requies.
                        earliest_execution_time = max(earliest_execution_time, finish_time + len(path))
                for arrival_time in arrival_times:
                    # TODO(jcw) --- note that if the problem is that the buffers
                    # get full, it's unlikely that dealying further will solve
                    # the problem.  Not 100% sure what the actual solution to
                    # this will be.
                    reserved = buffer_schedule_clone.occupy_buffer(loc, arrival_time, earliest_execution_time)
                    if CGRACompileSettings['DebugGetValidSlots']:
                        print("Trying to reserve buffering space from arrival time ", arrival_time, "...")
                        print("Reserved:", reserved)
                    buffer_slots.append((arrival_time, earliest_execution_time))
                    if not reserved:
                        # Not enough buffering
                        failed = True
                        break
                if not failed:
                    # We were able to route everything tof this possible placement.
                    tries_since_slot_found = 0
                    yield t, loc, paths, buffer_slots
                else:
                    tries_since_slot_found += 1
                    
            t += 1

    # Returns a fixed-length tensor for this schedule.
    # It focuses on the last few cycles.
    def to_rlmap_tensor(self, node, time_window_size=1):
        # Build up a tensor of timesxcgra.dimxcgra.dim as per
        # RLMap paper.
        # Note that they don't use a times dimension, as their
        # PEs are fixed within a single schedule.
        # We want to fcous the results ardoung the operation
        # that we are looking at.
        time_window, _ = self.get_location(node)
        # Aim is to be symmetric around the central time window.
        start_time = time_window - (time_window_size // 2)
        end_time = time_window + ((time_window_size - 1) // 2)

        result_tensor = []
        for t in range(start_time, end_time + 1):
            if t < 0:
                # If this is a time before the start of the schedule, just
                # add some zeroes.
                result_tensor += ([0] * ((self.cgra.dim + 1) * (self.cgra.dim + 1)))
                continue
            if t >= len(self.operations):
                # Likewise if we are past the end fo the current schedule
                result_tensor += ([0] * ((self.cgra.dim + 1) * (self.cgra.dim + 1)))
                continue

            for loc in range(self.cgra.dim + 1):
                elem = self.operations[t][loc]
                if elem is None:
                    result_tensor += [0] * (self.cgra.dim + 1)
                else:
                    # Get preds and succs from this node:
                    pred_nodes = self.dfg.get_preds(elem)
                    succ_nodes = self.dfg.get_succs(elem)
                    state_vector = [0] * (self.cgra.dim + 1)

                    for l in pred_nodes:
                        time, loc = self.get_location(l)
                        state_vector[loc] = 1
                    for l in succ_nodes:
                        time, loc = self.get_location(l)
                        state_vector[loc] = 2
                    result_tensor += state_vector

        return result_tensor

    def __str__(self):
        res = "Schedule is \n"
        for t in range(len(self.operations)):
            res += "time " + str(t) + ": "
            res += str([str(n) for n in self.operations[t]])
            res += "\n"

        return res

    def locations(self):
        for x in range(self.cgra.dim + 1):
            yield x

    def add_timestep(self):
        ops = []
        for x in range(self.cgra.dim + 1):
            ops.append(None)
        return ops

    def initialize_schedule(self):
        ops = []

        ops.append(self.add_timestep())
        return ops

    def get_node(self, optime, oploc):
        if optime < len(self.operations):
            return self.operations[optime][oploc]
        else:
            return None

    # See how long the thing scheduled at (T, X) lasts
    # for --- note that if you pass in T + N, and the op
    # started at T, you'll get true_latnecy - N.
    def get_latency(self, optime, oploc):
        op = self.get_node(optime, oploc)
        old_op = op
        t = optime

        while op is not None and op == old_op:
            t += 1
            old_op = op
            op = self.get_node(t, oploc)

        return t - optime

    # Return true if the CGRA slots are free between
    # start_tiem and end_time in location (x, y)
    def slots_are_free(self, x, start_time, end_time):
        for t in range(start_time, end_time):
            # Add more timesteps to the schedule as required.
            while t >= len(self.operations):
                self.operations.append(self.add_timestep())

            print ("Looking at time ", t, "op location", x)
            print( "oplen is ", len(self.operations[t]))
            if self.operations[t][x] is not None:
                return False
        return True

    # Return the earliest time after earliest time that we can
    # fit an op of length 'length' in location x, y
    def get_free_time(self, earliest_time, length, loc):
        while not self.slots_are_free(loc, earliest_time, earliest_time + length):
            earliest_time += 1
        return earliest_time

    def set_operation(self, time, loc, node, latency):
        while time + latency >= len(self.operations):
            self.operations.append(self.add_timestep())

        if self.slots_are_free(loc, time, time + latency):
            for t in range(time, time + latency):
                self.operations[t][loc] = node
            return True
        else:
            # Not set
            return False

    # Blindly clear the operation from time to time + latency.
    def clear_operation(self, time, loc, latency):
        while time + latency >= len(self.operations):
            self.operaitons.append(self.add_timestep())

        cleared = False
        for t in range(time, time + latency):
            cleared = True
            self.operations[t][loc] = None

        assert cleared # sanity-check that we actually did something.

    def get_location(self, node: Node):
        # TODO -- make a hash table or something more efficient if required.
        for t in range(len(self.operations)):
            for x in range(self.cgra.dim + 1):
                if self.operations[t][x] is None:
                    continue
                if self.operations[t][x].name == node.name:
                    return t, x
        return None, None

    def free_times(self, x):
        occupied_before = False
        for t in range(len(self.operations)):
            if self.operations[t][x] is not None:
                occupied_before = True
            else:
                if occupied_before:
                    # This was occupired at the last timestep t,
                    # so it's become freed at this point.
                    occupied_before = False
                    yield t

    def has_use(self, x):
        for t in range(len(self.operations)):
            if self.operations[t][x] is not None:
                return True
        return False

    def alloc_times(self, x):
        free_before = True
        for t in range(len(self.operations)):
            if self.operations[t][x] is not None:
                # Was previously free.
                if free_before:
                    # Now was not free before.
                    free_before = False
                    yield t
            else:
                free_before = True

class BufferSchedule(object):
    def __init__(self):
        self.schedule = []

    def clone(self):
        new_sched = BufferSchedule()
        for bufs in self.schedule:
            new_sched.schedule.append(dict(bufs))
        return new_sched

    def occupy_buffer(self, loc, from_time, to_time):
        for t in range(from_time, to_time + 1):
            while t >= len(self.schedule):
                self.schedule.append({})

            if loc in self.schedule[t]:
                self.schedule[t][loc] += 1
                max_buf = CGRACompileSettings['BufferLimits']
                # if the max buffering is set to 0 or -ve, assume
                # infinite buffering.
                if max_buf > 0 and self.schedule[t][loc] > max_buf:
                    return False
            else:
                self.schedule[t][loc] = 1

        return True

class NOCSchedule(object):
    def __init__(self):
        self.schedule = []

    def clone(self):
        # Return a deep copy of this schedule.
        new_schedule = NOCSchedule()
        for conns in self.schedule:
            new_schedule.schedule.append(dict(conns))
        return new_schedule

    def occupy_path(self, path: DataPath):
        start_cycle = path.start_cycle
        for hop in path.path:
            self.occupy_connection(path.source_node, start_cycle, hop)
            start_cycle += 1

    def occupy_connection(self, node: Node, time: int, connection):
        while time >= len(self.schedule):
            self.schedule.append({})

        if connection in self.schedule[time] and self.schedule[time][connection] != node.name:
            # Can't occuoy an already occupied connection.
            print("Tried to occupy connection already occupied by ", self.schedule[time][connection], "with node", node.name)
            assert False
        else:
            self.schedule[time][connection] = node.name

    def is_occupied(self, source_dfg_node: Node, time, hop):
        if time >= len(self.schedule):
            # Not occupied if beyond current suecule
            return False
        else:
            if hop in self.schedule[time]:
                if self.schedule[time] == source_dfg_node.name:
                    return False # Technically ocupied, but can be
                    # shared.
                print("Slot is occupired with node ", self.schedule[time].name)
                print("Looking to use it for node ", source_dfg_node.name)
                return True
            else:
                return False

class Schedule(object):
    def __init__(self, cgra, dfg):
        # Note that we don't store the DFG because this actually
        # creates the schedule, but so that this can be tensorized.
        self.dfg = dfg
        self.cgra = cgra

        self.operations = InternalSchedule(cgra, self.dfg)

    def __str__(self):
        return "CGRA:" + str(self.operations)

    def set_operation(self, time, index, node, latency):
        return self.operations.set_operation(time, index, node, latency)

    def to_rlmap_tensor(self, node, time_window_size=1):
        # Get the RLMap Tensor --- note that it is node dependent
        # as this is a compiler that can support time-multiplexing
        # of operations on nodes.
        return self.operations.to_rlmap_tensor(node, time_window_size=time_window_size)

    def swap(self, origin_time, origin_index, target_time, target_index, dfg, allow_invalid=True):
        # This is a slightly non-trivial function since operations may have non-one
        # latency.  We treat swap-points as the starting-points of the operation ---
        # if the target point is in the middle of another operation, we choose to
        # schedule this /at the start of the other operation/
        
        # First, we need to make sure that the whole target
        # window is clear:
        op_latency = self.operations.get_latency(origin_time, origin_index)
        operation_node = self.operations.get_node(origin_time, origin_index)
        # Check that the target window is clear:
        # IF its' not clear, the easiest thing to do is a no-op.
        assert target_time is not None
        assert target_index is not None
        assert operation_node is not None

        target_window_is_clear = self.operations.set_operation(target_time, target_index, operation_node, operation_node.operation.latency)
        # Now do the swap of operations
        if target_window_is_clear and target_time >= 0: # Dont' swap into past!
            print("Doing swap between ", origin_index, 'at', origin_time, 'to', target_index, 'at', target_time, 'with latency', op_latency, "(invalid swaps is allowed is ", allow_invalid, ")")
            InitializationInterval, _ = self.get_InitializationInterval(dfg)
            assert InitializationInterval is not None #We are tryign to preserve this invariant through the scheduling.
            self.operations.set_operation(target_time, target_index, operation_node, op_latency)
            self.operations.clear_operation(origin_time, origin_index, op_latency)
            if not allow_invalid:
                # Check that this produced a valid schedule.
                # TODO --- make this check more efficient -- we don't have
                # to recompute the whole InitializationInterval.
                InitializationInterval, _ = self.get_InitializationInterval(dfg)
                if InitializationInterval is None:
                    print("Undo the swap!")
                    # Undo the swap
                    self.operations.set_operation(origin_time, origin_index, operation_node, op_latency)
                    self.operations.clear_operation(target_time, target_index, op_latency)
                    InitializationInterval, _ = self.get_InitializationInterval(dfg)
                    assert InitializationInterval is not None
                    return False
            return True
        else:
            return False

    # This returns an iterator that iterates over possible
    # valid slots for an operation.  This allows for things like
    # random placement.
    def get_valid_slots(self, dependencies, latency, noc_schedule, buffer_schedule):
        return self.operations.get_valid_slots(dependencies, latency, noc_schedule, buffer_schedule)

    def clear_operation(self, time, index, latency):
        self.operations.clear_operation(time, index, latency)

    def get_location(self, node):
        return self.operations.get_location(node)

    def compute_and_reserve_communication_distance(self, cycle, n1, n2, noc_schedule):
        # Compute the shortest path:
        n1_t, n1_loc = self.get_location(n1)
        n2_t, n2_loc = self.get_location(n2)

        # TODO -- a sanity-check that cycle is after this might be a good idea.
        path = self.cgra.noc.shortest_available_path(cycle, n1, n1_loc, n2_loc, noc_schedule)

        if path is None:
            # TODO --- we should probably punish the agent a lot here
            # rather than crashing?
            print("Schedule has not valid path between ", n1_loc, "and", n2_loc, "at time", cycle)
            return None
        else:
            noc_schedule.occupy_path(path)
        
        # I think we don't need the whole path?  Not too sure though.
        return len(path)

    def get_InitializationInterval(self, dfg):
        # Compute the InitializationInterval of the current schedule.

        # We don't require the placement part to be actually correct ---
        # do the actual schedule what we generate can differ
        # from the schedule we have internally.
        actual_schedule = InternalSchedule(self.cgra, dfg)
        noc_schedule = NOCSchedule() # The NOC schedule is recomputed
        # every time because it is dependent on the actual
        # schedule.
        buffer_schedule = BufferSchedule()

        # What cycle does this node get executed on?
        cycles_start = {}
        # What cycle does the result of this node become
        # available on?
        cycles_end = {}

        # Keep track of when resources can be re-used.
        freed = {} # When we're done
        used = {} # When we start

        # We keep track of whether scheduling is finished
        # elsewhere --- this is just a sanity-check.
        finished = True

        # Step 1 is to iterate over all the nodes
        # in a BFS manner.
        for node in dfg.bfs():
            # For each node, compute the latency,
            # and the delay to get the arguments to
            # reach it.
            preds = dfg.get_preds(node)

            # Get the time that this operation has
            # been scheduled for.
            scheduled_time, loc = self.get_location(node)
            earliest_time = scheduled_time

            if scheduled_time is None:
                finished = False
                # This is not a complete operation
                continue

            if CGRACompileSettings['DebugGetInitializationInterval']:
                print("Looking at node ", node)
                print("Has preds ", [str(p) for p in preds])

            arrival_times = []
            for pred in preds:
                if pred.name not in cycles_end:
                    finished = False
                    continue

                pred_cycle = cycles_end[pred.name]
                if CGRACompileSettings['DebugGetInitializationInterval']:
                    print ("Have pred that finishes at cycle", pred_cycle)

                # Compute the time to this node, and
                # reserve those paths on the NoC.
                distance = self.compute_and_reserve_communication_distance(pred_cycle, pred, node, noc_schedule)

                if CGRACompileSettings['DebugGetInitializationInterval']:
                    print("Failed due to distance not working", distance)
                if distance is None:
                    # This schedule isn't possible due to conflicting memory requirements.
                    return None, False

                # Compute when this predecessor reaches this node:
                arrival_time = distance + pred_cycle
                earliest_time = max(earliest_time, arrival_time)
                arrival_times.append(arrival_time)

            # Setup the buffering requirements:
            for arr_time in arrival_times:
                ntim, nloc = self.get_location(node)
                reserved = buffer_schedule.occupy_buffer(nloc, arrival_time, earliest_time)
                if not reserved:
                    # This schedule isn't possible due to buffering requirements.
                    # TODO --- can we delay computation to get the buffering
                    # satified?
                    return None, False

            # TODO --- compute a penalty based on the gap between
            # operations to account for buffering.

            # Check that the PE is actually free at this time --- if it
            # isn't, push the operation back.
            latency = operation_latency(node.operation)
            free_time = actual_schedule.get_free_time(earliest_time, latency, loc)
            actual_schedule.set_operation(free_time, loc, node, latency)
            if free_time != earliest_time:
                # We should probably punish the agent for this.
                # Doesn't have any correctness issues as long as we
                # assume infinite buffering (which we shouldn't do, and
                # will eventually fix).
                print("Place failed to place node in a sensible place: it is already in use!")

            # TODO --- do we need to punish this more? (i.e. integrate
            # buffering requirements?)

            # This node should run at the earliest time available.
            cycles_start[node.name] = free_time
            cycles_end[node.name] = free_time + operation_latency(node.operation)

            if CGRACompileSettings['DebugGetInitializationInterval']:
                print("Node ", node.name, "has earliest time", earliest_time)

        # Now that we've done that, we need to go through all the nodes and
        # work out the InitializationInterval.
        # When was this computation slot last used? (i.e. when could
        # we overlap the next iteration?)
        min_InitializationInterval = 0
        for loc in actual_schedule.locations():
            # Now, we could achieve better performance
            # by overlapping these in a more fine-grained
            # manner --- but that seems like a lot of effort
            # for probably not much gain?
            # there ar probably loops where the gain
            # is not-so-marginal.
            if actual_schedule.has_use(loc):
                # Can only do this for PEs that actually have uses!
                last_free = max(actual_schedule.free_times(loc))
                first_alloc = min(actual_schedule.alloc_times(loc))

                difference = last_free - first_alloc
                if CGRACompileSettings['DebugGetInitializationInterval']:
                    print ("Diff at loc", loc, "is", difference)
                min_InitializationInterval = max(min_InitializationInterval, difference)

        # TODO --- we should probably return some kind of object
        # that would enable final compilation also.
        return min_InitializationInterval, finished

# Create a dummy CGRA that is a bunch of PEs in a row with neighbor-wise communciations
nodes = [1, 2, 3, 4]
neighbours_dict = {}
for n in range(1, len(nodes)):
    neighbours_dict[n] = [n + 1, n - 1]
neighbours_dict[0] = [n + 1]
neighbours_dict[len(nodes)] = [n - 1]

compilation_session_noc = DictNOC(nodes, neighbours_dict)
compilation_session_cgra = CGRA(nodes, compilation_session_noc)

action_space = [ActionSpace(name="Schedule",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=[str(x) for x in compilation_session_cgra.cells_as_list()]
                )
                # int64_box=Int64Box(
                #     low=Int64Tensor(shape=[2], value=[0, 0]),
                #     high=Int64Tensor(shape=[2], value=[compilation_session_cgra.x_dim, compilation_session_cgra.y_dim])
                # )
                )
            )
        ]

MAX_WINDOW_SIZE = 100

# This is here rather than in the RP environment because
# it's needed to define the observation space.
rlmap_time_depth = 20
# Have an entry for each cell in the compilation_session CGRA and also
# a note of the current operation
rlmap_tensor_size = ((compilation_session_cgra.dim + 1) * (compilation_session_cgra.dim + 1)) * rlmap_time_depth + 1
relative_placement_directions = ["no_action", "up", "down", "north", "south", "east", "west", "sooner", "later"]
observation_space = [
            # ObservationSpace(
            #     name="dfg",
            #     space=Space(
            #         string_value=StringSpace(length_range=(Int64Range(min=0)))
            #     ),
            #     deterministic=True,
            #     platform_dependent=False,
            #     default_observation=Event(string_value="")
            # ),
            ObservationSpace(name="ir",
                space=Space(
                    # TODO -- I think this should be a window of operations
                    # around the current one.
                    int64_sequence=Int64SequenceSpace(length_range=Int64Range(min=0, max=MAX_WINDOW_SIZE), scalar_range=Int64Range(min=0, max=len(Operations)))
                )
            ),
            ObservationSpace(name="CurrentInstruction",
                space=Space(
                    int64_value=Int64Range(min=0, max=len(Operations)),
                # TODO -- also need to figure out how to make this
                # a graph?
                ),
                deterministic=True,
                platform_dependent=False
            ),
            ObservationSpace(name="CurrentInstructionIndex",
                space=Space(
                    int64_value=Int64Range(min=0, max=MAX_WINDOW_SIZE)
                )),
            ObservationSpace(name="InitializationInterval",
                space=Space(
                    int64_value=Int64Range(min=0)
                )),
            ObservationSpace(name="RLMapObservations",
                space=Space(
                    int64_box=Int64Box(
                        low=Int64Tensor(shape=[rlmap_tensor_size], value=([0] * rlmap_tensor_size)),
                        high=Int64Tensor(shape=[rlmap_tensor_size], value=([100000] * rlmap_tensor_size))
                    )
                )
            )

            # ObservationSpace(
            #     name="Schedule",
            #     space=Space(
            #         int64_box=Int64Box(
            #             low=Int64Tensor(shape=[2], value=[0, 0]),
            #             high=Int64Tensor(shape=[2], value=[cgra.x_dim, cgra.y_dim])
            #         )
            #     )
            # )
        ]

class CGRASession(CompilationSession):
    def __init__(self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Starting a compilation session for CGRA" + str(self.cgra))
        # Load the DFG (from a test_dfg.json file):
        self.dfg = pickle.loads(benchmark.program.contents)
        self.schedule = Schedule(self.cgra, self.dfg)

        self.current_operation_index = 0
        self.time = 0 # Starting schedulign time --- we could do
        # this another way also, by asking the agent to come up with a raw
        # time rather than stepping through.
        # TODO -- load this properly.
        self.dfg_to_ops_list()

    def reset(self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        observation_space: Union[
            OptionalArgumentValue, str, ObservationSpaceSpec
        ] = OptionalArgumentValue.UNCHANGED,
        reward_space: Union[
            OptionalArgumentValue, str, Reward
        ] = OptionalArgumentValue.UNCHANGED,
    ):
        print("Reset started")
        if benchmark is not None:
            self.dfg = pickle.loads(benchmark.program.contents)
        else:
            self.dfg = None
        self.schedule = Schedule(self.cgra, self.dfg)
        self.current_operation_index = 0
        self.time = 0
        print("Reset complete")

    def dfg_to_ops_list(self):
        # Embed the DFG into an operations list that we go through ---
        # it contains two things: the name of the node, and the index
        # that corresponds to within the Operations list.
        self.ops = []
        self.node_order = []
        for op in self.dfg.bfs():
            # Do we need to do a topo-sort here?
            ind = operation_index_of(op.operation)
            if ind == -1:
                print("Did not find operation " + str(op.operation) + " in the set of Operations")
                assert False

            self.ops.append(ind)
            self.node_order.append(op)

    cgra = compilation_session_cgra
    action_spaces = action_space

    observation_spaces = observation_space
    # TODO --- a new observation space corresponding to previous actions

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        # print("Action has fields {}".format(str(action.__dict__)))
        print("Action is {}".format(str(action)))

        response = action.int64_value
        if response == -1:
            # Do a reset of the env:
            self.reset()
            return False, None, True

        # Update the CGRA to schedule the current operation at this space:
        # Take 0 to correspond to a no-op.
        had_effect = False
        if response > 0:
            # Schedule is set up to take the operation at the response index
            # index - 1.
            if self.current_operation_index >= len(self.node_order):
                # We've scheduled past the end!
                return False, None, False

            node = self.node_order[self.current_operation_index]
            latency = operation_latency(node.operation)
            op_set = self.schedule.set_operation(self.time, response - 1, node, latency)

            # Check that the InitializationInterval still exists:
            InitializationInterval, finished = self.schedule.get_InitializationInterval(self.dfg)
            has_InitializationInterval = InitializationInterval is not None
            if not has_InitializationInterval:
                # Unset that operation:
                print("Setting operation resulted in failed DFG mapping")
                print(self.schedule)
                self.schedule.clear_operation(self.time, response - 1, latency)
                print("After clearning, have")
                print(self.schedule)
                op_set = False # Need to punish.
                new_InitializationInterval, _ = self.schedule.get_InitializationInterval(self.dfg)
                assert (new_InitializationInterval is not None) # This should not
                # be non-existent after un-scheduling.

            if op_set:
                had_effect = True
                print("Scheduled operation", str(self.node_order[self.current_operation_index]))
                print("Got an InitializationInterval of ", InitializationInterval)
                self.current_operation_index += 1
        elif response == 0:
            self.time += 1

        done = False
        if self.current_operation_index >= len(self.ops):
            done = True

        print("At end of cycle, have schedule")
        print(self.schedule)
        print("Done is ", done)

        return done, None, had_effect

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing an observation over the space")

        if observation_space.name == "ir":
            # TODO --- This should be a DFG?
            return Event(int64_tensor=Int64Tensor(shape=[len(self.ops)], value=self.ops))
        elif observation_space.name == "Schedule":
            # TODO -- needs to return the schedule for the past
            # CGRA history also?
            box_value = self.schedule.current_iteration
            return Event(int64_box_value=box_value)
        elif observation_space.name == "CurrentInstruction":
            # Return the properties of the current instruction.
            if self.current_operation_index >= len(self.ops):
                # I don't get why this is ahpepning --- just make
                # sure the agent doesn't yse this.  I think it
                # might happen on the last iteration.
                return Event(int64_value=-1)
            else:
                return Event(int64_value=self.ops[self.current_operation_index])
        elif observation_space.name == "CurrentInstructionIndex":
            # Return a way to localize the instruction within the graph.
            return Event(int64_value=self.current_operation_index)
        elif observation_space.name == "InitializationInterval":
            print("Computing InitializationInterval for schedule:")
            print(self.schedule)
            ii, finished = self.schedule.get_InitializationInterval(self.dfg)
            print("Got InitializationInterval", ii)
            print ("Finished is ", finished)
            return Event(int64_value=ii)
        elif observation_space.name == "RLMapObservations":
            print("Getting RLMap Observations")
            print("Observation space is " + str(type(observation_space)))
            current_operation_index = self.current_operation_index
            node = self.node_order[current_operation_index]
            # TODO --- add encoding of the CGRA constraints (not required for faithful
            # reimplementation of RLMap, but probably required for a fair comparison.)
            schedule_encoding = self.schedule.to_rlmap_tensor(node, time_window_size=rlmap_time_depth)

            full_res = [current_operation_index] + schedule_encoding
            if len(full_res) != rlmap_tensor_size:
                print("Tensor sizes don't match!", len(full_res), ' and ', rlmap_tensor_size)
                assert False

            return Event(int64_tensor=Int64Tensor(shape=[len(full_res)], value=full_res))

def make_cgra_compilation_session():
    return CGRASession