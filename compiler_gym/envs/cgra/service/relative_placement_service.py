from pathlib import Path
from typing import Tuple, Optional, Union, List
from compiler_gym.envs.cgra.service.cgra_service import CGRASession, observation_space, Schedule, CGRA, relative_placement_directions
from compiler_gym.spaces import Reward
import traceback
from compiler_gym.service import CompilationSession
from compiler_gym.util.gym_type_hints import ObservationType, OptionalArgumentValue
from compiler_gym.views import ObservationSpaceSpec
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
import pickle

"""
Unlike in direct placement, in relative placement, we take an operation and schedule it
to it's nearby neighbours that support the operation.

"""

action_space = [
    ActionSpace(name="move",
    space=Space(
        named_discrete=NamedDiscreteSpace(
            # has a max of 9 connection dimensions (none, up, down, n, s, e, w, sooner, later)
            name=relative_placement_directions
        )
    ))
]

class RelativePlacementCGRASession(CGRASession):
    def __init__(self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark):
        try:
            print("Initailziing relplace session")
            super().__init__(working_directory, action_space, benchmark)

            # For the relative placmenet CGRA, we need to come up with an initial placement strategy.
            # While it may not be important for all classes of algorithm that this is consistent
            # after every reset, it is important for some (e.g. genetic algorithms)
            self.dfg = pickle.loads(benchmark.program.contents)
            print("Loaded DFG " + str(self.dfg))
            # TODO --- support better seeds.
            self.schedule = Schedule(self.cgra, self.dfg)
            self.initial_placement = self.get_initial_placement(self.dfg, 0)

            # At the same time, the results of this are sensitive to the starting position,
            # so, it's important that we can control the starting position.
            self.current_operation_index = 0
            
            # This is a constant that says how many times we should iterate over the array.
            self.max_iterations = 10
            
            self.iteration_number = 0
        except Exception as e:
            print(traceback.format_exc())
            raise e
            

    observation_spaces: List[ObservationSpace] = observation_space
    action_spaces = action_space

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
        try:
            return super().reset(benchmark, action_space, observation_space, reward_space)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def get_initial_placement(self, dfg, seed):
        try:
            # For now, just place the nodes in order on the CGRA.
            # Iterate through the PEs, and then increment the clock cycle
            # if we can't place.
            pe_ind = 0
            time = 0
            max_pe = self.cgra.cells_as_list()
            nodes = dfg.bfs()
            iterating = True
            was_set = True
            while iterating:
                # only move to next node if we properly set the operation last time.
                if was_set:
                    n = next(nodes, None)
                    if n is None:
                        # Finished scheduling!
                        iterating = False
                        continue
                was_set = self.schedule.set_operation(time, pe_ind, n, n.operation.latency)
                if was_set:
                    print("Set initial placement for node", str(n))
                    print("Position is ", self.schedule.get_location(n))

                # TODO -- should we check that this produces a schedule with an II?
                # Aim is to start with a very spread-out schedule that should just work ---
                # let the SA algorithm compress it, rather than trying to make
                # the SA algorithm find a valid schedule.
                pe_ind += 1
                time += n.operation.latency
                if pe_ind >= len(max_pe):
                    pe_ind = 0
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        try:
            if self.iteration_number == self.max_iterations:
                # The iteration is finished.
                return True, None, False

            step = action.int64_value
            print("Got step ", step)
            action_to_do = relative_placement_directions[step]
            print("Got step", step, "which entails moving in direction", action_to_do)
            # Get the ndoe --- the dfg.nodes is a dict, so need to access through the names
            # list.
            current_operation_node_name = self.dfg.node_names[self.current_operation_index]
            current_operation = self.dfg.nodes[current_operation_node_name]

            current_time, current_location = self.schedule.get_location(current_operation)
            print("For node ", current_operation, "found location", current_location)
            new_time = current_time
            new_location = current_location
            if action_to_do == "sooner":
                new_time -= 1
            elif action_to_do == "later":
                new_time += 1
            else:
                if action_to_do == "no_action":
                    new_location = None
                else:
                    new_location = self.cgra.get_neighbour(action_to_do, current_location)

            if new_location is not None:
                print("Swapping between", current_location, 'and', new_location)
                swapped = self.schedule.swap(current_time, current_location, new_time, new_location)
            else:
                # If the new location is none, that means that we picked a direction
                # that is invalid (ie. doesn't exist for the node in question).  To make
                # it easier on the RL/GA algorithms, we'll just silently skiup this here.
                swapped = False

            # Prepare for next iteration:
            self.current_operation_index += 1
            if self.current_operation_index > len(self.dfg.nodes) - 1:
                # Wrap around for another pass through the nodes.
                self.current_operation_index = 0
                self.iteration_number += 1

            print("After iteration, schedule is ", self.schedule)
            print("Swapped is ", swapped)

            return False, None, swapped
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        try:
            result = super().get_observation(observation_space=observation_space)
            if observation_space.name == 'II':
                ii, finished = self.schedule.get_II(self.dfg)
                if not finished:
                    # The RLLib library can't handle nones, so
                    # Just return a large punishment if this fails
                    # to schedule.
                    result = Event(int64_value=-100)
                else:
                    result = Event(int64_value=-ii)
                print ("got the result.", result)
            elif observation_space.name == 'RLMapObservations':
                # print("Got RLMap Observations", result.int64_tensor)
                pass
            return result
        except Exception as e:
            print(traceback.format_exc())
            raise e

def make_cgra_compilation_session():
    return RelativePlacementCGRASession