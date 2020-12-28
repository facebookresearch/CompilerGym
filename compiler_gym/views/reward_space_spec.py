# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.service import scalar_range2tuple
from compiler_gym.service.proto import RewardSpace


class RewardSpaceSpec(object):
    """Specification of a reward space.

    :ivar id: The name of the reward space.
    :vartype id: str

    :ivar index: The index into the list of reward spaces that the service
        supports.
    :vartype index: int

    :ivar range: The lower and upper bounds of the reward.
    :vartype range: Tuple[float, float]

    :ivar success_threshold: The cumulative reward threshold before an episode is
        considered successful. For example, episodes where reward is scaled to
        an existing heuristic can be considered "successful" when the reward
        exceeds the existing heuristic.
    :vartype success_threshold: Optional[float]

    :ivar deterministic: Whether the reward signal is deterministic.
        Examples of non-deterministic rewards are those with measurement noise
        such as program runtime.
    :vartype deterministic: bool

    :ivar platform_dependent: Whether the reward values depend on the execution
        environment of the service.
    :vartype platform_dependent: bool
    """

    def __init__(self, index: int, spec: RewardSpace):
        self.id = spec.name
        self.index = index
        self.range = scalar_range2tuple(spec.range)
        self.success_threshold = (
            spec.success_threshold if spec.has_success_threshold else None
        )
        self.deterministic = spec.deterministic
        self.platform_dependent = spec.platform_dependent

    def __repr__(self) -> str:
        return f"RewardSpaceSpec({self.id})"
