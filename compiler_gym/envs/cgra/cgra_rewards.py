from compiler_gym.spaces import Reward

class IntermediateIIReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="II",
            observation_spaces=["II"],
            default_value=0,
            default_negates_returns=True,
            deterministic=True,
            platform_dependent=True,
        )
        pass

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        print("Computing Reward: got II of ", observations[0])
        if observations[0] is None:
            # If we just failed to generate a valid schedule all together,
            # return a punishment.  Not 100% sure what this punishment should
            # be though.
            return -1.0
        # Add a constant negative reward for not figuring it out?
        return -float(observations[0]) - 0.1


"""
For algorithms where a 'right' answer is quick to arrive at,
the intermediate rewards are less important.
"""
class FinalIIReward(Reward):
    def __init__(self):
        super().__init__(
            name='II',
            observation_spaces=['II', 'Done'],
            default_value=0,
            default_negates_returns=True,
            deterministic=True,
            platform_dependent=True
        )

    def reset(self, benchmark: str, observation_view):
        del benchmark

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        print ("Computing Reward: get II of ", observations[0])
        print ("Got finished: ", observations[1])

        if observations[0] is None:
            return -0.1
        if observations[1]:
            return -float(observations[0]) - 0.1
        else:
            return -0.1