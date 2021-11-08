# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from compiler_gym.wrappers import CompilerEnvWrapper

logger = logging.getLogger(__name__)


# TODO(github.com/facebookresearch/CompilerGym/issues/469): Once step() and
# reset() no longer raise exceptions than this wrapper class can be removed.
class JustKeepGoingEnv(CompilerEnvWrapper):
    """This wrapper class prevents the step(), reset(), or close() methods from
    raising an exception.

        Just keep swimming ...
            |\\    o
            | \\    o
        |\\ /  .\\ o
        | |       (
        |/\\     /
            |  /
            |/
    """

    def step(self, *args, **kwargs):
        try:
            return self.env.step(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("step() error: %s", e)

            # Return "null" observation / reward.
            default_observation = (
                self.env.observation_space_spec.default_value
                if self.env.observation_space
                else None
            )
            default_reward = (
                float(
                    self.env.reward_space_spec.reward_on_error(self.env.episode_reward)
                )
                if self.env.reward_space
                else None
            )

            self.close()

            return default_observation, default_reward, True, {"error_details": str(e)}

    def reset(self, *args, **kwargs):
        try:
            return super().reset(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("reset() error, retrying: %s", e)
            self.close()
            return self.reset(*args, **kwargs)

    def close(self):
        try:
            self.env.close()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Ignoring close() error: %s", e)
