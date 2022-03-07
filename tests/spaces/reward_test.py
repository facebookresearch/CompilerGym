# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.spaces.Reward."""
import pytest

from compiler_gym.spaces import Reward
from tests.test_main import main


def test_reward_id_backwards_compatibility():
    """Test that Reward.id is backwards compatible with Reward.name.

    See: github.com/facebookresearch/CompilerGym/issues/381
    """
    with pytest.warns(DeprecationWarning, match="renamed `name`"):
        reward = Reward(id="foo")

    assert reward.id == "foo"
    assert reward.name == "foo"


if __name__ == "__main__":
    main()
