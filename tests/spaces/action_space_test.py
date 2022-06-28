# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym/spaces/action_space.py."""
from compiler_gym.spaces import ActionSpace, Discrete, NamedDiscrete
from tests.test_main import main


class MockActionSpace:
    name = "mock"
    foo = 1

    def sample(self):
        return 1

    def seed(self, s):
        pass

    def contains(self, x):
        pass

    def __repr__(self) -> str:
        return self.name


def test_action_space_forward(mocker):
    a = MockActionSpace()
    ma = ActionSpace(a)

    assert ma.name == "mock"
    assert ma.foo == 1

    mocker.spy(a, "sample")
    assert ma.sample() == 1
    assert a.sample.call_count == 1

    mocker.spy(a, "seed")
    ma.seed(10)
    assert a.seed.call_count == 1

    mocker.spy(a, "contains")
    10 in ma
    assert a.contains.call_count == 1


def test_action_space_comparison():
    a = MockActionSpace()
    b = ActionSpace(a)
    c = MockActionSpace()

    assert b == a
    assert b.wrapped == a
    assert b != c


def test_action_space_default_string_conversion():
    """Test that to_string() and from_string() are forward to subclasses."""
    a = Discrete(name="a", n=3)
    ma = ActionSpace(a)

    assert ma.to_string([0, 1, 0]) == "0,1,0"
    assert ma.from_string("0,1,0") == [0, 1, 0]


def test_action_space_forward_string_conversion():
    """Test that to_string() and from_string() are forward to subclasses."""
    a = NamedDiscrete(name="a", items=["a", "b", "c"])
    ma = ActionSpace(a)

    assert ma.to_string([0, 1, 2, 0]) == "a b c a"
    assert ma.from_string("a b c a") == [0, 1, 2, 0]


def test_action_space_str():
    ma = ActionSpace(MockActionSpace())
    assert str(ma) == "ActionSpace(mock)"


if __name__ == "__main__":
    main()
