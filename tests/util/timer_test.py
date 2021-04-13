# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:timer."""
from time import sleep

from compiler_gym.util import timer
from tests.test_main import main


def test_humanize_duration_seconds():
    assert timer.humanize_duration(5) == "5.000s"
    assert timer.humanize_duration(500.111111) == "500.1s"


def test_humanize_duration_ms():
    assert timer.humanize_duration(0.0055) == "5.5ms"
    assert timer.humanize_duration(0.5) == "500.0ms"
    assert timer.humanize_duration(0.51) == "510.0ms"
    assert timer.humanize_duration(0.9999) == "999.9ms"


def test_humanize_duration_us():
    assert timer.humanize_duration(0.0005) == "500.0us"
    assert timer.humanize_duration(0.0000119) == "11.9us"


def test_humanize_duration_ns():
    assert timer.humanize_duration(0.0000005) == "500.0ns"
    assert timer.humanize_duration(0.0000000019) == "1.9ns"


def test_humanize_duration_negative_seconds():
    assert timer.humanize_duration(-1.5) == "-1.500s"


def test_humanize_duration_hms():
    assert timer.humanize_duration_hms(0.05) == "0:00:00"
    assert timer.humanize_duration_hms(0.999) == "0:00:00"
    assert timer.humanize_duration_hms(5) == "0:00:05"
    assert timer.humanize_duration_hms(500.111111) == "0:08:20"
    assert timer.humanize_duration_hms(4210.4) == "1:10:10"
    assert timer.humanize_duration_hms(36000) == "10:00:00"


def test_timer_elapsed_before_reset():
    t = timer.Timer()
    assert t.time == 0
    sleep(0.1)
    assert t.time == 0


def test_timer_elapsed_remains_constant():
    with timer.Timer() as t:
        sleep(0.1)
    elapsed_a = t.time
    assert elapsed_a > 0
    sleep(0.1)
    elapsed_b = t.time
    assert elapsed_b == elapsed_a


if __name__ == "__main__":
    main()
