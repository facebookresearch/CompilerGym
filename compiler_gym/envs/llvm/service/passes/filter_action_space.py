# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Filter the list of LLVM passes to use as an action space.

This scripts reads a list of passes from stdin and for each, calls
config.include_pass() to determine whether it should be printed to stdout.
"""
import csv
import logging
import sys
from typing import Iterable

from compiler_gym.envs.llvm.service.passes import config
from compiler_gym.envs.llvm.service.passes.common import Pass


def filter_passes(pass_iterator: Iterable[Pass]) -> Iterable[Pass]:
    """Apply config.include_pass() to an input sequence of passes.

    :param pass_iterator: An iterator over Pass objects.
    :returns: A subset of the input Pass iterator.
    """
    total_count = 0
    selected_count = 0

    for pass_ in pass_iterator:
        total_count += 1
        if config.include_pass(pass_):
            selected_count += 1
            logging.debug(
                f"Selected {pass_.name} pass ({pass_.flag}) from {pass_.source}",
            )
            yield pass_

    print(
        f"Selected {selected_count} of {total_count} LLVM passes to use as actions",
        file=sys.stderr,
    )


def main(argv):
    """Main entry point."""
    del argv
    reader = csv.reader(sys.stdin, delimiter=",", quotechar='"')
    next(reader)

    pass_iterator = (Pass(*row) for row in reader)
    filtered_passes = filter_passes(pass_iterator)

    writer = csv.writer(sys.stdout, delimiter=",", quotechar='"')
    writer.writerow(Pass._fields)
    writer.writerows(sorted(list(filtered_passes), key=lambda r: r.name))


if __name__ == "__main__":
    main(sys.argv)
