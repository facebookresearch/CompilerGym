# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
from io import StringIO
from typing import Any, Iterable, Optional

from absl import flags
from tabulate import tabulate as tabulate_lib

flags.DEFINE_string(
    "tablefmt",
    "grid",
    "The format of tables to print. "
    "For a full list of options, see: https://github.com/astanin/python-tabulate#table-format",
)
FLAGS = flags.FLAGS


def tabulate(
    rows: Iterable[Iterable[Any]],
    headers: Iterable[str],
    tablefmt: Optional[str] = None,
) -> str:
    """A wrapper around the third-party tabulate function that adds support
    for a --tablefmt flag, and the ability to format to tab- or comma-separate
    formats.

    :param rows: The data to tabulate.
    :param headers: A list of table headers.
    :param tablefmt: A table format to override the --tablefmt flag.
    :return: A formatted table as a string.
    """
    tablefmt = tablefmt or FLAGS.tablefmt

    if tablefmt == "tsv" or tablefmt == "csv":
        sep = {"tsv": "\t", "csv": ","}[tablefmt]
        buf = StringIO()
        writer = csv.writer(buf, delimiter=sep)
        writer.writerow([str(x) for x in headers])
        for row in rows:
            writer.writerow([str(x) for x in row])
        return buf.getvalue()
    else:
        return tabulate_lib(
            rows,
            headers=headers,
            tablefmt=tablefmt,
        )
