# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
from io import StringIO
from typing import Any, Iterable, Optional

from tabulate import tabulate as tabulate_lib


def tabulate(
    rows: Iterable[Iterable[Any]],
    headers: Iterable[str],
    tablefmt: Optional[str] = "grid",
) -> str:
    """A wrapper around the third-party tabulate function that adds support
    for tab- and comma-separate formats.

    :param rows: The data to tabulate.
    :param headers: A list of table headers.
    :param tablefmt: The format of tables to print. For a full list of options,
        see: https://github.com/astanin/python-tabulate#table-format.
    :return: A formatted table as a string.
    """
    tablefmt

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
