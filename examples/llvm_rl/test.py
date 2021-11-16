# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from pathlib import Path
from typing import List

from llvm_rl.model import Model


def main(argv):
    paths = argv[1:] or ["~/logs/compiler_gym/llvm_rl"]

    models: List[Model] = []
    for path in paths:
        models += Model.from_logsdir(Path(path).expanduser())

    for model in models:
        model.test()


if __name__ == "__main__":
    main(sys.argv)
