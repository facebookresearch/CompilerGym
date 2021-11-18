# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging as logging_
import sys
from typing import Optional


def init_logging(level: int = logging_.INFO, logger: Optional[logging_.Logger] = None):
    logger = logger or logging_.getLogger()

    logger.setLevel(level)
    handler = logging_.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging_.Formatter(
        fmt="%(asctime)s %(name)s] %(message)s", datefmt="%m%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
