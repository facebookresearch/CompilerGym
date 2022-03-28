# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_cc_library(
  NAME
    headers
  HDRS
    "ActionEnum.h"
    "ActionHeaders.h"
    "ActionSwitch.h"
  PUBLIC
)
