# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME errors
  SRCS
    __init__.py
    dataset_errors.py
    download_errors.py
    service_errors.py
    validation_errors.py
)
