# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import compiler_gym.errors

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
ValidationError = compiler_gym.errors.ValidationError
