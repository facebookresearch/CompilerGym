# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the validation error` tuple."""
from typing import Any, Dict

from pydantic import BaseModel


class ValidationError(BaseModel):
    """A ValidationError describes an error encountered in a call to
    :meth:`env.validate() <compiler_gym.envs.CompilerEnv.validate>`.
    """

    type: str
    """A short name describing the type of error that occured. E.g.
    :code:`"Runtime crash"`.
    """

    data: Dict[str, Any] = {}
    """A JSON-serializable dictionary of data that further describes the error.
    This data dictionary can contain any information that may be relevant for
    diagnosing the underlying issue, such as a stack trace or an error line
    number. There is no specified schema for this data, validators are free to
    return whatever data they like. Setting this field is optional.
    """

    def __lt__(self, rhs):
        # Implement the < operator so that lists of ValidationErrors can be
        # sorted.
        if not isinstance(rhs, ValidationError):
            return True
        return (self.type, self.data) <= (rhs.type, rhs.data)
