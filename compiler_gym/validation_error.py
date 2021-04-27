# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the validation error` tuple."""
from typing import Any, Dict, NamedTuple


class ValidationError(NamedTuple):
    """A ValidationError describes an error encountered in a call to
    :meth:`env.validate() <compiler_gym.envs.CompilerEnv.validate>`.
    """

    type: str
    """A short name describing the type of error that occured. E.g.
    :code:`"Runtime crash"`.
    """

    data: Dict[str, Any] = {}
    """A JSON-serialized dictionary of data that further describes the error.
    This data dictionary can contain any information that may be relevant for
    diagnosing the underlying issue, such as a stack trace or an error line
    number. There is no specified schema for this data, validators are free to
    return whatever data they like. Setting this field is optional.
    """

    def json(self):
        """Get the error as a JSON-serializable dictionary.

        :return: A JSON dict.
        """
        return self._asdict()  # pylint: disable=no-member

    @classmethod
    def from_json(cls, data) -> "ValidationError":
        """Create a validation error from JSON data.

        :param data: A JSON dict.
        :return: A validation error.
        """
        return cls(**data)
