# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import numpy as np
from gym.spaces import Space

from compiler_gym.spaces.common import issubdtype
from compiler_gym.spaces.scalar import Scalar


class Sequence(Space):
    """A sequence of values. Each element of the sequence is of `dtype`. The
    length of the sequence is bounded by `size_range`.

    Example:

    ::

        >>> space = Sequence(size_range=(0, None), dtype=str)
        >>> space.contains("Hello, world!")
        True

    ::

        >>> space = Sequence(size_range=(256, 256), dtype=bytes)
        >>> space.contains("Hello, world!")
        False

    :ivar size_range: A tuple indicating the `(lower, upper)` bounds for
        sequence lengths. An upper bound of `None` means no upper bound. All
        sequences must have a lower bound of length >= 0.
    :ivar dtype: The data type for each element in a sequence.
    :ivar opaque_data_format: An optional string describing an opaque data
        format, e.g. a data structure that is serialized to a string/binary
        array for transmission to the client. It is up to the client and service
        to agree on how to decode observations using this value. For example,
        an opaque_data_format of `string_json` could be used to indicate that
        the observation is a string-serialized JSON value.
    """

    def __init__(
        self,
        name: str,
        size_range: Tuple[int, Optional[int]] = (0, None),
        dtype=bytes,
        opaque_data_format: Optional[str] = None,
        scalar_range: Optional[Scalar] = None,
    ):
        """Constructor.

        :param name: The name of the space.

        :param size_range: A tuple indicating the `(lower, upper)` bounds for
            sequence lengths. An upper bound of `None` means no upper bound. All
            sequences must have a lower bound of length >= 0.

        :param dtype: The data type for each element in a sequence.

        :param opaque_data_format: An optional string describing an opaque data
            format, e.g. a data structure that is serialized to a string/binary
            array for transmission to the client. It is up to the client and
            service to agree on how to decode observations using this value. For
            example, an opaque_data_format of `string_json` could be used to
            indicate that the observation is a string-serialized JSON value.

        :param scalar_range: If specified, this denotes the legal range of each
            element in the sequence. This is enforced by :meth:`contains()
            <compiler_gym.spaces.Sequence.contains>` checks.
        """
        self.name = name
        self.size_range = size_range
        self.dtype = dtype
        self.opaque_data_format = opaque_data_format
        self.scalar_range = scalar_range

    def __repr__(self) -> str:
        upper_bound = "inf" if self.size_range[1] is None else self.size_range[1]
        d = f" -> {self.opaque_data_format}" if self.opaque_data_format else ""
        return (
            f"{self.dtype.__name__}_list<>[{int(self.size_range[0])},{upper_bound}]){d}"
        )

    def contains(self, x):
        lower_bound = self.size_range[0]
        upper_bound = float("inf") if self.size_range[1] is None else self.size_range[1]
        if not (lower_bound <= len(x) <= upper_bound):
            return False

        # TODO(cummins): The dtype API is inconsistent. When dtype=str or
        # dtype=bytes, we expect this to be the type of the entire sequence. But
        # for dtype=int, we expect this to be the type of each element. We
        # should distinguish these differences better.
        if self.dtype in {str, bytes}:
            if not isinstance(x, self.dtype):
                return False
        elif hasattr(x, "dtype"):
            if not issubdtype(x.dtype, self.dtype):
                return False

        # Run the bounds check on every scalar element, if there is a scalar
        # range specified.
        elif self.scalar_range:
            return all(self.scalar_range.contains(s) for s in x)
        else:
            for element in x:
                if not issubdtype(type(element), self.dtype):
                    return False

        return True

    def sample(self):
        """
        .. warning::
            The `Sequence` space cannot be sampled from.

        :raises NotImplementedError: Not supported.
        """
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return False
        return (
            self.name == other.name
            and self.size_range == other.size_range
            and np.dtype(self.dtype) == np.dtype(other.dtype)
            and self.opaque_data_format == other.opaque_data_format
            and self.scalar_range == other.scalar_range
        )
