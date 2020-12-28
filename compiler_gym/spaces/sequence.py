# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

from gym.spaces import Space


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
        size_range: Tuple[int, Optional[int]] = (0, None),
        dtype=bytes,
        opaque_data_format: Optional[str] = None,
    ):
        """Constructor.

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
        """
        self.size_range = size_range
        self.dtype = dtype
        self.opaque_data_format = opaque_data_format

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
        for element in x:
            if not isinstance(element, self.dtype):
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
            self.size_range == other.size_range
            and self.dtype == other.dtype
            and self.opaque_data_format == other.opaque_data_format
        )
