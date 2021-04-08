# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Any, Callable


def memoized_property(func: Callable[..., Any]) -> Callable[..., Any]:
    """A property decorator that memoizes the result.

    This is used to memoize the results of class properties, to be used when
    computing the property value is expensive.

    :param func: The function which should be made to a property.

    :returns: The decorated property function.
    """
    attribute_name = "_memoized_property_" + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute_name):
            setattr(self, attribute_name, func(self))

        return getattr(self, attribute_name)

    return decorator
