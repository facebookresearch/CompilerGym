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


def frozen_class(cls):
    """Prevents setting attributes on a class after construction.

    Wrap a class definition to declare it frozen:

        @frozen_class class MyClass:
            def __init__(self):
                self.foo = 0

    Any attempt to set an attribute outside of construction will then raise an
    error:

        >>> c = MyClass()
        >>> c.foo = 5
        >>> c.bar = 10
        TypeError
    """
    cls._frozen = False

    def frozen_setattr(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError(
                f"Cannot set attribute {key} on frozen class {cls.__name__}"
            )
        object.__setattr__(self, key, value)

    def init_decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = frozen_setattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls
