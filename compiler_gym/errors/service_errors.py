# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines service related errors."""


class ServiceError(Exception):
    """Error raised from the service."""


class SessionNotFound(ServiceError):
    """Requested session ID not found in service."""


class ServiceOSError(ServiceError, OSError):
    """System error raised from the service."""


class ServiceInitError(ServiceError, OSError):
    """Error raised if the service fails to initialize."""


class EnvironmentNotSupported(ServiceInitError):
    """Error raised if the runtime requirements for an environment are not
    met on the current system."""


class ServiceTransportError(ServiceError, OSError):
    """Error that is raised if communication with the service fails."""


class ServiceIsClosed(ServiceError, TypeError):
    """Error that is raised if trying to interact with a closed service."""
