# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class DownloadFailed(IOError):
    """Error thrown if a download fails."""


class TooManyRequests(DownloadFailed):
    """Error thrown by HTTP 429 response."""
