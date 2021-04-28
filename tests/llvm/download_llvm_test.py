# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compiler_gym.third_party import llvm
from compiler_gym.util.runfiles_path import site_data_path
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def test_download_llvm_threaded_load_test(temporary_environ, tmpwd: Path, mocker):
    """A load test for download_llvm_files() that checks that redundant
    downloads are not performed when multiple simultaneous calls to
    download_llvm_files() are issued.
    """
    mocker.spy(llvm, "_download_llvm_files")
    mocker.spy(llvm, "download")

    # Force the LLVM download function to run.
    llvm._LLVM_DOWNLOADED = False

    # Force a temporary new site data path and sanity check it.
    temporary_environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd)
    assert str(site_data_path(".")).endswith(str(tmpwd))

    # Perform a bunch of concurrent calls to download_llvm_files().
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(llvm.download_llvm_files) for _ in range(100)]
        for future in futures:
            future.result()

    # For debugging in case of error.
    print("Downloads:", llvm._download_llvm_files.call_count)  # pylint: disable
    for root, _, filenames in os.walk(tmpwd):
        print(root)
        for filename in filenames:
            print(Path(root) / filename)

    # Check that the files were unpacked.
    assert (tmpwd / "llvm-v0" / "LICENSE").is_file()
    assert (tmpwd / "llvm-v0" / "bin" / "clang").is_file()

    # Check that the underlying download implementation was only called a single
    # time.
    assert llvm._download_llvm_files.call_count == 1  # pylint: disable
    assert llvm.download.call_count == 1


if __name__ == "__main__":
    main()
