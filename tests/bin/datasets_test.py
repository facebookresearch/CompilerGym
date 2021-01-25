# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end tests for //compiler_gym/bin:benchmarks."""
import json
import os
import tarfile
import tempfile
from pathlib import Path

import gym
import pytest
from absl import flags

from compiler_gym.bin.datasets import main
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS


def run_main(*args):
    FLAGS.unparse_flags()
    FLAGS(["argv"] + list(args))
    return main(["argv0"])


@pytest.fixture(scope="function")
def site_data() -> Path:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        os.environ["COMPILER_GYM_SITE_DATA"] = str(d)
        yield d


def test_llvm_site_directories_are_created(site_data):
    run_main("--env=llvm-v0")
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks").is_dir()
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks.inactive").is_dir()


def test_llvm_activate_non_existent_dataset(site_data):
    del site_data
    invalid = "nonexistent"
    with pytest.raises(ValueError) as ctx:
        run_main("--env=llvm-v0", "--activate", invalid)
    assert f"Inactive dataset not found: {invalid}" == str(ctx.value)


def test_llvm_deactivate_non_existent_dataset(site_data):
    del site_data
    invalid = "nonexistent"
    run_main("--env=llvm-v0", "--deactivate", invalid)


def test_llvm_activate_invalid_metadata_file(site_data):
    dataset = "foo"

    # Make an inactive dataset.
    (site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / dataset).mkdir(
        parents=True
    )
    (site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / dataset / "file").touch()
    (site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / f"{dataset}.json").touch()

    # Activate the dataset.
    with pytest.raises(OSError) as ctx:
        run_main("--env=llvm-v0", "--activate", dataset)

    assert "Failed to read dataset metadata file" in str(ctx.value)


def test_llvm_activate_deactivate_dataset(site_data):
    dataset = "foo"

    # Make an inactive dataset.
    (site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / dataset).mkdir(
        parents=True
    )
    (site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / dataset / "file").touch()
    with open(
        str(site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / f"{dataset}.json"),
        "w",
    ) as f:
        json.dump(
            {
                "name": dataset,
                "license": "",
                "file_count": 1,
                "size_bytes": 0,
            },
            f,
        )

    # Activate the dataset.
    run_main("--env=llvm-v0", "--activate", dataset)
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / dataset / "file").is_file()
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / f"{dataset}.json").is_file()
    assert not (
        site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / dataset / "file"
    ).is_file()
    assert not (
        site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / f"{dataset}.json"
    ).is_file()

    # Deactivate the dataset.
    run_main("--env=llvm-v0", "--deactivate", dataset)
    assert not (
        site_data / "llvm/10.0.0/bitcode_benchmarks" / dataset / "file"
    ).is_file()
    assert not (
        site_data / "llvm/10.0.0/bitcode_benchmarks" / f"{dataset}.json"
    ).is_file()
    assert (
        site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / dataset / "file"
    ).is_file()
    assert (
        site_data / "llvm/10.0.0/bitcode_benchmarks.inactive" / f"{dataset}.json"
    ).is_file()


def test_llvm_download_url_404(site_data):
    del site_data
    invalid_url = "https://facebook.com/not/a/valid/url"
    with pytest.raises(OSError) as ctx:
        run_main("--env=llvm-v0", "--download", invalid_url)
    assert str(ctx.value) == f"GET returned status code 404: {invalid_url}"


def test_llvm_download_invalid_protocol(site_data):
    del site_data
    invalid_url = "invalid://facebook.com"
    with pytest.raises(OSError) as ctx:
        run_main("--env=llvm-v0", "--download", invalid_url)
    assert invalid_url in str(ctx.value)


def test_llvm_download_name(site_data):
    dataset = "npb-v0"
    assert not (site_data / "llvm/10.0.0/bitcode_benchmarks" / dataset).is_dir()

    run_main("--env=llvm-v0", "--download", dataset)

    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / dataset).is_dir()
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / f"{dataset}.json").is_file()


def test_llvm_download_url(site_data):
    dataset = "npb-v0"
    assert not (site_data / "llvm/10.0.0/bitcode_benchmarks" / dataset).is_dir()

    env = gym.make("llvm-v0")
    try:
        url = env.available_datasets[dataset].url
    finally:
        env.close()
    run_main("--env=llvm-v0", "--download", url)

    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / dataset).is_dir()
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / f"{dataset}.json").is_file()


def test_llvm_download_local_path(site_data):
    with tempfile.TemporaryDirectory() as d:
        # Create a dataset archive with a single file.
        d = Path(d)

        (d / "a").touch()
        with open(str(d / "testdataset.json"), "w") as f:
            json.dump(
                {
                    "name": "testdataset",
                    "license": "",
                    "file_count": 1,
                    "size_bytes": 0,
                },
                f,
            )

        with tarfile.open(str(d / "dataset.tar.bz2"), "x:bz2") as f:
            f.add(d / "a", "testdataset/a")
            f.add(d / "testdataset.json", "testdataset.json")

        run_main("--env=llvm-v0", "--download", f"file:///{d / 'dataset.tar.bz2'}")

    assert (
        site_data / "llvm/10.0.0/bitcode_benchmarks" / "testdataset" / "a"
    ).is_file()
    assert (site_data / "llvm/10.0.0/bitcode_benchmarks" / "testdataset.json").is_file()


def test_llvm_download_local_path_not_found(site_data):
    del site_data
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        path = d / "invalid"

        with pytest.raises(FileNotFoundError) as ctx:
            run_main("--env=llvm-v0", "--download", f"file:///{path}")

        assert str(ctx.value) == f"File not found: {path}"


def test_llvm_download_local_path_invalid_file_type(site_data):
    del site_data
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        path = d / "invalid"
        path.touch()

        with pytest.raises(tarfile.ReadError) as ctx:
            run_main("--env=llvm-v0", "--download", f"file:///{path}")

        assert str(ctx.value) == "not a bzip2 file"


if __name__ == "__main__":
    _test_main()
