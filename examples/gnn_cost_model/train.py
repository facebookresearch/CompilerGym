# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This module trains a cost model with a GNN on a LLVM-IR transition database
predicting some output reward (the default is instruction count).

Example usage:

    $ python train_cost_model.py --num_epoch 10 --batch_size 16 --dataset_size 64
"""
import collections
import io
import logging
import pickle
import sys
import tarfile
import time
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from absl import app, flags
from fasteners import InterProcessLock
from torch.utils.data import DataLoader

import compiler_gym.util.flags.nproc  # noqa flag definition
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, transient_cache_path
from compiler_gym.util.timer import Timer, humanize_duration

from .compiler_gym_dataset import CompilerGymDataset
from .model import GNNEncoder

flags.DEFINE_integer(
    "dataset_size", -1, "How large should the dataset be, -1 if no constraint"
)
flags.DEFINE_integer("num_epoch", 100, "Number of epochs for training")
flags.DEFINE_integer("batch_size", 4, "Number of epochs for training")
flags.DEFINE_string(
    "db",
    "https://dl.fbaipublicfiles.com/compiler_gym/state_transition_dataset/2021-11-15-csmith.tar.bz2",
    "URL of the dataset to use.",
)
flags.DEFINE_string(
    "db_sha256",
    "0b101a17fdbb1851f38ca46cc089b0026eb740e4055a4fe06b4c899ca87256a2",
    "SHA256 checksum of the dataset database.",
)
flags.DEFINE_string(
    "vocab_db",
    "https://dl.fbaipublicfiles.com/compiler_gym/state_transition_dataset/2021-11-15-vocab.tar.bz2",
    "URL of the vocabulary database to use.",
)
flags.DEFINE_string(
    "vocab_db_sha256",
    "af7781f57e6ef430c561afb045fc03693783e668b21826b32234e9c45bd1882c",
    "SHA256 checksum of the vocabulary database.",
)
flags.DEFINE_string(
    "device", "cuda:0" if torch.cuda.is_available() else "cpu", "The device to run on."
)

FLAGS = flags.FLAGS


logger = logging.getLogger(__name__)

_DB_DOWNLOAD_LOCK = Lock()


def dataset_looper(epoch_num, data_loader, model, device, optimizer=None, train=True):
    times = collections.defaultdict(float)
    losses = []
    unscaled_mse = []
    epoch_grad_clip = []
    t1 = time.time()
    for data in data_loader:
        if data is None:
            continue
        graph, labels = data

        times["get_data"] += time.time() - t1
        t1 = time.time()

        labels = torch.Tensor(labels).to(device)
        graph = graph.to(device)
        loss, unscaled, _ = model.get_loss(graph, labels)
        losses.append(loss.cpu().data.numpy())
        unscaled_mse.append(unscaled.cpu().data.numpy())
        times["model_forward"] += time.time() - t1
        t1 = time.time()

        if train:
            optimizer.zero_grad()
            loss.backward()
            grad_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=400.0
            )
            epoch_grad_clip.append(grad_clip.cpu().data.numpy())
            optimizer.step()

            times["model_backward"] += time.time() - t1
            t1 = time.time()
    avg_loss, avg_unscaled = (
        np.mean(losses),
        np.mean(unscaled_mse),
    )
    avg_grad_clip = None
    if train:
        avg_grad_clip = np.mean(epoch_grad_clip)
    times = ", ".join(f"{k}: {humanize_duration(v)}" for k, v in times.items())
    print(
        f"  Epoch {epoch_num + 1} {'training' if train else 'validation'} took: "
        f"{{ {times} }}, loss: {avg_loss}, unscaled: {avg_unscaled}, "
        f"grad_clip {avg_grad_clip}"
    )

    return avg_loss, avg_unscaled, avg_grad_clip


def train(dataset, data_loader, model, num_epoch, device):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epoch):
        with Timer(f"Epoch {epoch + 1} of {num_epoch} ({(epoch + 1) / num_epoch:.1%})"):
            dataset.set_distribution_type("train")
            dataset_looper(epoch, data_loader, model, device, optimizer)
            dataset.set_distribution_type("dev")
            dataset_looper(epoch, data_loader, model, device, train=False)


def download_and_unpack_database(db: str, sha256: str) -> Path:
    """Download the given database, unpack it to the local filesystem, and
    return the path.
    """
    local_dir = cache_path(f"state_transition_dataset/{sha256}")
    with _DB_DOWNLOAD_LOCK, InterProcessLock(
        transient_cache_path(".state_transition_database_download.LOCK")
    ):
        if not (local_dir / ".installed").is_file():
            tar_data = io.BytesIO(download(db, sha256))

            local_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Unpacking database to %s ...", local_dir)
            with tarfile.open(fileobj=tar_data, mode="r:bz2") as arc:
                arc.extractall(str(local_dir))

            (local_dir / ".installed").touch()

    unpacked = [f for f in local_dir.iterdir() if f.name != ".installed"]
    if len(unpacked) != 1:
        print(
            f"fatal: Archive {db} expected to contain one file, contains: {len(unpacked)}",
            file=sys.stderr,
        )

    return unpacked[0]


def main(argv):
    """Main entry point."""
    del argv  # unused

    node_vocab_pth = download_and_unpack_database(
        db=FLAGS.vocab_db, sha256=FLAGS.vocab_db_sha256
    )
    root_pth = download_and_unpack_database(db=FLAGS.db, sha256=FLAGS.db_sha256)

    with open(node_vocab_pth, "rb") as f:
        vocab = pickle.load(f)

    model = GNNEncoder(
        # Add one to the vocab size to accomodate for the out-of-vocab element.
        node_vocab_size=len(vocab) + 1,
        node_hidden_size=64,
    )

    # This is required to get the vocab into the right state
    # as the vocab is over all nodes of the graph
    vocab = {"text": vocab}

    model.to(FLAGS.device)
    print(model)

    dataset = CompilerGymDataset(root_pth, vocab=vocab, dataset_size=FLAGS.dataset_size)
    dataset_loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.nproc,
        collate_fn=dataset.collate_fn,
    )

    train(dataset, dataset_loader, model, FLAGS.num_epoch, FLAGS.device)


if __name__ == "__main__":
    app.run(main)
