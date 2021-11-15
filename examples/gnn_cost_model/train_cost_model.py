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
import pickle
import time

import numpy as np
import torch
from absl import app, flags
from compiler_gym_dataset import CompilerGymDataset
from model import GNNEncoder
from torch.utils.data import DataLoader

flags.DEFINE_list(
    "flags",
    [
        "-dataset-size",
        "-num_epoch",
        "-batch_size",
    ],
    "List of possible flags for training",
)
flags.DEFINE_integer(
    "dataset_size", -1, "How large should the dataset be, -1 if no constraint"
)
flags.DEFINE_integer("num_epoch", 100, "Number of epochs for training")
flags.DEFINE_integer("batch_size", 4, "Number of epochs for training")

FLAGS = flags.FLAGS


def get_dataset(file_dir, batch_size, vocab, num_workers=16, dataset_size=-1):
    dataset = CompilerGymDataset(file_dir, vocab=vocab, dataset_size=dataset_size)
    train_data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return dataset, train_data_loader


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
    print(
        f"Epoch num {epoch_num} training {train} took: {times}, loss: {avg_loss}, unscaled: {avg_unscaled}, grad_clip {avg_grad_clip}"
    )

    return avg_loss, avg_unscaled, avg_grad_clip


def train(dataset, data_loader, model, num_epoch, device):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epoch):
        start_time = time.time()

        dataset.set_distribution_type("train")
        print("Running training with length of: ", len(dataset))
        dataset_looper(epoch, data_loader, model, device, optimizer)
        dataset.set_distribution_type("dev")
        print("Running dev with length of: ", len(dataset))
        dataset_looper(epoch, data_loader, model, device, train=False)

        print("took: ", time.time() - start_time, "for an epoch")


def main(argv):
    """Main entry point."""
    del argv  # unused
    root_pth = (
        "/checkpoint/bcui/compiler_gym/replay_dataset/frozen/2021-06-15-cbench-v1.db"
    )
    node_vocab_pth = "/checkpoint/bcui/compiler_gym/replay_dataset/cbench/06-23/2021-06-23-cbench-vocab.pkl"
    batch_size = FLAGS.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab_fp = open(node_vocab_pth, "rb")
    vocab = pickle.load(vocab_fp)
    model = GNNEncoder(len(vocab), 64)

    # This is required to get the vocab into the right state
    # as the vocab is over all nodes of the graph
    vocab = {"text": vocab}

    model.to(device)
    print(model)

    dataset, dataset_loader = get_dataset(
        root_pth, batch_size, vocab, dataset_size=FLAGS.dataset_size
    )

    train(dataset, dataset_loader, model, FLAGS.num_epoch, device)
    vocab_fp.close()


if __name__ == "__main__":
    app.run(main)
