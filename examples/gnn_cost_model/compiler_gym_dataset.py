# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sqlite3
import collections
from sqlite3 import Error
import pickle
import zlib
import time
import os
from concurrent.futures import ProcessPoolExecutor

import dgl
import numpy as np
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import torch
from torch.utils.data import DataLoader


class CompilerGymDataset(DGLDataset):
    def __init__(
        self,
        filepath,
        num_workers=64,
        max_len_nodes=50000,
        input_key="dgl_graph",
        output_key="reward",
        table_name="Observations",
        train_prop=0.8,
        vocab=None,
        dataset_size=-1,
    ):
        """
        The class loads a CompilerGym Database which contains 'States' and 'Observations'
        as tables. The tables contain the necessary information for doing supervised learning.
        This class handles all of the underlying structure including differentiating between
        training and dev, creating the 'dgl graph', and colating individual graphs into a larger
        graph, which is used for training.
        Inputs:
            - filepath: the path to the dataset
            - num_wokers: number of workers used to fetch the instances
            - max_len_nodes: maximum number of nodes in the grpah
            - input_key: the key that we save to the input observation
            - output_key: the key that we want to generate supervised loss off of
            - table_name: the table name in the database that has the primary keys
            - train_prop: proportion of training instances
            - vocab: the vocab mapping text to integer indices of a embedding table
            - dataset_size: size of the dataset we want to use, default -1 means use the whole datbase
        """
        self.filepath = filepath
        self.num_workers = num_workers
        self.max_len_nodes = max_len_nodes

        self.graph_key = input_key
        self.output_key = output_key
        self.table_name = table_name

        self.train_prop = train_prop
        self.vocab = vocab
        self.dataset_size = dataset_size

        self.distribution_type = "train"

        print("using filepath: ", self.filepath)

        super().__init__(name="CopmilerGym")

    def process(self):
        """
        Called during initialization of the class and initializes the underlying
        functions needed for supervised learning
        """
        self.initialize_database()

    def initialize_database(self):
        print("using: ", self.filepath, " as dataset")
        self.cursor = self.get_cursor()

        self.train_size = int(self.train_prop * self.get_full_db_length())
        self.dev_size = self.get_full_db_length() - self.train_size

        self.select_distribution_indices()
        self.get_observation_indices()

        print("intialized database: ", self.filepath)

    def select_distribution_indices(self):
        total_size = self.get_full_db_length()

        self.all_indices = set(range(total_size))
        self.train_indices = np.random.choice(
            total_size, size=self.train_size, replace=False
        )
        self.dev_indices = list(self.all_indices - set(self.train_indices))

        assert len(self.train_indices) == self.train_size
        assert len(self.dev_indices) == self.dev_size

    def get_observation_indices(self):
        self.all_state_indices = get_all_states(self.cursor, self.dataset_size)

    def get_full_db_length(self):
        if self.dataset_size == -1:
            return get_database_size(self.cursor, self.table_name)
        else:
            return self.dataset_size

    def __getitem__(self, i):
        return self.get_instance(i)

    def get_instance(self, i):
        """
        Given an index (i), determined by the length of the current dataset ('train', 'dev')
        get the desired instance
        """
        index = None
        if self.distribution_type == "train":
            index = self.train_indices[i]
        elif self.distribution_type == "dev":
            index = self.dev_indices[i]

        cursor = self.get_cursor()

        cur_state = self.all_state_indices[index]
        s = get_observation_from_table(cursor, cur_state[3])
        # This reward is hardcoded right now to be the number of instruction
        # counts in the given LLVM-IR graph.
        reward = s[0][1]

        programl = pickle.loads(zlib.decompress(s[0][3]))

        dgl_graph = process_networkx_graph(programl, self.vocab)
        return {self.output_key: reward, self.graph_key: dgl_graph}

    def __len__(self):
        if self.distribution_type == "train":
            return self.train_size
        elif self.distribution_type == "dev":
            return self.dev_size

    def collate_fn(self, samples):
        samples = [sample for sample in samples if sample is not None]
        # Takes a list of graphs and makes it into one big graph that dgl operates on
        ret = None
        if samples:
            dgl_graph = dgl.batch([sample[self.graph_key] for sample in samples])
            reward = [sample[self.output_key] for sample in samples]
            ret = (dgl_graph, reward)
        return ret

    def set_distribution_type(self, dist_type):
        assert dist_type in ["train", "dev"]
        self.distribution_type = dist_type

    def get_cursor(self):
        connection = sqlite3.connect(self.filepath)
        return connection.cursor()


def get_database_size(cursor, table):
    return cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]


def get_all_states(cursor, db_size):
    if db_size == -1:
        cursor.execute("SELECT * from States")
    else:
        cursor.execute(f"SELECT * from States LIMIT {db_size}")

    return cursor.fetchall()


def get_observation_from_table(cursor, hash):
    """
    Gets the observation for a state_id from a given database
    Inputs:
        - cursor: the db cursor
        - state_id: the state_id we want (primary key in the table)
    """
    cursor.execute(f"SELECT * from Observations where state_id = '{hash}'")
    return cursor.fetchall()


def process_networkx_graph(
    graph,
    vocab,
    node_feature_list=["text", "type"],
    edge_feature_list=["flow", "position"],
):
    """
    Handles all of the requirements of taking a networkx graph and converting it into a
    dgl graph
    Inputs:
        - graph: the networkx graph
        - vocab: the vocabulary, a mapping from word to index.
        - node_feature_list: a list of textual features from the networkx node that we want to make sure
            are featurizable into a vector.
        - edge_feature_list: a list of textual features from the networkx edges that we want to make sure
            are featurizable into a vector.
    """
    update_graph_with_vocab(graph.nodes, node_feature_list, vocab)
    update_graph_with_vocab(graph.edges, edge_feature_list, vocab)

    dgl_graph = fast_networkx_to_dgl(graph)
    return dgl_graph


def fast_networkx_to_dgl(
    graph, node_attrs=["text_idx", "type"], edge_attrs=["flow", "position"]
):
    """
    Takes a networkx graph and its given node attributes and edge attributes
    and converts it corresponding dgl graph
    Inputs:
        - graph: the networkx graph
        - node_attrs: node attributes to convert
        - edge_attrs: edge attributes to convert
    """

    edges = [edge for edge in graph.edges()]
    dgl_graph = dgl.graph(edges, num_nodes=graph.number_of_nodes())

    for feat in edge_attrs:
        edge_assigns = torch.tensor(
            [val[-1] for val in graph.edges(data=feat)], dtype=torch.int64
        )
        dgl_graph.edata[feat] = edge_assigns

    for feat in node_attrs:
        node_assigns = torch.tensor(
            [val[-1] for val in graph.nodes(data=feat)], dtype=torch.int64
        )
        dgl_graph.ndata[feat] = node_assigns

    return dgl_graph


def update_graph_with_vocab(graph_fn, features, vocab):
    """
    Given a networkx attribute (function) and features update it with a vocab if possible.
    If it cannot be updated, the features should already be numerical features.
    Inputs:
        - graph_fn: a networkx graph function (describing nodes or edges)
        - features: the feature from the function that should be updated
        - vocab: A dict mapping text to int
    """

    for feature_name in features:
        curr_vocab = None
        if feature_name in vocab:
            curr_vocab = vocab[feature_name]
        for graph_item in graph_fn(data=feature_name):
            feature = graph_item[-1]
            idx = graph_item[0]

            if feature_name in vocab:
                assert feature in curr_vocab
                update_networkx_feature(
                    graph_fn, idx, f"{feature_name}_idx", curr_vocab[feature]
                )
            else:
                assert isinstance(
                    feature, int
                ), f"{(feature_name, feature)} is not an int"


def update_networkx_feature(graph_fn, idx, feature_name, feature):
    graph_fn[idx][feature_name] = feature
