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
import utils.db_utils as db_utils
import utils.parsing_utils as parsing_utils
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
        self.all_state_indices = db_utils.get_all_states(self.cursor, self.dataset_size)

    def get_full_db_length(self):
        if self.dataset_size == -1:
            return db_utils.get_database_size(self.cursor, self.table_name)
        else:
            return self.dataset_size

    def __getitem__(self, i):
        return self.get_instance(i)

    def get_instance(self, i):
        index = None
        if self.distribution_type == "train":
            index = self.train_indices[i]
        elif self.distribution_type == "dev":
            index = self.dev_indices[i]

        cursor = self.get_cursor()

        cur_state = self.all_state_indices[index]
        s = db_utils.get_observation_from_table(cursor, cur_state[3])
        # TODO: make this a more general reward getting function
        reward = s[0][1]

        programl = pickle.loads(zlib.decompress(s[0][3]))

        dgl_graph = parsing_utils.process_networkx_graph(programl, self.vocab)
        return {"reward": reward, "dgl_graph": dgl_graph}

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
