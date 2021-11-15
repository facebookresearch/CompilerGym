# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import dgl
import numpy as np
import torch
import torch.nn as nn


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_vocab_size,
        node_hidden_size,
        use_node_embedding=True,
        n_steps=1,
        n_etypes=3,
        n_message_passes=0,
        reward_dim=1,
        gnn_type="GatedGraphConv",
        heads=None,
        feat_drop=0.0,
        concat_intermediate=True,
    ):
        super(GNNEncoder, self).__init__()

        self.use_node_embedding = use_node_embedding
        self.node_hidden_size = node_hidden_size
        self.n_steps = n_steps
        self.n_etypes = n_etypes
        self.n_message_passes = n_message_passes
        self.reward_dim = reward_dim
        self.gnn_type = gnn_type
        self.heads = heads
        self.feat_drop = feat_drop
        self.concat_intermediate = concat_intermediate

        if self.use_node_embedding:
            self.node_embedding = nn.Embedding(node_vocab_size, node_hidden_size)

        embed_dim = self.node_hidden_size
        if self.gnn_type == "GatedGraphConv":
            self.ggcnn = nn.ModuleList(
                [
                    dgl.nn.pytorch.conv.GatedGraphConv(
                        in_feats=self.node_hidden_size,
                        out_feats=self.node_hidden_size,
                        n_steps=self.n_steps,
                        n_etypes=self.n_etypes,
                    )
                    for _ in range(self.n_message_passes)
                ]
            )
            if self.concat_intermediate:
                embed_dim = (self.n_message_passes + 1) * embed_dim
        else:
            raise NotImplementedError("")

        self.reward_predictor = nn.Sequential(
            nn.Linear(embed_dim, self.node_hidden_size),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size, self.reward_dim),
        )

        self.mse_loss = nn.MSELoss()

    def forward(self, g):
        with g.local_scope():
            self.featurize_nodes(g)

            res = g.ndata["feat"]
            if self.concat_intermediate:
                intermediate = [dgl.mean_nodes(g, "feat")]
            if self.gnn_type == "GatedGraphConv":
                for i, layer in enumerate(self.ggcnn):
                    res = layer(g, res, g.edata["flow"])
                    if self.concat_intermediate:
                        g.ndata["feat"] = res
                        intermediate.append(dgl.mean_nodes(g, "feat"))
            g.ndata["feat"] = res

            if self.concat_intermediate and self.gnn_type == "GatedGraphConv":
                graph_agg = torch.cat(intermediate, axis=1)
            else:
                graph_agg = dgl.mean_nodes(g, "feat")
        res = self.reward_predictor(graph_agg)
        return res, graph_agg

    def get_loss(self, g, labels, eps=0.0):
        """
        Loss function, scales the reward to the same loss function from
        R2D2 (https://openreview.net/pdf?id=r1lyTjAqYX). It also allows
        us to see the difference between the unscaled reward and its
        associated prediction
        """
        preds, _ = self.forward(g)
        preds = preds.squeeze(1)
        scaled_labels = rescale(labels, eps=eps)
        inv_scale_pred = inv_rescale(preds, eps=eps)

        return (
            self.mse_loss(preds, scaled_labels),
            self.mse_loss(inv_scale_pred, labels),
            ((labels - inv_scale_pred).abs() / labels).mean(),
        )

    def featurize_nodes(self, g):
        # This is very CompilerGym specific, can be rewritten for other tasks
        features = []
        if self.use_node_embedding:
            features.append(self.node_embedding(g.ndata["text_idx"]))

        g.ndata["feat"] = torch.cat(features)

    def get_edge_embedding(self, g):
        # TODO: this should can be for positional embeddings
        pass


def rescale(x, eps=1e-3):
    sign = get_sign(x)
    x_abs = get_abs(x)
    if isinstance(x, np.ndarray):
        return sign * (np.sqrt(x_abs + 1) - 1) + eps * x
    else:
        return sign * ((x_abs + 1).sqrt() - 1) + eps * x


def inv_rescale(x, eps=1e-3):
    sign = get_sign(x)
    x_abs = get_abs(x)
    if eps == 0:
        return sign * (x * x + 2.0 * x_abs)
    else:
        return sign * (
            (((1.0 + 4.0 * eps * (x_abs + 1.0 + eps)).sqrt() - 1.0) / (2.0 * eps)).pow(
                2
            )
            - 1.0
        )


def get_sign(x):
    if isinstance(x, np.ndarray):
        return np.sign(x)
    elif isinstance(x, torch.Tensor):
        return x.sign()
    else:
        raise NotImplementedError(f"Data type: {type(x)} is not implemented")


def get_abs(x):
    if isinstance(x, np.ndarray):
        return np.abs(x)
    elif isinstance(x, torch.Tensor):
        return x.abs()
    else:
        raise NotImplementedError(f"Data type: {type(x)} is not implemented")
