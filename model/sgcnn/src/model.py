################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network model defintion
################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import (
    GCNConv,
    GlobalAttention,
    global_add_pool,
    NNConv,
    avg_pool_x,
    avg_pool,
    max_pool_x,
    GatedGraphConv,
)  # NOTE: maybe use the default version of GatedGraphConv, and make a PNET wrapper?
from torch_geometric.utils import (
    to_dense_batch,
    add_self_loops,
    remove_self_loops,
    normalized_cut,
    dense_to_sparse,
    is_undirected,
    to_undirected,
)
from torch_geometric.utils import (
    normalized_cut,
    scatter_,
    contains_self_loops,
    add_self_loops,
)
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import Batch
from ggcnn import GatedGraphConv, PotentialNetAttention
from torch.nn import init


# from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import coalesce

# NOTE: need to change this to add pool
from torch_geometric.nn.pool import avg_pool_x


def maybe_num_nodes(index, num_nodes=None):

    return index.max().item() + 1 if num_nodes is None else num_nodes


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


class PotentialNetPropagation(torch.nn.Module):
    def __init__(
        self,
        feat_size=19,
        gather_width=64,
        k=2,
        neighbor_threshold=None,
        output_pool_result=False,
        bn_track_running_stats=False,
    ):
        super(PotentialNetPropagation, self).__init__()
        assert neighbor_threshold is not None

        self.neighbor_threshold = neighbor_threshold
        self.bn_track_running_stats = bn_track_running_stats
        self.edge_attr_size = 1

        self.k = k
        self.gather_width = gather_width
        self.feat_size = feat_size
        self.edge_network_nn = nn.Sequential(
            nn.Linear(self.edge_attr_size, int(self.feat_size / 2)),
            nn.Softsign(),
            nn.Linear(int(self.feat_size / 2), self.feat_size),
            nn.Softsign(),
        )

        self.edge_network = NNConv(
            self.feat_size,
            self.edge_attr_size * self.feat_size,
            nn=self.edge_network_nn,
            root_weight=True,
            aggr="add",
        )
        self.gate = GatedGraphConv(
            self.feat_size, self.k, edge_network=self.edge_network
        )

        self.attention = PotentialNetAttention(
            net_i=nn.Sequential(
                nn.Linear(self.feat_size * 2, self.feat_size),
                nn.Softsign(),
                nn.Linear(self.feat_size, self.gather_width),
                nn.Softsign(),
            ),
            net_j=nn.Sequential(
                nn.Linear(self.feat_size, self.gather_width), nn.Softsign()
            ),
        )
        self.output_pool_result = output_pool_result
        if self.output_pool_result:
            self.global_add_pool = global_add_pool

    def forward(self, data, edge_index, edge_attr):

        # propagtion
        h_0 = data
        h_1 = self.gate(h_0, edge_index, edge_attr)
        h_1 = self.attention(h_1, h_0)

        return h_1


class GraphThreshold(torch.nn.Module):
    def __init__(self, t):
        super(GraphThreshold, self).__init__()
        self.t = nn.Parameter(t, requires_grad=True).cuda()

    def filter_adj(self, row, col, edge_attr, mask):
        mask = mask.squeeze()
        return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

    def forward(self, edge_index, edge_attr):
        """Randomly drops edges from the adjacency matrix
        :obj:`(edge_index, edge_attr)` with propability :obj:`p` using samples from
        a Bernoulli distribution.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_attr (Tensor): Edge weights or multi-dimensional
                edge features. (default: :obj:`None`)
            force_undirected (bool, optional): If set to :obj:`True`, forces undirected output.
            (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        """

        N = maybe_num_nodes(edge_index, None)
        row, col = edge_index

        mask = edge_attr <= self.t

        row, col, edge_attr = self.filter_adj(row, col, edge_attr, mask)

        edge_index = torch.stack(
            [torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0
        )
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

        return edge_index, edge_attr


class PotentialNetFullyConnected(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PotentialNetFullyConnected, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / 1.5)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 1.5), int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 2), out_channels),
        )

    def forward(self, data, return_hidden_feature=False):

        if return_hidden_feature:
            return self.output[:-2](data), self.output[:-4](data), self.output(data)
        else:
            return self.output(data)


class PotentialNetParallel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        covalent_gather_width=128,
        non_covalent_gather_width=64,
        covalent_k=1,
        non_covalent_k=1,
        covalent_neighbor_threshold=None,
        non_covalent_neighbor_threshold=None,
    ):
        super(PotentialNetParallel, self).__init__()

        assert (
            covalent_neighbor_threshold is not None
            and non_covalent_neighbor_threshold is not None
        )

        self.covalent_neighbor_threshold = GraphThreshold(
            torch.ones(1).cuda() * covalent_neighbor_threshold
        )
        self.non_covalent_neighbor_threshold = GraphThreshold(
            torch.ones(1).cuda() * non_covalent_neighbor_threshold
        )  # need to add params for upper/lower covalent/non_covalent_t
        self.global_add_pool = global_add_pool

        self.covalent_propagation = PotentialNetPropagation(
            feat_size=in_channels,
            gather_width=covalent_gather_width,
            neighbor_threshold=self.covalent_neighbor_threshold,
            k=covalent_k,
        )

        self.non_covalent_propagation = PotentialNetPropagation(
            feat_size=covalent_gather_width,
            gather_width=non_covalent_gather_width,
            neighbor_threshold=self.non_covalent_neighbor_threshold,
            k=non_covalent_k,
        )

        self.global_add_pool = global_add_pool

        self.output = PotentialNetFullyConnected(
            non_covalent_gather_width, out_channels
        )

    def forward(self, data, return_hidden_feature=False):

        data.x = data.x.cuda()
        data.edge_attr = data.edge_attr.cuda()
        data.edge_index = data.edge_index.cuda()
        data.batch = data.batch.cuda()

        # make sure that we have undirected graph
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)

        # make sure that nodes can propagate messages to themselves
        if not contains_self_loops(data.edge_index):
            data.edge_index, data.edge_attr = add_self_loops(
                data.edge_index, data.edge_attr.view(-1)
            )

        """
        # now select the top 5 closest neighbors to each node


        dense_adj = sparse_to_dense(edge_index=data.edge_index, edge_attr=data.edge_attr)

        #top_k_vals, top_k_idxs = torch.topk(dense_adj, dim=0, k=5, largest=False)

        #dense_adj = torch.zeros_like(dense_adj).scatter(1, top_k_idxs, top_k_vals)
        
        dense_adj[dense_adj == 0] = 10000   # insert artificially large values for 0 valued entries that will throw off NN calculation
        top_k_vals, top_k_idxs = torch.topk(dense_adj, dim=1, k=15, largest=False)
        dense_adj = torch.zeros_like(dense_adj).scatter(1, top_k_idxs, top_k_vals)
        
        data.edge_index, data.edge_attr = dense_to_sparse(dense_adj)
        """

        # covalent_propagation
        # add self loops to enable self propagation
        covalent_edge_index, covalent_edge_attr = self.covalent_neighbor_threshold(
            data.edge_index, data.edge_attr
        )
        (
            non_covalent_edge_index,
            non_covalent_edge_attr,
        ) = self.non_covalent_neighbor_threshold(data.edge_index, data.edge_attr)

        # covalent_propagation and non_covalent_propagation
        covalent_x = self.covalent_propagation(
            data.x, covalent_edge_index, covalent_edge_attr
        )
        non_covalent_x = self.non_covalent_propagation(
            covalent_x, non_covalent_edge_index, non_covalent_edge_attr
        )

        # zero out the protein features then do ligand only gather...hacky sure but it gets the job done
        non_covalent_ligand_only_x = non_covalent_x
        non_covalent_ligand_only_x[data.x[:, 14] == -1] = 0
        pool_x = self.global_add_pool(non_covalent_ligand_only_x, data.batch)

        # fully connected and output layers
        if return_hidden_feature:
            # return prediction and atomistic features (covalent result, non-covalent result, pool result)

            avg_covalent_x, _ = avg_pool_x(data.batch, covalent_x, data.batch)
            avg_non_covalent_x, _ = avg_pool_x(data.batch, non_covalent_x, data.batch)

            fc0_x, fc1_x, output_x = self.output(pool_x, return_hidden_feature=True)

            return avg_covalent_x, avg_non_covalent_x, pool_x, fc0_x, fc1_x, output_x
        else:
            return self.output(pool_x)
