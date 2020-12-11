################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Gated Graph Convolutional layer defintion (adapted from https://github.com/rusty1s/pytorch_geometric/blob/fddee1288d3327d96eec9960ab89841d37780d66/torch_geometric/nn/conv/gated_graph_conv.py)
################################################################################


import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, scatter_
#from torch_geometric.utils import softmax
#from torch_scatter import scatter as scatter_
from torch_geometric.nn.inits import uniform, reset


# TODO: is this defined correctly for batches?
class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, out_channels, num_layers, edge_network, aggr="add", bias=True):
        super(GatedGraphConv, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.edge_network = (
            edge_network  # TODO: make into a list of neural networks for each edge_attr
        )

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    # TODO: remove none defautl for edge_attr
    def forward(self, x, edge_index, edge_attr):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        assert h.size(1) <= self.out_channels

        # if input size is < out_channels, pad input with 0s to equal the sizes
        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])

            # source = h[edge_index[0].long()]
            # sink = h[edge_index[1].long()]
            # edge_feat = torch.cat([sink, edge_attr.view(-1, 1)], dim=1)

            # # TODO: edge_feat is defined over number of edges, while h is defined over num nodes dimension, convert back to node space by summing the features for all neighbors for each node
            # h += self.edge_network(h, edge_index, edge_attr)  # TODO: find a better network than NNConv

            # m = torch.matmul(h, self.weight[i])

            m = self.propagate(edge_index=edge_index, x=h, aggr="add")
            h = self.rnn(m, h)

        return h

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out

    def __repr__(self):
        return "{}({}, num_layers={})".format(
            self.__class__.__name__, self.out_channels, self.num_layers
        )


class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),
    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.
    Args:
        gate_nn (nn.Sequential): Neural network
            :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to \mathbb{R}`.
        nn (nn.Sequential, optional): Neural network
            :math:`h_{\mathbf{\Theta}}`. (default: :obj:`None`)
    """

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, size)
        out = scatter_("add", gate * x, batch, size)

        return out

    def __repr__(self):
        return "{}(gate_nn={}, nn={})".format(
            self.__class__.__name__, self.gate_nn, self.nn
        )


class PotentialNetAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(PotentialNetAttention, self).__init__()

        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        return torch.nn.Softmax(dim=1)(
            self.net_i(torch.cat([h_i, h_j], dim=1))
        ) * self.net_j(h_j)
