"""
modified from https://github.com/graphdeeplearning/benchmarking-gnns
"""

import torch
from torch import nn as nn
from torch.nn import functional as F

import dgl
import dgl.nn as dglnn
from dgl import function as dglfn


class GatedGCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):
        with g.local_scope():
            h_in = h  # for residual connection
            e_in = e  # for residual connection

            g.ndata['h'] = h
            g.ndata['Ah'] = self.A(h)
            g.ndata['Bh'] = self.B(h)
            g.ndata['Dh'] = self.D(h)
            g.ndata['Eh'] = self.E(h)
            g.edata['e'] = e
            g.edata['Ce'] = self.C(e)

            g.apply_edges(dglfn.u_add_v('Dh', 'Eh', 'DEh'))
            g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
            g.edata['sigma'] = torch.sigmoid(g.edata['e'])
            g.update_all(dglfn.u_mul_e('Bh', 'sigma', 'm'), dglfn.sum('m', 'sum_sigma_h'))
            g.update_all(dglfn.copy_e('sigma', 'm'), dglfn.sum('m', 'sum_sigma'))
            g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
            # g.update_all(self.message_func,self.reduce_func)
            h = g.ndata['h']  # result of graph convolution
            e = g.edata['e']  # result of graph convolution

            if self.batch_norm:
                h = self.bn_node_h(h)  # batch normalization
                e = self.bn_node_e(e)  # batch normalization

            h = F.relu(h)  # non-linear activation
            e = F.relu(e)  # non-linear activation

            if self.residual:
                h = h_in + h  # residual connection
                e = e_in + e  # residual connection

            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)

        y = self.FC_layers[self.L](y)
        return y

class GGCNEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, out_dim, n_layers, dropout, readout_pooling, batch_norm, residual):
        super().__init__()

        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   batch_norm, residual) for _ in range(n_layers)])

        self.pool = {'sum': dglnn.SumPooling(),
                     'mean': dglnn.AvgPooling(),
                     'max': dglnn.MaxPooling()}[readout_pooling]
        self.MLP_layer = MLP(hidden_dim, out_dim)

    def forward(self, g, h, e):
        h, e = self.forward_feature(g, h, e)
        hg = self.readout(g, h)

        return hg

    def forward_feature(self, g: dgl.DGLGraph, h, e):
        for conv in self.layers:
            h, e = conv(g, h, e)
        return h, e

    def readout(self, g, h):
        hg = self.pool(g, h)
        return self.MLP_layer(hg)
