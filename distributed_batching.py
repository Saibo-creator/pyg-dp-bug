#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : distributed_batching.py
# @Date : 2023-04-11-13-23
# @Project: sim
# @AUTHOR : Saibo Geng
# @Desc :

import os
import pdb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset as Dataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5):
        super().__init__()

        self.dropout = dropout

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.convs.append(GINEConv(nn, train_eps=True))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, adj_t, batch):
        x = self.atom_encoder(x)
        edge_attr = adj_t.coo()[2]
        adj_t = adj_t.set_value(self.bond_encoder(edge_attr), layout='coo')

        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


def run(rank, world_size: int, dataset_name: str, root: str):

    dataset = Dataset(dataset_name, root,
                      pre_transform=T.ToSparseTensor(attr='edge_attr'))
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(dataset_name)

    train_dataset = dataset[split_idx['train']]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              sampler=train_sampler)
    pdb.set_trace()


if __name__ == '__main__':
    dataset_name = 'ogbg-molhiv'
    root = './data/OGB'

    # Download and process the dataset on main process.
    dataset = Dataset(dataset_name, root,
            pre_transform=T.ToSparseTensor(attr='edge_attr'))
    pdb.set_trace()

    """
    (Pdb) dataset
    PygGraphPropPredDataset(41127)
    (Pdb) dataset[0]
    Data(x=[19, 9], y=[1, 1], adj_t=[19, 19, 3, nnz=40], num_nodes=19)
    (Pdb) dataset[1]
    Data(x=[39, 9], y=[1, 1], adj_t=[39, 39, 3, nnz=88], num_nodes=39)
    (Pdb) dataset[2]
    Data(x=[21, 9], y=[1, 1], adj_t=[21, 21, 3, nnz=48], num_nodes=21)

    """

