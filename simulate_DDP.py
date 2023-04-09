import os
import os.path as osp
import time
from typing import List

import torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml
from torch.utils.data import DistributedSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.nn.conv import HANConv
from torch_geometric.utils.unbatch import unbatch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fake_dataset import FakeHeteroDataset


class Net(torch.nn.Module):
    def __init__(self, n_layer, metadata, num_classes):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(n_layer):
            self.convs.append(HANConv(in_channels=128, out_channels=128,
                                      heads=4,
                                      metadata=metadata))
        self.lin_dict = torch.nn.ModuleDict()
        self.lin = torch.nn.Linear(128, num_classes)

    def forward(self, graph):

        if isinstance(graph, List):

            graph = Batch.from_data_list(graph)
        else:
            graph = graph

        for i, conv in enumerate(self.convs):
            nodes_repr_dict = conv(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)[0]
            for k in nodes_repr_dict.keys():
                graph.x_dict[k] = nodes_repr_dict[k]

        x_tuples = unbatch(graph["v0"]["x"], graph["v0"]["batch"])
        x_select = torch.stack([x[0] for x in x_tuples])
        return F.log_softmax(self.lin(x_select), dim=1)

def run(rank, world_size: int, dataset_name: str, root: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              sampler=train_sampler)

    torch.manual_seed(12345)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()


    for epoch in range(1, 51):
        model.train()

        total_loss = torch.zeros(2).to(rank)
        for data in train_loader:
            data = data.to(rank)
            optimizer.zero_grad()
            logits = model(data.x, data.adj_t, data.batch)
            loss = criterion(logits, data.y.to(torch.float))
            loss.backward()
            optimizer.step()
            total_loss[0] += float(loss) * logits.size(0)
            total_loss[1] += data.num_graphs

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        loss = float(total_loss[0] / total_loss[1])

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}' )
        dist.barrier()

    dist.destroy_process_group()

def run(rank, world_size: int):

    train_dataset = FakeHeteroDataset(num_graphs=100,
                                            num_node_types=3,
                                            num_edge_types=152,
                                            avg_num_nodes=2700,
                                            avg_degree=1,
                                            avg_num_channels=128,
                                            edge_dim=64,
                                            num_classes=10,
                                            fix_num_channels=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              sampler=train_sampler)
    metadata = train_dataset.data.metadata()
    torch.manual_seed(12345)
    model = Net(n_layer=4, metadata=metadata, num_classes=train_dataset.num_classes)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, 51):
        model.train()

        total_loss = torch.zeros(2).to(rank)
        for data in train_loader:
            data = data.to(rank)
            optimizer.zero_grad()
            logits = model(data.x, data.adj_t, data.batch)
            loss = criterion(logits, data.y.to(torch.float))
            loss.backward()
            optimizer.step()
            total_loss[0] += float(loss) * logits.size(0)
            total_loss[1] += data.num_graphs

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        loss = float(total_loss[0] / total_loss[1])

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    # # Download and process the dataset on main process.
    # Dataset(dataset_name, root,
    #         pre_transform=T.ToSparseTensor(attr='edge_attr'))

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    args = (world_size,)
    mp.spawn(run, args=args, nprocs=world_size, join=True)


"""
DataParallel with Multi GPUs doesn't speed up with large graph  data

Hello,
    I am trying to use DataParallel to speed up the training process with large graph data.
    The large graph data here means a list of graphs, each with roughly 10,000 nodes and 100,000 edges.
    I am using HANConv to train the model because the graphs are heterogeneous.
    I tried to use DataParallel to speed up the training process with multiple GPUs.
    However, I found that the training process only shows a very small speedup, e.g., 1.2x with 8 GPUs.
     I am using PyG 2.2.0 and PyTorch 1.12.1+cu116. 
     Below is a minimal example to reproduce the problem.
    
    On my machine, which is equipped with 8 NVIDIA GeForce GTX TITAN X, running the code below:
    - With 1 GPU, the forward-backward pass takes 0.8s, the epoch time is 41s, `python simulate_clean.py --num_gpu 1`
    - With 2 GPUs, the forward-backward pass takes 1.1s, the epoch time is 32s. `python simulate_clean.py --num_gpu 2`
    - With 4 GPUs, the forward-backward pass takes 2.1s, the epoch time is 32s. `python simulate_clean.py --num_gpu 4`
    - With 6 GPUs, the forward-backward pass takes 4.3s, the epoch time is 45s. `python simulate_clean.py --num_gpu 6`

    Per_device_bs = 2 for all the experiments, so the batch size is 2, 4, 8, 12 for 1, 2, 4, 6 GPUs respectively.

    My understanding is that the forward-backward pass should be faster with more GPUs,if the training is compute-bound.
    But in this case, the forward-backward pass is actually slower with more GPUs so the bound must be somewhere else.
    My guess is that the transfer of data between GPUs is the bottleneck. But I am not sure if this is the case.
    The graphs are represented as a list of HeteroData, and each of them is roughly several MB in size.

    If indeed is the case, how could we overcome this problem?
    (I guess in computer vision, the image data could also be large, and we can use DataParallel to speed up the training process.
    How is that possible?)
"""