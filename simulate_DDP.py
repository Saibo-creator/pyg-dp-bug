import os
import os.path as osp
import pickle
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


def run(rank, world_size: int):


    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # load the dataset with pickle
    pickle_path = osp.join(osp.dirname(osp.realpath(__file__)), 'fake_dataset.pkl')
    with open("tmp.pickle", "rb") as f:
        train_dataset = pickle.load(f)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=4,
                              sampler=train_sampler)

    torch.manual_seed(12345)
    model = Net(n_layer=4, metadata=metadata, num_classes=train_dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, 51):
        model.train()

        total_loss = torch.zeros(2).to(rank)
        for data in tqdm(train_loader):
            data = data.to(rank)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.nll_loss(logits, data.y)
            # loss = criterion(logits, data.y.to(torch.float))
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

    train_dataset = FakeHeteroDataset(num_graphs=100,
                                            num_node_types=3,
                                            num_edge_types=152,
                                            avg_num_nodes=2700,
                                            avg_degree=1,
                                            avg_num_channels=128,
                                            edge_dim=64,
                                            num_classes=10,
                                            fix_num_channels=True)
    # save the dataset with pickle
    pickle_path = osp.join(osp.dirname(osp.realpath(__file__)), 'fake_dataset.pkl')
    with open("tmp.pickle", "wb") as f:
        pickle.dump((train_dataset), f)

    metadata = train_dataset.data.metadata()



    gpu_ids = "6,7"  # Specify the IDs of the GPUs you want to use, separated by commas
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    args = (world_size,)
    mp.spawn(run, args=args, nprocs=world_size, join=True)
