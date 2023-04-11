import os.path as osp
import time
from typing import List

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.nn.conv import HANConv
from torch_geometric.utils.unbatch import unbatch
from tqdm import tqdm
import cProfile

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


def main(num_gpu=1, device_ids=None):

    large_graph_dataset = FakeHeteroDataset(num_graphs=100,
                                            num_node_types=3,
                                            num_edge_types=152,
                                            avg_num_nodes=2700,
                                            avg_degree=1,
                                            avg_num_channels=128,
                                            edge_dim=64,
                                            num_classes=10,
                                            fix_num_channels=True)

    per_gpu_bs = 2
    bs = per_gpu_bs * num_gpu

    loader = DataListLoader(large_graph_dataset, batch_size=bs, shuffle=True)

    metadata = large_graph_dataset.data.metadata()
    model = Net(n_layer=4, metadata=metadata, num_classes=large_graph_dataset.num_classes)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Let's use {num_gpu} GPUs!")
        device_ids = list(range(num_gpu)) if device_ids is None else device_ids
        model = DataParallel(model, device_ids=device_ids)
        model.to(device_ids[0])
    else:
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for data_list in tqdm(loader):
        optimizer.zero_grad()
        start = time.time()
        output = model(data_list)
        end = time.time()
        print(f'Forward pass time: {end - start}')
        y = torch.cat([data.y for data in data_list]).to(output.device)
        loss = F.nll_loss(output, y)
        loss.backward()
        print(f'Backward pass time: {time.time() - end}')
        # optimizer.step()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--device_ids', type=int, nargs='+', default=None)
    args = parser.parse_args()

    main(num_gpu=args.num_gpu, device_ids=args.device_ids)
    cProfile.run(f'main(num_gpu={args.num_gpu}, device_ids={args.device_ids})')
