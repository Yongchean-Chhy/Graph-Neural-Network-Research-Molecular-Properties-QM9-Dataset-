import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# torch-geometric
import torch_geometric.nn as geom_nn
from torch_geometric.nn import global_mean_pool

#config
from config import get_gnn_output_dim, gnn_layer_by_name, CHEMICAL_ACCURACY_MU


class GNNBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim, num_layers, layer_type, dropout=0.1, heads=1, concat=True, edge_attr_dim=None):
        super().__init__()

        edge_aware_layers = {"GATv2Conv"}
        self.layer_type = layer_type
        self.concat = concat

        layers = []
        input_dim = in_channels

        for i in range(num_layers - 1):
            layer_kwargs = {}
            if layer_type in ["GAT", "GATv2Conv"]:
                layer_kwargs["heads"] = heads
                layer_kwargs["concat"] = concat
            if layer_type in edge_aware_layers and edge_attr_dim is not None:
                layer_kwargs["edge_dim"] = edge_attr_dim

            conv = gnn_layer_by_name[layer_type](
                in_channels=input_dim,
                out_channels=hidden_dim,
                **layer_kwargs
            )
            layers += [conv, nn.ReLU(), nn.Dropout(dropout)]
            input_dim = get_gnn_output_dim(layer_type, hidden_dim, heads, concat)

        # Final conv layer
        final_kwargs = {}
        if layer_type in ["GAT", "GATv2Conv"]:
            final_kwargs["heads"] = heads
            final_kwargs["concat"] = concat
        if layer_type in edge_aware_layers and edge_attr_dim is not None:
            final_kwargs["edge_dim"] = edge_attr_dim

        final_conv = gnn_layer_by_name[layer_type](
            in_channels=input_dim,
            out_channels=out_dim,
            **final_kwargs
        )
        layers.append(final_conv)
        self.model = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for layer in self.model:
            if isinstance(layer, geom_nn.MessagePassing):
                try:
                    x = layer(x, edge_index, edge_attr=edge_attr)
                except TypeError:
                    x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

class ToweredGNNModel(nn.Module):
    def __init__(
        self,
        n_input_dim,
        tower_out_dim,
        n_output_dim,
        num_towers=4,
        tower_layers=3,
        layer_type="GCN",
        dropout=0.1,
        heads=1,
        concat=True,
        edge_attr_dim=None
    ):
        super().__init__()
        self.towers = nn.ModuleList([
            GNNBlock(
                in_channels=n_input_dim,
                hidden_dim=tower_out_dim,
                out_dim=tower_out_dim,
                num_layers=tower_layers,
                layer_type=layer_type,
                dropout=dropout,
                heads=heads,
                concat=concat,
                edge_attr_dim=edge_attr_dim
            )
            for _ in range(num_towers)
        ])

        total_dim = num_towers * get_gnn_output_dim(layer_type, tower_out_dim, heads, concat)
        self.readout = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, 1)
        )
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch, edge_attr=None):
        tower_outputs = []
        for tower in self.towers:
            h = tower(x, edge_index, batch, edge_attr=edge_attr)
            tower_outputs.append(h)
        h_cat = torch.cat(tower_outputs, dim=-1)
        h_pool = self.pool(h_cat, batch)
        return self.readout(h_pool)
    
class EdgeMLP(nn.Module):
    def __init__(self, edge_attr_dim, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(edge_attr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels * out_channels)
        )

    def forward(self, edge_attr):
        return self.network(edge_attr)

