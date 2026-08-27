import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

# Paths
model_path = "models/"
scaler_params = "scalers/"

# GNN layer mapping
gnn_layer_by_name = {
    
    "GGNN": geom_nn.NNConv

}

# Paper constant
CHEMICAL_ACCURACY_MU = 0.1

# Target columns in QM9
QM9_TARGETS = {
    "mu": 0.1,
    "alpha": .1,
    "homo": .043,
    "lumo": .043,
    "gap": .043,
    "R2": 1.2,
    "zpve": 0.0012,
    "UO": .043,
    "U": .043,
    "H": .043,
    "G": .043,
    "Cv": .043,
    "Omega":10

}

# For GAT-specific output shape correction
def get_gnn_output_dim(layer_type, out_channels, heads=1, concat=True):
    if layer_type in ["GAT", "GATv2Conv"] and concat:
        return out_channels * heads
    return out_channels
