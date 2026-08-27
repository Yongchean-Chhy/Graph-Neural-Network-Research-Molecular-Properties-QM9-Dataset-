# Gated Graph Neural Networks for Molecular Property Prediction

This project implements and trains **Gated Graph Neural Networks (GGNNs)** to predict quantum-chemical properties of small organic molecules from the [QM9 dataset](http://quantum-machine.org/datasets/), using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [PyTorch Lightning](https://lightning.ai/).

Each molecule is represented as a graph — atoms as nodes, bonds as edges — and the model learns to regress a target property (e.g. dipole moment) directly from that graph structure.

## Background

QM9 contains ~134k stable small organic molecules (up to 9 heavy atoms), each with 19 properties computed via DFT (density functional theory), including dipole moment, polarizability, HOMO/LUMO energies, and heat capacity.

This implementation is based on the Gated Graph Sequence Neural Networks architecture ([Li et al., 2016](https://arxiv.org/abs/1511.05493)): instead of a standard convolution-style update, each node's hidden state is updated at every propagation step using a **GRU cell**, allowing the model to retain and gate information over multiple rounds of message passing.

## Project structure

```
GGNN /
├── classes.py     # Core GGNN layer, model, Lightning module, and training loop
├── tower.py        # Generic multi-architecture GNN module (GCN/GAT/GATv2 "towers"), used for comparisons
├── utils.py        # QM9 loading + data transforms (target selection, normalization, feature ablation)
├── config.py       # Paths, layer-name registry, QM9 target list, chemical accuracy constants
├── train.ipynb     # Main notebook: loads QM9, trains the GGNN, compares with/without edge features
├── data/            # QM9 raw + PyG-processed dataset files
└── models/          # Saved model checkpoints
```

### Core components

- **`EdgeAwareGGNNLayer`** (`classes.py`) — a custom PyTorch Geometric `MessagePassing` layer implementing one GGNN propagation step: aggregates neighbor messages (optionally combined with encoded edge features), then updates each node's state with `nn.GRUCell`.
- **`GNNModel`** (`classes.py`) — stacks several `EdgeAwareGGNNLayer`s, pools node embeddings per graph with mean pooling, and predicts a single scalar target through a small MLP readout head.
- **`GraphLevelGNN`** (`classes.py`) — a `pytorch_lightning.LightningModule` wrapper handling training/validation/test steps, MSE loss, MAE, and the "error ratio" metric (MAE ÷ chemical accuracy threshold — the standard way QM9 results are benchmarked against DFT accuracy).
- **`train_graph_regressor`** (`classes.py`) — orchestrates dataset splitting, `Trainer` setup (with best-checkpoint saving on validation MAE), training, and test-set evaluation.

## Setup

```bash
pip install torch torch_geometric pytorch_lightning torchmetrics rdkit pandas matplotlib
```

> The QM9 dataset will be automatically downloaded and processed into `data/` the first time it's loaded via PyTorch Geometric's `QM9` dataset class (already cached in this repo under `data/processed/`).

## Usage

Open and run `GGNN /train.ipynb`. It will:

1. Load QM9 and select a target property (default: target index `0`, the dipole moment `mu`).
2. Subsample to 20,000 molecules and split 70/10/20 into train/val/test.
3. Normalize the target using training-set statistics.
4. Train a `GGNN` model **with** and **without** edge attributes, to measure the effect of bond-feature information.
5. Plot the resulting test error ratios against the paper's reported baseline.

Example training call:

```python
model, results = train_graph_regressor(
    model_name="GGNN",
    dataset=qm9,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    n_input_dim=qm9.num_node_features,
    n_hidden_features=128,
    n_output_dim=128,
    num_layers=3,
    edge_attr_dim=qm9[0].edge_attr.shape[1],
    use_edge_attr=True,
)
```

## Metrics

Model performance is reported as:

- **MSE loss** — training objective.
- **MAE** — mean absolute error on the (normalized) target.
- **Error ratio** — `MAE / chemical_accuracy`, where chemical accuracy is a per-target constant defined in `config.py`. An error ratio below 1.0 indicates the model has reached "chemical accuracy" for that property, the standard benchmark used in the QM9 literature.

## Notes

- `tower.py` provides a more general, swappable-architecture variant (`ToweredGNNModel`) that runs several parallel GNN "towers" (GCN, GAT, GATv2, etc.) and concatenates their outputs — useful for architecture comparisons, but separate from the core GGNN model.
- `utils.py` also includes a `FeatureAblation` transform for testing which input node features are most important to prediction quality.

## References

- Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2016). [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).
- Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). [Quantum chemistry structures and properties of 134 kilo molecules](https://www.nature.com/articles/sdata201422) (QM9 dataset).
