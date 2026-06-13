"""GCN and H2GCN training harness. Exposes train_one_run(model, data_dir, seed)
returning a metrics dict. Used by run_batch.py to sweep (model, dataset, seed).
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

import json

from .earlystopping import EarlyStopping
from metric.metrics import f1_loss, BCE_loss, _eval_rocauc, ap_score, ap_score_full


def load_synthetic(data_dir: str, train_percent: float = 0.6, seed: int = 42) -> Data:
    features = torch.tensor(
        pd.read_csv(os.path.join(data_dir, 'features.csv')).values,
        dtype=torch.float,
    )
    labels = torch.tensor(
        pd.read_csv(os.path.join(data_dir, 'labels.csv')).values,
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        np.load(os.path.join(data_dir, 'edge_index.npy')),
        dtype=torch.long,
    )

    n = features.shape[0]
    # Symmetric normalised adjacency with self-loops: D^{-1/2} (A + I) D^{-1/2}.
    adj = SparseTensor(
        row=edge_index[1], col=edge_index[0],
        sparse_sizes=(n, n),
        trust_data=True,
    )
    adj_t = adj.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # 60/20/20 random node split for train/val/test.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_end = int(n * train_percent)
    val_end = train_end + (n - train_end) // 2

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    G = Data(x=features, y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.adj_t = adj_t
    G.n_id = torch.arange(n)
    return G


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, class_channels, normalize=False)

    def forward(self, x, adj_t):
        x = F.relu(self.conv1(x, adj_t))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj_t)
        return F.sigmoid(x)


class H2GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.dense1 = torch.nn.Linear(nfeat, nhid)
        self.dense2 = torch.nn.Linear(nhid * 7, nclass)
        self.dropout = dropout
        self.conv1 = GCNConv(nhid, nhid, normalize=False)
        self.conv2 = GCNConv(nhid * 2, nhid * 2, normalize=False)
        self.relu = torch.nn.ReLU()

    def forward(self, features, adj_t):
        # Ego representation kept separate; 1-hop (x1) and 2-hop (x2) aggregated
        # via successive GCN convs and then concatenated with the ego term.
        x = self.relu(self.dense1(features))
        x11 = self.conv1(x, adj_t)
        x12 = self.conv1(x11, adj_t)
        x1 = torch.cat((x11, x12), -1)
        x21 = self.conv2(x1, adj_t)
        x22 = self.conv2(x21, adj_t)
        x2 = torch.cat((x21, x22), -1)
        x = torch.cat((x, x1, x2), dim=-1)
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        return F.sigmoid(x)


_DEFAULT_HIDDEN = {'GCN': 256, 'H2GCN': 64}


def _build_model(model_name: str, in_dim: int, hidden: int, out_dim: int) -> torch.nn.Module:
    if model_name == 'GCN':
        return GCN(in_dim, hidden, out_dim)
    if model_name == 'H2GCN':
        return H2GCN(in_dim, hidden, out_dim)
    raise ValueError(f"Unknown model: {model_name}")


def train_one_run(
    model_name: str,
    data_dir: str,
    seed: int = 42,
    train_percent: float = 0.6,
    hidden: int | None = None,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 1000,
    patience: int = 100,
    device: str = 'cpu',
    checkpoint_path: str = 'checkpoint.pt',
    verbose: bool = False,
    log_per_label: bool = False,
) -> dict:
    """Train one (model, dataset, seed) configuration and return metrics.

    Training and evaluation logic mirrors gcn_synthetic.py / h2gcn_synthetic.py
    exactly: per-epoch train/val/test F1 + AUC-ROC, BCE val loss for early
    stopping, best-checkpoint reload at the end. Per-epoch printing is gated
    on `verbose` so batch runs stay quiet.
    """
    if hidden is None:
        hidden = _DEFAULT_HIDDEN[model_name]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    G = load_synthetic(data_dir, train_percent=train_percent, seed=seed)

    model = _build_model(model_name, G.x.shape[1], hidden, G.y.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    x = G.x.to(device)
    labels = G.y.to(device)
    adj_t = G.adj_t.to(device)
    train_mask = G.train_mask.to(device)
    val_mask = G.val_mask.to(device)
    test_mask = G.test_mask.to(device)

    early_stopping = EarlyStopping(
        patience=patience, verbose=verbose, path=checkpoint_path
    )

    def model_train():
        model.train()
        optimizer.zero_grad()

        output = model(x, adj_t)
        loss_train = BCE_loss(output[train_mask], labels[train_mask])

        micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
        roc_auc_train_macro = _eval_rocauc(labels[train_mask], output[train_mask])

        loss_train.backward()
        optimizer.step()

        return loss_train, micro_train, macro_train, roc_auc_train_macro

    @torch.no_grad()
    def model_test():
        model.eval()

        output = model(x, adj_t)

        loss_val = BCE_loss(output[val_mask], labels[val_mask])
        micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
        roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])

        micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
        roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
        ap_test = ap_score(labels[test_mask], output[test_mask])

        return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test

    epochs_trained = 0
    for epoch in range(1, epochs):
        epochs_trained = epoch
        loss_train, micro_train, macro_train, roc_auc_train_macro = model_train()
        loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
        if verbose:
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f} '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            if verbose:
                print("Early stopping")
            break

    if verbose:
        print("Optimization Finished!")
    model.load_state_dict(torch.load(checkpoint_path))
    loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()

    # Recompute AP family from the best-checkpoint outputs so all three
    # variants come from the same predictions.
    with torch.no_grad():
        model.eval()
        out_full = model(x, adj_t)
        macro_ap, micro_ap, per_label_ap = ap_score_full(
            labels[test_mask], out_full[test_mask]
        )

    if verbose:
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test macro AP: {macro_ap:.4f}, '
              f'Test micro AP: {micro_ap:.4f}, '
              )

    result = {
        'model': model_name,
        'dataset': os.path.basename(os.path.normpath(data_dir)),
        'data_dir': data_dir,
        'seed': seed,
        'train_percent': train_percent,
        'hidden': hidden,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs_trained': epochs_trained,
        'micro_f1': float(micro_test),
        'macro_f1': float(macro_test),
        'auc_roc_macro': float(roc_auc_test_macro),
        'ap': float(macro_ap),
        'ap_micro': float(micro_ap),
    }
    if log_per_label:
        result['ap_per_label'] = json.dumps(per_label_ap)
    return result
