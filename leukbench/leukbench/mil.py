"""Patient-level models.

pooled_features : summary statistics of a patient's cell features (mean, std, 10/50/90th pct)
                  -> logistic regression. A strong, transparent baseline.
GatedABMIL      : gated attention MIL (Ilse et al., ICML 2018) over cell embeddings, as used for
                  AML subtyping (Hehr et al. 2023) and in cAItomorph's aggregator family.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pooled_features(X: np.ndarray, groups: np.ndarray, patients: list) -> np.ndarray:
    out = []
    for g in patients:
        x = X[groups == g]
        out.append(np.concatenate([np.nanmean(x, 0), np.nanstd(x, 0),
                                   np.nanpercentile(x, 10, 0), np.nanpercentile(x, 50, 0),
                                   np.nanpercentile(x, 90, 0)]))
    return np.asarray(out, dtype="float32")


class GatedABMIL(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Sigmoid())
        self.att_w = nn.Linear(hidden // 2, 1)
        self.cls = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: [n_cells, in_dim]
        h = self.fc(x)
        a = self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1)
        a = torch.softmax(a, dim=0)
        z = (a.unsqueeze(-1) * h).sum(0)
        return self.cls(z).squeeze(-1), a


class _Standardiser:
    """Mean/std learned on TRAINING cells only; NaNs replaced by training medians."""

    def fit(self, X):
        self.med = np.nanmedian(X, 0)
        Xf = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-6
        return self

    def __call__(self, X):
        Xf = np.where(np.isnan(X), self.med, X)
        return ((Xf - self.mu) / self.sd).astype("float32")


def train_abmil(bags: list[np.ndarray], y: np.ndarray, seed: int = 0, epochs: int = 40, lr: float = 2e-4,
                weight_decay: float = 1e-3, cell_dropout: float = 0.2, hidden: int = 128, device: str = "cpu"):
    """Train on a list of bags (raw features). Returns a predict(bags) -> probabilities function."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    std = _Standardiser().fit(np.concatenate(bags))
    tb = [torch.from_numpy(std(b)) for b in bags]
    model = GatedABMIL(tb[0].shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(epochs):
        model.train()
        for i in rng.permutation(len(tb)):
            x = tb[i]
            if cell_dropout and len(x) > 10:                   # bag augmentation
                keep = rng.random(len(x)) > cell_dropout
                x = x[torch.from_numpy(keep)]
            logit, _ = model(x.to(device))
            loss = loss_fn(logit.unsqueeze(0), torch.tensor([float(y[i])], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(new_bags, return_attention=False):
        model.eval()
        ps, atts = [], []
        for b in new_bags:
            logit, a = model(torch.from_numpy(std(b)).to(device))
            ps.append(torch.sigmoid(logit).item())
            atts.append(a.cpu().numpy())
        return (np.asarray(ps), atts) if return_attention else np.asarray(ps)

    return predict
