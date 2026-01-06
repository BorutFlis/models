"""
Autoencoder-based anomaly detection (PyTorch)

What it does:
- Trains an autoencoder to reconstruct "normal" data.
- Uses reconstruction error as an anomaly score.
- Flags anomalies by a threshold (quantile or mean+std).

Works for:
- Tabular/time-series windows (shape: [N, D])
- Any numeric features

Dependencies:
  pip install torch numpy scikit-learn

Usage (quick):
  1) Put your data into X: np.ndarray of shape (N, D)
  2) Optionally provide y (0 normal / 1 anomaly) for evaluation
  3) Run train_autoencoder(...) then score_anomalies(...)

Notes:
- If you already have known anomalies mixed in, try training on a "mostly normal" subset.
- For time series, create sliding windows first, then feed windows as rows.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from abstract_models.imputation import median_imputer

DATA_DIR = "../data"


# ----------------------------
# Model
# ----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), latent_dim: int = 16, dropout: float = 0.0):
        super().__init__()

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                enc_layers += [nn.Dropout(dropout)]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                dec_layers += [nn.Dropout(dropout)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# ----------------------------
# Training / Scoring helpers
# ----------------------------
@dataclass
class AEConfig:
    hidden_dims: Tuple[int, ...] = (128, 64)
    latent_dim: int = 16
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruction_errors(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    errs = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            xhat = model(xb)
            # per-sample MSE
            e = torch.mean((xb - xhat) ** 2, dim=1)
            errs.append(e.detach().cpu().numpy())

    return np.concatenate(errs, axis=0)


def train_autoencoder(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    config: AEConfig = AEConfig(),
) -> Dict[str, object]:
    """
    Returns a dict with:
      - model: trained AE
      - scaler: fitted StandardScaler
      - train_errors: reconstruction errors on train
      - val_errors: reconstruction errors on val (if provided)
    """

    set_seed(config.seed)

    # Scale data (critical for stable training)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val) if X_val is not None else None

    model = AutoEncoder(
        input_dim=Xtr.shape[1],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(config.device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss(reduction="mean")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr.astype(np.float32))),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val = math.inf
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0

        for (xb,) in train_loader:
            xb = xb.to(config.device)

            opt.zero_grad(set_to_none=True)
            xhat = model(xb)
            loss = loss_fn(xhat, xb)
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)

        if Xva is not None:
            val_errs = reconstruction_errors(model, Xva, config.batch_size, config.device)
            val_loss = float(val_errs.mean())

            # simple checkpoint on mean val reconstruction error
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch in {1, 2, 3} or epoch % 10 == 0 or epoch == config.epochs:
                print(f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        else:
            if epoch in {1, 2, 3} or epoch % 10 == 0 or epoch == config.epochs:
                print(f"epoch {epoch:03d} | train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_errs = reconstruction_errors(model, Xtr, config.batch_size, config.device)
    val_errs = reconstruction_errors(model, Xva, config.batch_size, config.device) if Xva is not None else None

    return {
        "model": model,
        "scaler": scaler,
        "train_errors": train_errs,
        "val_errors": val_errs,
        "config": config,
    }


def choose_threshold(errors: np.ndarray, method: str = "quantile", q: float = 0.995, k: float = 3.0) -> float:
    """
    method:
      - "quantile": threshold = quantile(errors, q)
      - "std": threshold = mean + k*std
    """
    if method == "quantile":
        return float(np.quantile(errors, q))
    if method == "std":
        return float(errors.mean() + k * errors.std(ddof=1))
    raise ValueError("method must be 'quantile' or 'std'")


def score_anomalies(
    model: nn.Module,
    scaler: StandardScaler,
    X: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
    threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns:
      - scores: reconstruction error (higher = more anomalous)
      - is_anomaly: boolean mask (if threshold provided)
    """
    Xs = scaler.transform(X)
    scores = reconstruction_errors(model, Xs, batch_size=batch_size, device=device)

    out = {"scores": scores}
    if threshold is not None:
        out["is_anomaly"] = scores > threshold
    return out


def evaluate_if_labels_available(scores: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    y: 0 normal, 1 anomaly
    """
    y = y.astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "avg_precision": float(average_precision_score(y, scores)),
    }


# ----------------------------
# Example (synthetic demo)
# ----------------------------
if __name__ == "__main__":

    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "early_diagnosis_NT.csv"), index_col=[0, 1])
    X = df
    X = median_imputer.fit_transform(X)

    # Train/val split (train on mostly normal; here we filter to normal for training)
    X_train = X[:120000]
    X_val = X[120000:]

    cfg = AEConfig(epochs=30, batch_size=512, latent_dim=8, hidden_dims=(64, 32), lr=1e-3)
    pack = train_autoencoder(X_train, X_val, cfg)

    # Pick threshold from validation (normal-only) errors
    val_errs = pack["val_errors"]
    thr = choose_threshold(val_errs, method="quantile", q=0.995)

    # Score all data
    scored = score_anomalies(
        pack["model"], pack["scaler"], X,
        batch_size=cfg.batch_size, device=cfg.device, threshold=thr
    )

    # How many flagged?
    flagged = int(scored["is_anomaly"].sum())
    print("flagged anomalies:", flagged, f"({flagged / n:.2%})")
