from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .losses import total_loss


def train_model(model, train_loader, val_loader, cfg: Config):
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()

        train_losses = []
        train_phys = []

        for batch in train_loader:
            optimizer.zero_grad()

            loss, parts = total_loss(batch, model, cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_phys.append(parts["physics"].item())

        model.eval()
        val_losses = []
        val_phys = []

        # NOTE: PINN validation still needs autograd to compute time-derivatives for the physics loss.
        # We do NOT call backward/optimizer here, but we must keep grads enabled.
        with torch.enable_grad():
            for batch in val_loader:
                loss, parts = total_loss(batch, model, cfg)
                val_losses.append(loss.detach().item())
                val_phys.append(parts["physics"].detach().item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        improved = (best_val_loss - avg_val) > float(cfg.early_stop_min_delta)
        if improved:
            best_val_loss = avg_val
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train Loss: {avg_train:.6f} | "
                f"Val Loss: {avg_val:.6f} | "
                f"Train Phys: {np.mean(train_phys):.6f} | "
                f"Val Phys: {np.mean(val_phys):.6f}"
            )

        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch:04d} "
                f"(no val improvement > {cfg.early_stop_min_delta} for {cfg.early_stop_patience} epochs)."
            )
            break

    model.load_state_dict(best_state)
    return model

