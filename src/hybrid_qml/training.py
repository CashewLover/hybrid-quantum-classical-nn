from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_qml.data import DatasetBundle
from hybrid_qml.protocol import ProtocolConfig


@dataclass
class RunUnitResult:
    experiment: str
    variant: str
    seed: int
    epochs: int
    history: dict[str, list[float]]
    final_metrics: dict[str, float | int | None]
    param_counts: dict[str, int]
    elapsed_seconds: float
    grid_x: np.ndarray | None
    grid_y: np.ndarray | None
    grid_frames: dict[int, np.ndarray]


def build_decision_grid(
    X_train: np.ndarray,
    resolution: int,
    pad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = float(X_train[:, 0].min() - pad), float(X_train[:, 0].max() + pad)
    y_min, y_max = float(X_train[:, 1].min() - pad), float(X_train[:, 1].max() + pad)
    gx = np.linspace(x_min, x_max, resolution, dtype=np.float64)
    gy = np.linspace(y_min, y_max, resolution, dtype=np.float64)
    xx, yy = np.meshgrid(gx, gy)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return gx, gy, points


def _evaluate_split(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(X)
        loss = float(F.cross_entropy(logits, y).item())
        acc = float((logits.argmax(dim=-1) == y).to(torch.float64).mean().item())
    return loss, acc


def _evaluate_grid(
    model: nn.Module,
    grid_tensor: torch.Tensor,
    resolution: int,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(grid_tensor)
        probs = torch.softmax(logits, dim=-1)[:, 1]
    return probs.detach().cpu().numpy().reshape(resolution, resolution)


def train_and_evaluate(
    model: nn.Module,
    dataset: DatasetBundle,
    experiment: str,
    variant: str,
    seed: int,
    epochs: int,
    grid_resolution: int,
    cfg: ProtocolConfig,
    device: str = "cpu",
    save_frames: bool = True,
) -> RunUnitResult:
    t0 = time.perf_counter()

    X_train_t = torch.tensor(dataset.X_train, dtype=torch.float64, device=device)
    y_train_t = torch.tensor(dataset.y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(dataset.X_test, dtype=torch.float64, device=device)
    y_test_t = torch.tensor(dataset.y_test, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    batch_gen = torch.Generator(device="cpu").manual_seed(cfg.batch_seed_offset + seed)
    n_train = X_train_t.shape[0]

    capture_epochs: set[int] = set()
    grid_x: np.ndarray | None = None
    grid_y: np.ndarray | None = None
    grid_t: torch.Tensor | None = None
    if save_frames and seed == 0:
        capture_epochs = {ep for ep in cfg.frame_epochs if ep <= epochs} | {epochs}
        grid_x, grid_y, grid_pts = build_decision_grid(dataset.X_train, grid_resolution)
        grid_t = torch.tensor(grid_pts, dtype=torch.float64, device=device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    grid_frames: dict[int, np.ndarray] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=batch_gen).to(device=device)
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        tr_loss, tr_acc = _evaluate_split(model, X_train_t, y_train_t)
        te_loss, te_acc = _evaluate_split(model, X_test_t, y_test_t)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if epoch in capture_epochs and grid_t is not None:
            grid_frames[epoch] = _evaluate_grid(model, grid_t, grid_resolution)

    mean_entropy: float | None = None
    if getattr(model, "has_quantum_block", False):
        model.eval()
        with torch.no_grad():
            ent_inputs = model.encode_for_middle(X_test_t[dataset.entropy_indices])
            mean_entropy = float(model.middle.compute_mean_single_qubit_entropy(ent_inputs))

    elapsed = time.perf_counter() - t0
    param_counts = model.parameter_counts()
    final_metrics: dict[str, float | int | None] = {
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_test_acc": history["test_acc"][-1],
        "mean_single_qubit_entropy": mean_entropy,
        "middle_params": param_counts["middle"],
        "total_params": param_counts["total"],
    }

    return RunUnitResult(
        experiment=experiment,
        variant=variant,
        seed=seed,
        epochs=epochs,
        history=history,
        final_metrics=final_metrics,
        param_counts=param_counts,
        elapsed_seconds=elapsed,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_frames=grid_frames,
    )
