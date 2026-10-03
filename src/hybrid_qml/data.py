from dataclasses import dataclass
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from hybrid_qml.protocol import ProtocolConfig


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    entropy_indices: np.ndarray


def _fixed_entropy_indices(n_test: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(sample_size, n_test)
    return np.sort(rng.choice(n_test, size=k, replace=False))


def make_moons_dataset(seed: int, cfg: ProtocolConfig) -> DatasetBundle:
    X_train_raw, y_train = make_moons(
        n_samples=cfg.n_train,
        shuffle=True,
        noise=cfg.moons_noise,
        random_state=cfg.train_data_seed_offset + seed,
    )
    X_test_raw, y_test = make_moons(
        n_samples=cfg.n_test,
        shuffle=True,
        noise=cfg.moons_noise,
        random_state=cfg.test_seed_moons,
    )

    scaler = StandardScaler()
    X_train = np.asarray(scaler.fit_transform(X_train_raw), dtype=np.float64)
    X_test = np.asarray(scaler.transform(X_test_raw), dtype=np.float64)

    return DatasetBundle(
        X_train=X_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=X_test,
        y_test=np.asarray(y_test, dtype=np.int64),
        X_train_raw=np.asarray(X_train_raw, dtype=np.float64),
        X_test_raw=np.asarray(X_test_raw, dtype=np.float64),
        entropy_indices=_fixed_entropy_indices(
            cfg.n_test, cfg.entropy_sample_size, cfg.entropy_subsample_seed
        ),
    )


def _sample_parity_clusters(
    points_per_cluster: int,
    std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    centers = (
        (1.0, 1.0, 0),
        (-1.0, -1.0, 0),
        (1.0, -1.0, 1),
        (-1.0, 1.0, 1),
    )
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for cx, cy, label in centers:
        pts = rng.normal(
            loc=np.array([cx, cy], dtype=np.float64),
            scale=std,
            size=(points_per_cluster, 2),
        )
        xs.append(pts)
        ys.append(np.full(points_per_cluster, label, dtype=np.int64))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return np.asarray(X[perm], dtype=np.float64), np.asarray(y[perm], dtype=np.int64)


def make_parity_dataset(seed: int, cfg: ProtocolConfig) -> DatasetBundle:
    if cfg.n_train % 4 != 0 or cfg.n_test % 4 != 0:
        raise ValueError("Parity split sizes must be divisible by 4.")
    train_rng = np.random.default_rng(cfg.train_data_seed_offset + 50_000 + seed)
    test_rng = np.random.default_rng(cfg.test_seed_parity)

    X_train_raw, y_train = _sample_parity_clusters(
        cfg.n_train // 4, cfg.parity_cluster_std, train_rng
    )
    X_test_raw, y_test = _sample_parity_clusters(
        cfg.n_test // 4, cfg.parity_cluster_std, test_rng
    )

    scaler = StandardScaler()
    X_train = np.asarray(scaler.fit_transform(X_train_raw), dtype=np.float64)
    X_test = np.asarray(scaler.transform(X_test_raw), dtype=np.float64)

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        entropy_indices=_fixed_entropy_indices(
            cfg.n_test, cfg.entropy_sample_size, cfg.entropy_subsample_seed
        ),
    )
