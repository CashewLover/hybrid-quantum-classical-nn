from dataclasses import asdict, dataclass
import math
from typing import Any


MOONS_VARIANTS: tuple[str, ...] = (
    "hybrid_entangled",
    "hybrid_no_entanglement",
    "classical_mlp",
    "no_middle",
)

PARITY_VARIANTS: tuple[str, ...] = (
    "parity_entangled",
    "parity_no_entanglement",
    "parity_classical_mlp",
)

VARIANT_LABELS: dict[str, str] = {
    "hybrid_entangled": "Hybrid (entangled)",
    "hybrid_no_entanglement": "Hybrid (no entanglement)",
    "classical_mlp": "Classical MLP",
    "no_middle": "No middle block",
    "parity_entangled": "Parity quantum (entangled)",
    "parity_no_entanglement": "Parity quantum (entanglement-off)",
    "parity_classical_mlp": "Classical MLP (reference)",
}

HYPOTHESIS_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "H1",
        "experiment": "E2 (Moons)",
        "variant_a": "hybrid_entangled",
        "variant_b": "classical_mlp",
        "label_a": "Hybrid (entangled)",
        "label_b": "Classical MLP",
        "prediction": "Practically equivalent",
        "question": "Replacing classical MLP hidden block with quantum circuit on moons",
    },
    {
        "id": "H2",
        "experiment": "E2 (Moons)",
        "variant_a": "hybrid_entangled",
        "variant_b": "hybrid_no_entanglement",
        "label_a": "Hybrid (entangled)",
        "label_b": "Hybrid (no entanglement)",
        "prediction": "Practically equivalent",
        "question": "Removing entangling ring from moons quantum circuit",
    },
    {
        "id": "H3",
        "experiment": "E3 (Parity)",
        "variant_a": "parity_entangled",
        "variant_b": "parity_no_entanglement",
        "label_a": "Parity quantum (entangled)",
        "label_b": "Parity quantum (entanglement-off)",
        "prediction": "Entangled higher by more than the margin",
        "question": "Removing entangling ring when features interact only via entanglement",
    },
)


@dataclass(frozen=True)
class ProtocolConfig:
    title: str = (
        "Does the Quantum Layer Matter? An Entanglement-Ablation Study "
        "of a Hybrid Quantum\u2013Classical Neural Network"
    )
    n_qubits_moons: int = 4
    n_qubits_parity: int = 2
    n_layers: int = 3
    encoding_scale_moons: float = math.pi
    encoding_scale_parity: float = math.pi / 2.0

    n_train: int = 300
    n_test: int = 5000
    moons_noise: float = 0.20
    parity_cluster_std: float = 0.25

    lr: float = 0.05
    batch_size: int = 32
    moons_epochs: int = 40
    parity_epochs: int = 60

    entropy_sample_size: int = 500
    equivalence_margin: float = 0.02
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20260605
    additive_ceiling: float = 0.75
    additive_leak_tolerance: float = 0.05

    standard_seeds: int = 20
    quick_seeds: int = 3
    smoke_seeds: int = 1
    default_device: str = "cpu"

    frame_epochs: tuple[int, ...] = (1, 6, 11, 16, 21, 26, 31, 36)
    grid_resolution_standard: int = 100
    grid_resolution_fast: int = 40

    test_seed_moons: int = 900_001
    test_seed_parity: int = 900_002
    entropy_subsample_seed: int = 900_003
    train_data_seed_offset: int = 100_000
    outer_init_seed_offset: int = 200_000
    middle_init_seed_offset: int = 300_000
    batch_seed_offset: int = 400_000
    pilot_seed_start: int = 1000
    negative_control_seeds: tuple[int, ...] = (8000, 8001, 8002)
    negative_control_tolerance: float = 0.07

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSpec:
    preset: str
    seeds: int
    moons_epochs: int
    parity_epochs: int
    grid_resolution: int
    device: str
    is_confirmatory: bool
    is_preliminary: bool
    protocol: ProtocolConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "seeds": self.seeds,
            "moons_epochs": self.moons_epochs,
            "parity_epochs": self.parity_epochs,
            "grid_resolution": self.grid_resolution,
            "device": self.device,
            "is_confirmatory": self.is_confirmatory,
            "is_preliminary": self.is_preliminary,
            "protocol": self.protocol.to_dict(),
        }


def resolve_run_spec(
    preset: str = "standard",
    seeds_override: int | None = None,
    device_override: str | None = None,
    protocol: ProtocolConfig | None = None,
) -> RunSpec:
    cfg = protocol or ProtocolConfig()
    if preset not in {"smoke", "quick", "standard"}:
        raise ValueError(f"Unknown preset {preset!r}; choose from smoke, quick, standard.")

    if seeds_override is not None and seeds_override < 1:
        raise ValueError("Seed count must be at least 1.")

    device = (device_override or cfg.default_device).lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device {device!r}; choose 'cpu' or 'cuda'.")

    if preset == "smoke":
        default_seeds = cfg.smoke_seeds
        moons_ep = 2
        parity_ep = 2
        grid_res = cfg.grid_resolution_fast
    elif preset == "quick":
        default_seeds = cfg.quick_seeds
        moons_ep = max(1, cfg.moons_epochs // 4)
        parity_ep = max(1, cfg.parity_epochs // 4)
        grid_res = cfg.grid_resolution_fast
    else:
        default_seeds = cfg.standard_seeds
        moons_ep = cfg.moons_epochs
        parity_ep = cfg.parity_epochs
        grid_res = cfg.grid_resolution_standard

    seeds = seeds_override if seeds_override is not None else default_seeds
    is_confirmatory = (
        preset == "standard"
        and seeds_override is None
        and device_override is None
    )
    is_preliminary = seeds < 10

    return RunSpec(
        preset=preset,
        seeds=seeds,
        moons_epochs=moons_ep,
        parity_epochs=parity_ep,
        grid_resolution=grid_res,
        device=device,
        is_confirmatory=is_confirmatory,
        is_preliminary=is_preliminary,
        protocol=cfg,
    )
