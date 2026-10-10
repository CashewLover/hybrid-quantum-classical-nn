import torch
import torch.nn as nn

from hybrid_qml.protocol import ProtocolConfig


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MoonsClassifier(nn.Module):
    def __init__(
        self,
        in_linear: nn.Linear,
        middle: nn.Module,
        out_linear: nn.Linear,
        variant: str,
    ) -> None:
        super().__init__()
        self.in_linear = in_linear
        self.act = nn.Tanh()
        self.middle = middle
        self.out_linear = out_linear
        self.variant = variant
        self.has_quantum_block = hasattr(middle, "compute_mean_single_qubit_entropy")

    def encode_for_middle(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.in_linear(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_for_middle(x)
        h = self.middle(h)
        return self.out_linear(h)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "middle": count_trainable_parameters(self.middle),
            "total": count_trainable_parameters(self),
        }


def build_moons_model(
    variant: str,
    seed: int,
    cfg: ProtocolConfig,
    device: str = "cpu",
) -> MoonsClassifier:
    n = cfg.n_qubits_moons
    devices = [torch.cuda.current_device()] if device == "cuda" and torch.cuda.is_available() else []

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.outer_init_seed_offset + seed)
        in_linear = nn.Linear(2, n, dtype=torch.float64)
        out_linear = nn.Linear(n, 2, dtype=torch.float64)

        torch.manual_seed(cfg.middle_init_seed_offset + seed)
        if variant == "classical_mlp":
            middle: nn.Module = nn.Sequential(
                nn.Linear(n, n, dtype=torch.float64),
                nn.Tanh(),
                nn.Linear(n, n, dtype=torch.float64),
            )
        elif variant == "no_middle":
            middle = nn.Identity()
        elif variant in {"hybrid_entangled", "hybrid_no_entanglement"}:
            from hybrid_qml.quantum import QuantumBlock

            middle = QuantumBlock(
                n_qubits=n,
                n_layers=cfg.n_layers,
                encoding_scale=cfg.encoding_scale_moons,
                entangle=(variant == "hybrid_entangled"),
            )
        else:
            raise ValueError(f"Unknown moons variant: {variant!r}")

    model = MoonsClassifier(in_linear=in_linear, middle=middle, out_linear=out_linear, variant=variant)
    return model.to(device=device, dtype=torch.float64)


class FeaturewiseScaleShift(nn.Module):
    def __init__(self, n_features: int = 2) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features, dtype=torch.float64))
        self.shift = nn.Parameter(torch.zeros(n_features, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class ParityQuantumClassifier(nn.Module):
    def __init__(
        self,
        pre: FeaturewiseScaleShift,
        middle: nn.Module,
        out_linear: nn.Linear,
        variant: str,
    ) -> None:
        super().__init__()
        self.pre = pre
        self.middle = middle
        self.out_linear = out_linear
        self.variant = variant
        self.has_quantum_block = hasattr(middle, "compute_mean_single_qubit_entropy")

    def encode_for_middle(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.encode_for_middle(x)
        h = self.middle(u)
        return self.out_linear(h)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "middle": count_trainable_parameters(self.middle),
            "total": count_trainable_parameters(self),
        }


def build_parity_model(
    variant: str,
    seed: int,
    cfg: ProtocolConfig,
    device: str = "cpu",
) -> nn.Module:
    if variant == "parity_classical_mlp":
        return build_moons_model("classical_mlp", seed=seed, cfg=cfg, device=device)

    if variant not in {"parity_entangled", "parity_no_entanglement"}:
        raise ValueError(f"Unknown parity variant: {variant!r}")

    from hybrid_qml.quantum import QuantumBlock

    devices = [torch.cuda.current_device()] if device == "cuda" and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.outer_init_seed_offset + seed)
        pre = FeaturewiseScaleShift(cfg.n_qubits_parity)
        out_linear = nn.Linear(cfg.n_qubits_parity, 2, dtype=torch.float64)

        torch.manual_seed(cfg.middle_init_seed_offset + seed)
        middle = QuantumBlock(
            n_qubits=cfg.n_qubits_parity,
            n_layers=cfg.n_layers,
            encoding_scale=cfg.encoding_scale_parity,
            entangle=(variant == "parity_entangled"),
        )

    model = ParityQuantumClassifier(pre=pre, middle=middle, out_linear=out_linear, variant=variant)
    return model.to(device=device, dtype=torch.float64)
