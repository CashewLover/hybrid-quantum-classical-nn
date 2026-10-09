import math
from typing import Callable

import pennylane as qml
import torch
import torch.nn as nn


def quantum_circuit_ops(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    encoding_scale: float,
    entangle: bool,
) -> None:
    for layer_idx in range(n_layers):
        for q in range(n_qubits):
            qml.RY(encoding_scale * inputs[..., q], wires=q)
        for q in range(n_qubits):
            qml.Rot(
                weights[layer_idx, q, 0],
                weights[layer_idx, q, 1],
                weights[layer_idx, q, 2],
                wires=q,
            )
        if entangle:
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])


def _make_init_fn(bound: float = math.pi) -> Callable[[torch.Tensor], torch.Tensor]:
    def _init(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.uniform_(tensor, -bound, bound)

    return _init


class QuantumBlock(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        encoding_scale: float,
        entangle: bool,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_scale = float(encoding_scale)
        self.entangle = bool(entangle)

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _expval_qnode(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            quantum_circuit_ops(
                inputs=inputs,
                weights=weights,
                n_qubits=n_qubits,
                n_layers=n_layers,
                encoding_scale=self.encoding_scale,
                entangle=self.entangle,
            )
            return [qml.expval(qml.Z(q)) for q in range(n_qubits)]

        @qml.qnode(self.dev, interface="torch", diff_method=None)
        def _entropy_qnode(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            quantum_circuit_ops(
                inputs=inputs,
                weights=weights,
                n_qubits=n_qubits,
                n_layers=n_layers,
                encoding_scale=self.encoding_scale,
                entangle=self.entangle,
            )
            return [qml.vn_entropy(wires=[q], log_base=2) for q in range(n_qubits)]

        self._expval_qnode = _expval_qnode
        self._entropy_qnode = _entropy_qnode

        self.torch_layer = qml.qnn.TorchLayer(
            _expval_qnode,
            weight_shapes={"weights": (n_layers, n_qubits, 3)},
            init_method=_make_init_fn(math.pi),
        )
        self.torch_layer.double()

    @property
    def weights(self) -> nn.Parameter:
        return self.torch_layer.weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.torch_layer(x)
        return out.to(dtype=x.dtype, device=x.device)

    def compute_mean_single_qubit_entropy(self, inputs_to_block: torch.Tensor) -> float:
        with torch.no_grad():
            x_cpu = inputs_to_block.detach().to(device="cpu", dtype=torch.float64)
            w_cpu = self.weights.detach().to(device="cpu", dtype=torch.float64)
            per_qubit = self._entropy_qnode(x_cpu, w_cpu)
            stacked = torch.stack([e.reshape(-1) for e in per_qubit], dim=-1)
            return float(stacked.mean().item())
