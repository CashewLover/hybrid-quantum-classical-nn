from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from hybrid_qml.data import make_moons_dataset, make_parity_dataset
from hybrid_qml.models import build_moons_model, build_parity_model
from hybrid_qml.protocol import MOONS_VARIANTS, PARITY_VARIANTS, ProtocolConfig
from hybrid_qml.quantum import QuantumBlock
from hybrid_qml.training import train_and_evaluate


@dataclass(frozen=True)
class ValidationCheckResult:
    check_id: int
    name: str
    passed: bool
    detail: str


_I2 = np.eye(2, dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def _ry_matrix(theta: float) -> np.ndarray:
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz_matrix(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * phi), 0.0], [0.0, np.exp(0.5j * phi)]],
        dtype=np.complex128,
    )


def _rot_matrix(phi: float, theta: float, omega: float) -> np.ndarray:
    return _rz_matrix(omega) @ _ry_matrix(theta) @ _rz_matrix(phi)


def _kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _single_qubit_op(op: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    return _kron_all([op if q == wire else _I2 for q in range(n_qubits)])


def _cnot_op(control: int, target: int, n_qubits: int) -> np.ndarray:
    term0 = _kron_all([_P0 if q == control else _I2 for q in range(n_qubits)])
    term1 = _kron_all(
        [
            _P1 if q == control else (_X if q == target else _I2)
            for q in range(n_qubits)
        ]
    )
    return term0 + term1


def dense_statevector_expvals(
    x: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    encoding_scale: float,
    entangle: bool,
) -> tuple[np.ndarray, float]:
    dim = 1 << n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j

    cnot_gates = (
        [_cnot_op(q, (q + 1) % n_qubits, n_qubits) for q in range(n_qubits)]
        if entangle
        else []
    )
    z_ops = [_single_qubit_op(_Z, q, n_qubits) for q in range(n_qubits)]

    for layer_idx in range(n_layers):
        ry_layer = _kron_all(
            [_ry_matrix(float(encoding_scale * x[q])) for q in range(n_qubits)]
        )
        psi = ry_layer @ psi

        rot_layer = _kron_all(
            [
                _rot_matrix(
                    float(weights[layer_idx, q, 0]),
                    float(weights[layer_idx, q, 1]),
                    float(weights[layer_idx, q, 2]),
                )
                for q in range(n_qubits)
            ]
        )
        psi = rot_layer @ psi

        for gate in cnot_gates:
            psi = gate @ psi

    norm = float(np.linalg.norm(psi))
    expvals = np.array([float(np.real(np.vdot(psi, z_op @ psi))) for z_op in z_ops])
    return expvals, norm


def check_independent_reference() -> ValidationCheckResult:
    rng = np.random.default_rng(12345)
    max_err = 0.0
    max_norm_dev = 0.0
    worst_bound_violation = 0.0

    for n_qubits in (2, 4):
        for n_layers in (1, 3):
            for entangle in (False, True):
                scale = math.pi if n_qubits == 4 else math.pi / 2.0
                block = QuantumBlock(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    encoding_scale=scale,
                    entangle=entangle,
                )
                w_np = rng.uniform(-math.pi, math.pi, size=(n_layers, n_qubits, 3))
                x_np = rng.uniform(-1.0, 1.0, size=(5, n_qubits))
                with torch.no_grad():
                    block.weights.copy_(torch.tensor(w_np, dtype=torch.float64))
                    q_out = block(torch.tensor(x_np, dtype=torch.float64)).numpy()

                ref_rows = []
                for row in x_np:
                    expvals, norm = dense_statevector_expvals(
                        row, w_np, n_qubits, n_layers, scale, entangle
                    )
                    ref_rows.append(expvals)
                    max_norm_dev = max(max_norm_dev, abs(norm - 1.0))
                    worst_bound_violation = max(
                        worst_bound_violation, float(np.max(np.abs(expvals) - 1.0))
                    )
                ref_out = np.stack(ref_rows, axis=0)
                max_err = max(max_err, float(np.max(np.abs(q_out - ref_out))))

    passed = max_err <= 1e-9 and max_norm_dev <= 1e-12 and worst_bound_violation <= 1e-12
    detail = (
        f"max_err={max_err:.3e} (tol 1e-9), "
        f"max_norm_dev={max_norm_dev:.3e}, max_bound_excess={worst_bound_violation:.3e}"
    )
    return ValidationCheckResult(1, "Independent reference simulator", passed, detail)


def check_entanglement_switch() -> ValidationCheckResult:
    rng = np.random.default_rng(54321)
    n_qubits, n_layers = 4, 3
    w_init = torch.tensor(
        rng.uniform(-math.pi, math.pi, size=(n_layers, n_qubits, 3)),
        dtype=torch.float64,
    )
    x_base = torch.tensor(rng.uniform(-0.8, 0.8, size=(16, n_qubits)), dtype=torch.float64)

    block_off = QuantumBlock(n_qubits, n_layers, math.pi, entangle=False)
    block_on = QuantumBlock(n_qubits, n_layers, math.pi, entangle=True)
    with torch.no_grad():
        block_off.weights.copy_(w_init)
        block_on.weights.copy_(w_init)

        y_off_base = block_off(x_base)
        max_cross_off = 0.0
        for q in range(n_qubits):
            x_pert = x_base.clone()
            x_pert[:, q] += 0.7
            y_off_pert = block_off(x_pert)
            diff = (y_off_pert - y_off_base).abs()
            for other in range(n_qubits):
                if other != q:
                    max_cross_off = max(max_cross_off, float(diff[:, other].max().item()))

        entropy_off = block_off.compute_mean_single_qubit_entropy(x_base)

        y_on_base = block_on(x_base)
        x_pert0 = x_base.clone()
        x_pert0[:, 0] += 0.7
        y_on_pert0 = block_on(x_pert0)
        max_cross_on = float((y_on_pert0[:, 1:] - y_on_base[:, 1:]).abs().max().item())
        entropy_on = block_on.compute_mean_single_qubit_entropy(x_base)

    passed = (
        math.isfinite(entropy_off)
        and math.isfinite(entropy_on)
        and max_cross_off <= 1e-12
        and entropy_off <= 1e-10
        and max_cross_on > 1e-3
        and entropy_on > 1e-3
    )
    detail = (
        f"off_cross={max_cross_off:.3e}, off_entropy={entropy_off:.3e}, "
        f"on_cross={max_cross_on:.3e}, on_entropy={entropy_on:.3f}"
    )
    return ValidationCheckResult(2, "Entanglement switch", passed, detail)


def check_gradients() -> ValidationCheckResult:
    rng = np.random.default_rng(777)
    n_qubits, n_layers = 4, 3
    block = QuantumBlock(n_qubits, n_layers, math.pi, entangle=True)

    w_np = rng.uniform(-math.pi, math.pi, size=(n_layers, n_qubits, 3))
    x_t = torch.tensor(rng.uniform(-0.9, 0.9, size=(4, n_qubits)), dtype=torch.float64)
    coeff_t = torch.tensor(rng.normal(size=(4, n_qubits)), dtype=torch.float64)

    with torch.no_grad():
        block.weights.copy_(torch.tensor(w_np, dtype=torch.float64))

    block.zero_grad()
    loss = (block(x_t) * coeff_t).sum()
    loss.backward()
    grad_ad = block.weights.grad.detach().cpu().numpy().copy()

    grad_ps = np.zeros_like(grad_ad)
    shift = 0.5 * math.pi
    with torch.no_grad():
        base_w = block.weights.detach().clone()
        for l_idx in range(n_layers):
            for q_idx in range(n_qubits):
                for p_idx in range(3):
                    w_plus = base_w.clone()
                    w_plus[l_idx, q_idx, p_idx] += shift
                    block.weights.copy_(w_plus)
                    f_plus = float((block(x_t) * coeff_t).sum().item())

                    w_minus = base_w.clone()
                    w_minus[l_idx, q_idx, p_idx] -= shift
                    block.weights.copy_(w_minus)
                    f_minus = float((block(x_t) * coeff_t).sum().item())

                    grad_ps[l_idx, q_idx, p_idx] = 0.5 * (f_plus - f_minus)
        block.weights.copy_(base_w)

    denom = max(float(np.linalg.norm(grad_ps)), 1e-12)
    rel_err = float(np.linalg.norm(grad_ad - grad_ps) / denom)
    passed = rel_err <= 1e-6
    detail = f"rel_l2_err={rel_err:.3e} (tol 1e-6)"
    return ValidationCheckResult(3, "Autodiff vs parameter-shift gradients", passed, detail)


def check_parity_no_classical_mixing(cfg: ProtocolConfig | None = None) -> ValidationCheckResult:
    cfg = cfg or ProtocolConfig()
    max_offdiag = 0.0
    min_diag = float("inf")
    x_probe = torch.tensor(
        [[0.35, -0.82], [-1.10, 0.45], [0.05, 0.90]],
        dtype=torch.float64,
        requires_grad=True,
    )

    for variant in ("parity_entangled", "parity_no_entanglement"):
        model = build_parity_model(variant=variant, seed=0, cfg=cfg, device="cpu")
        with torch.no_grad():
            model.pre.scale.copy_(torch.tensor([1.73, -0.91], dtype=torch.float64))
            model.pre.shift.copy_(torch.tensor([0.22, -0.48], dtype=torch.float64))

        for b_idx in range(x_probe.shape[0]):
            for out_feat in range(2):
                x_in = x_probe[b_idx : b_idx + 1].detach().clone().requires_grad_(True)
                u = model.encode_for_middle(x_in)
                grad_vec = torch.autograd.grad(u[0, out_feat], x_in)[0][0]
                other_feat = 1 - out_feat
                max_offdiag = max(max_offdiag, float(abs(grad_vec[other_feat].item())))
                min_diag = min(min_diag, float(abs(grad_vec[out_feat].item())))

    passed = max_offdiag == 0.0 and min_diag > 0.0
    detail = f"max_cross_deriv={max_offdiag:.1f} (must be exact 0.0), min_diag_deriv={min_diag:.3f}"
    return ValidationCheckResult(4, "Parity no classical feature mixing", passed, detail)


def check_data_and_pairing(cfg: ProtocolConfig | None = None) -> ValidationCheckResult:
    cfg = cfg or ProtocolConfig()

    for name, fn in (("moons", make_moons_dataset), ("parity", make_parity_dataset)):
        ds = fn(0, cfg)
        assert ds.X_train.shape == (cfg.n_train, 2), f"{name} train shape mismatch"
        assert ds.X_test.shape == (cfg.n_test, 2), f"{name} test shape mismatch"
        assert int((ds.y_train == 0).sum()) == cfg.n_train // 2, f"{name} train balance mismatch"
        assert int((ds.y_test == 0).sum()) == cfg.n_test // 2, f"{name} test balance mismatch"
        assert np.all(np.abs(ds.X_train.mean(axis=0)) < 1e-10), f"{name} train mean not ~0"
        assert np.all(np.abs(ds.X_train.std(axis=0) - 1.0) < 1e-10), f"{name} train std not ~1"

        tr_set = {tuple(np.round(r, 12)) for r in ds.X_train_raw}
        te_set = {tuple(np.round(r, 12)) for r in ds.X_test_raw}
        assert len(tr_set & te_set) == 0, f"{name} duplicate points between train and test"

        if name == "parity":
            for X_r, y_r in ((ds.X_train_raw, ds.y_train), (ds.X_test_raw, ds.y_test)):
                interior = (np.abs(X_r[:, 0]) > 0.1) & (np.abs(X_r[:, 1]) > 0.1)
                sign_label = ((X_r[interior, 0] * X_r[interior, 1]) < 0).astype(np.int64)
                assert np.array_equal(sign_label, y_r[interior]), "parity labels violate sign rule"

    expected_moons = {
        "hybrid_entangled": (36, 58),
        "hybrid_no_entanglement": (36, 58),
        "classical_mlp": (40, 62),
        "no_middle": (0, 22),
    }
    moons_models = {v: build_moons_model(v, 0, cfg, "cpu") for v in MOONS_VARIANTS}
    ref_m = moons_models["hybrid_entangled"]
    for v, m in moons_models.items():
        assert torch.equal(m.in_linear.weight, ref_m.in_linear.weight), f"{v} in_linear weight mismatch"
        assert torch.equal(m.in_linear.bias, ref_m.in_linear.bias), f"{v} in_linear bias mismatch"
        assert torch.equal(m.out_linear.weight, ref_m.out_linear.weight), f"{v} out_linear weight mismatch"
        assert torch.equal(m.out_linear.bias, ref_m.out_linear.bias), f"{v} out_linear bias mismatch"
        pc = m.parameter_counts()
        assert (pc["middle"], pc["total"]) == expected_moons[v], f"{v} param count {pc}"

    assert torch.equal(
        moons_models["hybrid_entangled"].middle.weights,
        moons_models["hybrid_no_entanglement"].middle.weights,
    ), "Moons quantum variants must start with identical circuit weights"

    expected_parity = {
        "parity_entangled": (18, 28),
        "parity_no_entanglement": (18, 28),
        "parity_classical_mlp": (40, 62),
    }
    parity_models = {v: build_parity_model(v, 0, cfg, "cpu") for v in PARITY_VARIANTS}
    p_on = parity_models["parity_entangled"]
    p_off = parity_models["parity_no_entanglement"]
    assert torch.equal(p_on.pre.scale, p_off.pre.scale)
    assert torch.equal(p_on.pre.shift, p_off.pre.shift)
    assert torch.equal(p_on.middle.weights, p_off.middle.weights)
    assert torch.equal(p_on.out_linear.weight, p_off.out_linear.weight)
    for v, m in parity_models.items():
        pc = m.parameter_counts()
        assert (pc["middle"], pc["total"]) == expected_parity[v], f"{v} param count {pc}"

    p1 = torch.randperm(cfg.n_train, generator=torch.Generator().manual_seed(cfg.batch_seed_offset))
    p2 = torch.randperm(cfg.n_train, generator=torch.Generator().manual_seed(cfg.batch_seed_offset))
    assert torch.equal(p1, p2)

    detail = "moons(300/5000, params 36/36/40/0), parity(300/5000, params 18/18/40), pairing exact"
    return ValidationCheckResult(5, "Data, pairing, and parameter counts", True, detail)


def check_negative_control(cfg: ProtocolConfig | None = None) -> ValidationCheckResult:
    cfg = cfg or ProtocolConfig()
    test_accs: list[float] = []
    for seed in cfg.negative_control_seeds:
        ds = make_moons_dataset(seed, cfg)
        rng = np.random.default_rng(seed + 999)
        shuffled_y = rng.permutation(ds.y_train)
        shuffled_ds = type(ds)(
            X_train=ds.X_train,
            y_train=shuffled_y,
            X_test=ds.X_test,
            y_test=ds.y_test,
            X_train_raw=ds.X_train_raw,
            X_test_raw=ds.X_test_raw,
            entropy_indices=ds.entropy_indices,
        )
        model = build_moons_model("hybrid_entangled", seed, cfg, "cpu")
        res = train_and_evaluate(
            model=model,
            dataset=shuffled_ds,
            experiment="neg_control",
            variant="hybrid_entangled",
            seed=seed,
            epochs=cfg.moons_epochs,
            grid_resolution=cfg.grid_resolution_fast,
            cfg=cfg,
            device="cpu",
            save_frames=False,
        )
        test_accs.append(float(res.final_metrics["final_test_acc"]))

    mean_acc = float(np.mean(test_accs))
    dev = abs(mean_acc - 0.5)
    passed = dev <= cfg.negative_control_tolerance
    detail = f"mean_shuffled_test_acc={mean_acc:.4f} (|acc-0.5|={dev:.4f} <= {cfg.negative_control_tolerance:.2f})"
    return ValidationCheckResult(6, "Negative control (shuffled labels)", passed, detail)


ALL_VALIDATION_CHECKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def run_validation_checks(
    checks: tuple[int, ...] = (1, 2, 3),
    cfg: ProtocolConfig | None = None,
) -> list[ValidationCheckResult]:
    cfg = cfg or ProtocolConfig()
    registry: dict[int, Callable[[], ValidationCheckResult]] = {
        1: check_independent_reference,
        2: check_entanglement_switch,
        3: check_gradients,
        4: lambda: check_parity_no_classical_mixing(cfg),
        5: lambda: check_data_and_pairing(cfg),
        6: lambda: check_negative_control(cfg),
    }
    results: list[ValidationCheckResult] = []
    for cid in checks:
        res = registry[cid]()
        results.append(res)
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] Check {res.check_id} ({res.name}): {res.detail}")
    return results
