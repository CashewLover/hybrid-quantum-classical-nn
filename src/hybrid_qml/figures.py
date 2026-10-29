import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hybrid_qml.protocol import MOONS_VARIANTS, PARITY_VARIANTS, VARIANT_LABELS
from hybrid_qml.statistics import HypothesisResult, VariantSummary


VARIANT_STYLE: dict[str, dict[str, Any]] = {
    "hybrid_entangled": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "short": "Hybrid\n(entangled)",
    },
    "hybrid_no_entanglement": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-.",
        "short": "Hybrid\n(no ent.)",
    },
    "classical_mlp": {
        "color": "#009E73",
        "marker": "^",
        "linestyle": "--",
        "short": "Classical\nMLP",
    },
    "no_middle": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
        "short": "No middle\nblock",
    },
    "parity_entangled": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "short": "Parity\n(entangled)",
    },
    "parity_no_entanglement": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-.",
        "short": "Parity\n(ent. off)",
    },
    "parity_classical_mlp": {
        "color": "#009E73",
        "marker": "^",
        "linestyle": "--",
        "short": "Classical\nMLP (ref)",
    },
}

CLASS_STYLE = {
    0: {"color": "#0072B2", "marker": "o", "label": "Class 0"},
    1: {"color": "#D55E00", "marker": "^", "label": "Class 1"},
}


def _apply_clean_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)


def _draw_circuit_schematic(out_path: Path, manifest: dict[str, Any]) -> Path:
    proto = manifest["run_spec"]["protocol"]
    n_qubits = int(proto["n_qubits_moons"])
    n_layers = int(proto["n_layers"])

    fig, ax = plt.subplots(figsize=(9.5, 3.6), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    x_start, x_end = 0.6, 9.6
    y_positions = [n_qubits - 1 - q for q in range(n_qubits)]

    for q, y in enumerate(y_positions):
        ax.plot([x_start, x_end], [y, y], color="#222222", linewidth=1.3, zorder=1)
        ax.text(
            x_start - 0.15,
            y,
            rf"$|0\rangle$ ($q_{q}$)",
            ha="right",
            va="center",
            fontsize=9.5,
            color="#111111",
        )

    def box(cx: float, cy: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        rect = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5, color="#111111", zorder=4)

    layer_box = patches.FancyBboxPatch(
        (1.05, -0.82),
        5.55,
        n_qubits + 0.14,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor="#F8FAFC",
        edgecolor="#94A3B8",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(layer_box)
    ax.text(
        3.82,
        n_qubits - 0.52,
        f"Repeated for l = 1 ... L (L = {n_layers} layers; n = {n_qubits} moons / n = {int(proto['n_qubits_parity'])} parity)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#1E293B",
    )

    for q, y in enumerate(y_positions):
        box(1.85, y, 1.15, 0.52, rf"$R_Y(s \cdot x_{q})$", "#E6F2FA", "#0072B2")

    for q, y in enumerate(y_positions):
        box(3.35, y, 1.25, 0.52, rf"$\mathrm{{Rot}}(\phi,\theta,\omega)$", "#FFF2E6", "#D55E00")

    ring_x = [4.55 + 0.42 * q for q in range(n_qubits)]
    ring_box = patches.FancyBboxPatch(
        (4.30, -0.36),
        2.10,
        n_qubits - 0.52,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        facecolor="#F0FDF4",
        edgecolor="#009E73",
        linestyle=":",
        linewidth=1.1,
        zorder=1,
    )
    ax.add_patch(ring_box)
    ax.text(
        5.35,
        -0.50,
        "Entanglement switch (CNOT ring or I)",
        ha="center",
        va="top",
        fontsize=8,
        color="#00664B",
    )

    for q in range(n_qubits):
        c_y = y_positions[q]
        t_y = y_positions[(q + 1) % n_qubits]
        cx = ring_x[q]
        ax.plot([cx, cx], [c_y, t_y], color="#009E73", linewidth=1.3, zorder=2)
        ax.scatter([cx], [c_y], color="#009E73", s=26, zorder=4)
        circ = patches.Circle((cx, t_y), 0.14, facecolor="white", edgecolor="#009E73", linewidth=1.3, zorder=4)
        ax.add_patch(circ)
        ax.plot([cx - 0.11, cx + 0.11], [t_y, t_y], color="#009E73", linewidth=1.1, zorder=5)
        ax.plot([cx, cx], [t_y - 0.11, t_y + 0.11], color="#009E73", linewidth=1.1, zorder=5)

    ax.text(7.25, (n_qubits - 1) / 2, rf"$\times\, L={n_layers}$", ha="center", va="center", fontsize=11, fontweight="bold")

    for q, y in enumerate(y_positions):
        box(8.65, y, 1.15, 0.52, rf"$\langle Z_{q} \rangle$", "#F3E8F0", "#CC79A7")

    ax.set_xlim(-0.3, 10.0)
    ax.set_ylim(-1.05, n_qubits - 0.15)
    ax.set_title("Quantum middle block (gate-by-gate circuit)", fontsize=10.5, pad=6)

    fig.savefig(out_path, format="svg", facecolor="white")
    plt.close(fig)
    return out_path


def _plot_boundary_panel(
    ax: plt.Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    probs: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    title: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> Any:
    ax.set_facecolor("white")
    extent = [float(grid_x.min()), float(grid_x.max()), float(grid_y.min()), float(grid_y.max())]
    im = ax.imshow(
        probs,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        alpha=0.82,
    )
    xx, yy = np.meshgrid(grid_x, grid_y)
    ax.contour(xx, yy, probs, levels=[0.5], colors=["#111111"], linewidths=1.3, linestyles="-")

    for cls_val in (0, 1):
        mask = y_train == cls_val
        st = CLASS_STYLE[cls_val]
        ax.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            c=st["color"],
            marker=st["marker"],
            s=20,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            label=st["label"],
        )

    ax.set_title(title, fontsize=9.5)
    if show_xlabel:
        ax.set_xlabel("Standardized feature $x_1$", fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel("Standardized feature $x_2$", fontsize=8.5)
    ax.tick_params(labelsize=8)
    return im


def _draw_training_frames(
    fig_dir: Path,
    seed0_frames: dict[str, Any],
    seed0_data: dict[str, np.ndarray],
    seed0_record: dict[str, Any],
) -> tuple[Path, Path]:
    gif_path = fig_dir / "fig2_moons_training_frames.gif"
    final_png_path = fig_dir / "fig2_moons_training_final.png"

    grid_x = seed0_frames["grid_x"]
    grid_y = seed0_frames["grid_y"]
    X_tr = seed0_data["X_train"]
    y_tr = seed0_data["y_train"]
    frame_keys = sorted(
        (int(k.split("_")[-1]), k)
        for k in seed0_frames
        if k.startswith("frame_")
    )

    pil_frames: list[Image.Image] = []
    for idx, (ep, key) in enumerate(frame_keys):
        probs = seed0_frames[key]
        te_acc = float(seed0_record["history"]["test_acc"][ep - 1])
        tr_loss = float(seed0_record["history"]["train_loss"][ep - 1])

        fig, ax = plt.subplots(figsize=(5.0, 4.2), layout="constrained")
        fig.patch.set_facecolor("white")
        im = _plot_boundary_panel(
            ax,
            grid_x,
            grid_y,
            probs,
            X_tr,
            y_tr,
            title=f"Seed 0 Hybrid (entangled), epoch {ep}\ntrain loss = {tr_loss:.3f}, test acc = {te_acc:.3f}",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=0.9)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label("P(class = 1)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, facecolor="white")
        if idx == len(frame_keys) - 1:
            fig.savefig(final_png_path, format="png", dpi=260, facecolor="white")
        plt.close(fig)

        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

    if pil_frames:
        durations = [700] * max(0, len(pil_frames) - 1) + [1800]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
        )
    return gif_path, final_png_path


def _draw_moons_paired_accuracy(
    out_path: Path,
    raw_records: list[dict[str, Any]],
    summaries: list[VariantSummary],
    hypotheses: list[HypothesisResult],
    margin: float,
) -> Path:
    by_var_seed: dict[str, dict[int, float]] = {}
    for rec in raw_records:
        if rec["experiment"] == "e2_moons":
            by_var_seed.setdefault(rec["variant"], {})[int(rec["seed"])] = float(
                rec["final_metrics"]["final_test_acc"]
            )
    sum_map = {(s.experiment, s.variant): s for s in summaries}
    seeds = sorted(by_var_seed.get("hybrid_entangled", {}).keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")
    fig.patch.set_facecolor("white")

    _apply_clean_axes(ax1)
    x_pos = np.arange(len(MOONS_VARIANTS), dtype=np.float64)
    for s in seeds:
        ys = [by_var_seed[v][s] for v in MOONS_VARIANTS if s in by_var_seed.get(v, {})]
        if len(ys) == len(MOONS_VARIANTS):
            ax1.plot(x_pos, ys, color="#94A3B8", linewidth=0.75, alpha=0.45, zorder=1)

    rng = np.random.default_rng(42)
    for idx, var in enumerate(MOONS_VARIANTS):
        st = VARIANT_STYLE[var]
        vals = np.array([by_var_seed[var][s] for s in seeds], dtype=np.float64)
        jitter = rng.uniform(-0.07, 0.07, size=vals.size) if vals.size > 1 else np.zeros(1)
        ax1.scatter(
            np.full_like(vals, idx, dtype=np.float64) + jitter - 0.08,
            vals,
            color=st["color"],
            marker=st["marker"],
            s=24,
            alpha=0.65,
            zorder=2,
        )
        sm = sum_map[("e2_moons", var)]
        if sm.test_acc_ci_low is not None and sm.test_acc_ci_high is not None:
            yerr = [
                [sm.test_acc_mean - sm.test_acc_ci_low],
                [sm.test_acc_ci_high - sm.test_acc_mean],
            ]
            ax1.errorbar(
                [idx + 0.16],
                [sm.test_acc_mean],
                yerr=yerr,
                fmt=st["marker"],
                color="#111111",
                markerfacecolor=st["color"],
                markersize=6.5,
                capsize=4,
                linewidth=1.5,
                zorder=4,
            )
        else:
            ax1.scatter(
                [idx + 0.16],
                [sm.test_acc_mean],
                color=st["color"],
                edgecolors="#111111",
                marker=st["marker"],
                s=52,
                zorder=4,
            )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([VARIANT_STYLE[v]["short"] for v in MOONS_VARIANTS], fontsize=8.5)
    ax1.set_ylabel("Final-epoch test accuracy", fontsize=9.5)
    ax1.set_title(f"(a) Moons paired test accuracy (N = {len(seeds)} seeds)", fontsize=10)

    _apply_clean_axes(ax2)
    h_moons = [h for h in hypotheses if h.hypothesis_id in {"H1", "H2"}]
    ax2.axhspan(-margin, margin, color="#E2E8F0", alpha=0.65, label=f"Equivalence zone (±{margin:.2f})")
    ax2.axhline(0.0, color="#334155", linestyle="--", linewidth=1.0)
    ax2.axhline(-margin, color="#64748B", linestyle=":", linewidth=0.9)
    ax2.axhline(margin, color="#64748B", linestyle=":", linewidth=0.9)

    for idx, h in enumerate(h_moons):
        diffs = np.array(
            [by_var_seed[h.variant_a][s] - by_var_seed[h.variant_b][s] for s in seeds],
            dtype=np.float64,
        )
        jitter = rng.uniform(-0.06, 0.06, size=diffs.size) if diffs.size > 1 else np.zeros(1)
        col = "#0072B2" if h.hypothesis_id == "H1" else "#D55E00"
        mk = "o" if h.hypothesis_id == "H1" else "s"
        ax2.scatter(
            np.full_like(diffs, idx, dtype=np.float64) + jitter - 0.10,
            diffs,
            color=col,
            marker=mk,
            s=26,
            alpha=0.65,
            zorder=2,
        )
        if h.ci_low is not None and h.ci_high is not None:
            yerr = [[h.mean_diff - h.ci_low], [h.ci_high - h.mean_diff]]
            ax2.errorbar(
                [idx + 0.14],
                [h.mean_diff],
                yerr=yerr,
                fmt=mk,
                color="#111111",
                markerfacecolor=col,
                markersize=7,
                capsize=4.5,
                linewidth=1.6,
                zorder=4,
            )
        else:
            ax2.scatter([idx + 0.14], [h.mean_diff], color=col, edgecolors="#111111", marker=mk, s=55, zorder=4)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        ["H1: Hybrid(ent)\n- Classical MLP", "H2: Hybrid(ent)\n- Hybrid(no ent)"],
        fontsize=8.5,
    )
    ax2.set_ylabel("Paired accuracy difference (A - B)", fontsize=9.5)
    ax2.set_title(f"(b) Paired differences & 95% bootstrap CI (N = {len(seeds)})", fontsize=10)
    ax2.legend(loc="best", fontsize=8, frameon=True, facecolor="white")

    fig.savefig(out_path, format="svg", facecolor="white")
    plt.close(fig)
    return out_path


def _plot_curve_group(
    ax: plt.Axes,
    raw_records: list[dict[str, Any]],
    experiment: str,
    variants: list[str],
    title: str,
) -> int:
    _apply_clean_axes(ax)
    n_seeds = 0
    for var in variants:
        curves = [
            rec["history"]["test_acc"]
            for rec in raw_records
            if rec["experiment"] == experiment and rec["variant"] == var
        ]
        if not curves:
            continue
        arr = np.asarray(curves, dtype=np.float64)
        n_seeds = max(n_seeds, arr.shape[0])
        epochs = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros_like(mean)
        st = VARIANT_STYLE[var]
        ax.plot(
            epochs,
            mean,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.8,
            marker=st["marker"] if arr.shape[1] <= 16 else None,
            markersize=4,
            label=VARIANT_LABELS[var],
        )
        if arr.shape[0] >= 2:
            ax.fill_between(
                epochs,
                np.clip(mean - sd, 0.0, 1.0),
                np.clip(mean + sd, 0.0, 1.0),
                color=st["color"],
                alpha=0.14,
            )
    ax.set_xlabel("Epoch", fontsize=9.5)
    ax.set_ylabel("Test accuracy (mean ± 1 SD)" if n_seeds >= 2 else "Test accuracy", fontsize=9.5)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")
    return n_seeds


def _draw_moons_learning_curves(
    out_path: Path,
    raw_records: list[dict[str, Any]],
    n_seeds: int,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained", sharey=True)
    fig.patch.set_facecolor("white")
    _plot_curve_group(
        ax1,
        raw_records,
        "e2_moons",
        ["hybrid_entangled", "classical_mlp", "no_middle"],
        f"(a) Quantum block vs Classical MLP vs None (N = {n_seeds})",
    )
    _plot_curve_group(
        ax2,
        raw_records,
        "e2_moons",
        ["hybrid_entangled", "hybrid_no_entanglement"],
        f"(b) Entangled vs Entanglement-off (N = {n_seeds})",
    )
    ax2.set_ylabel("")
    fig.savefig(out_path, format="svg", facecolor="white")
    plt.close(fig)
    return out_path


def _draw_moons_boundaries_seed0(
    out_path: Path,
    seed0_frames: dict[str, dict[str, Any]],
    seed0_data: dict[str, np.ndarray],
    seed0_records: dict[str, dict[str, Any]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2), layout="constrained", sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    X_tr = seed0_data["X_train"]
    y_tr = seed0_data["y_train"]

    im_ref = None
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]
    for idx, (ax, var) in enumerate(zip(axes.flat, MOONS_VARIANTS)):
        fr = seed0_frames[var]
        rec = seed0_records[var]
        ep = int(rec["epochs"])
        te_acc = float(rec["final_metrics"]["final_test_acc"])
        title = f"{panel_tags[idx]} {VARIANT_LABELS[var]}\nseed 0, epoch {ep} (test acc = {te_acc:.3f})"
        im_ref = _plot_boundary_panel(
            ax,
            fr["grid_x"],
            fr["grid_y"],
            fr[f"frame_{ep}"],
            X_tr,
            y_tr,
            title=title,
            show_ylabel=(idx % 2 == 0),
            show_xlabel=(idx >= 2),
        )
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label("Predicted P(class = 1)", fontsize=9)
    fig.savefig(out_path, format="png", dpi=260, facecolor="white")
    plt.close(fig)
    return out_path


def _draw_parity_accuracy_and_curves(
    out_path: Path,
    raw_records: list[dict[str, Any]],
    summaries: list[VariantSummary],
    additive_ceiling: float,
) -> Path:
    by_var_seed: dict[str, dict[int, float]] = {}
    for rec in raw_records:
        if rec["experiment"] == "e3_parity":
            by_var_seed.setdefault(rec["variant"], {})[int(rec["seed"])] = float(
                rec["final_metrics"]["final_test_acc"]
            )
    sum_map = {(s.experiment, s.variant): s for s in summaries}
    seeds = sorted(by_var_seed.get("parity_entangled", {}).keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")
    fig.patch.set_facecolor("white")

    _apply_clean_axes(ax1)
    x_pos = np.arange(len(PARITY_VARIANTS), dtype=np.float64)
    ax1.axhline(0.5, color="#475569", linestyle="--", linewidth=1.1, label="Chance (0.50)")
    ax1.axhline(additive_ceiling, color="#991B1B", linestyle=":", linewidth=1.1, label=f"Additive ceiling ({additive_ceiling:.2f})")

    for s in seeds:
        ys = [by_var_seed[v][s] for v in PARITY_VARIANTS if s in by_var_seed.get(v, {})]
        if len(ys) == len(PARITY_VARIANTS):
            ax1.plot(x_pos, ys, color="#94A3B8", linewidth=0.75, alpha=0.45, zorder=1)

    rng = np.random.default_rng(99)
    for idx, var in enumerate(PARITY_VARIANTS):
        st = VARIANT_STYLE[var]
        vals = np.array([by_var_seed[var][s] for s in seeds], dtype=np.float64)
        jitter = rng.uniform(-0.06, 0.06, size=vals.size) if vals.size > 1 else np.zeros(1)
        ax1.scatter(
            np.full_like(vals, idx, dtype=np.float64) + jitter - 0.08,
            vals,
            color=st["color"],
            marker=st["marker"],
            s=25,
            alpha=0.65,
            zorder=2,
        )
        sm = sum_map[("e3_parity", var)]
        if sm.test_acc_ci_low is not None and sm.test_acc_ci_high is not None:
            yerr = [
                [sm.test_acc_mean - sm.test_acc_ci_low],
                [sm.test_acc_ci_high - sm.test_acc_mean],
            ]
            ax1.errorbar(
                [idx + 0.15],
                [sm.test_acc_mean],
                yerr=yerr,
                fmt=st["marker"],
                color="#111111",
                markerfacecolor=st["color"],
                markersize=6.5,
                capsize=4,
                linewidth=1.5,
                zorder=4,
            )
        else:
            ax1.scatter(
                [idx + 0.15],
                [sm.test_acc_mean],
                color=st["color"],
                edgecolors="#111111",
                marker=st["marker"],
                s=52,
                zorder=4,
            )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([VARIANT_STYLE[v]["short"] for v in PARITY_VARIANTS], fontsize=8.5)
    ax1.set_ylim(0.35, 1.05)
    ax1.set_ylabel("Final-epoch test accuracy", fontsize=9.5)
    ax1.set_title(f"(a) Parity test accuracy (N = {len(seeds)} seeds)", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")

    _plot_curve_group(
        ax2,
        raw_records,
        "e3_parity",
        list(PARITY_VARIANTS),
        f"(b) Parity learning curves (N = {len(seeds)} seeds)",
    )
    ax2.axhline(0.5, color="#475569", linestyle="--", linewidth=1.1, label="Chance (0.50)")
    ax2.set_ylim(0.35, 1.05)
    ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")

    fig.savefig(out_path, format="svg", facecolor="white")
    plt.close(fig)
    return out_path


def _draw_parity_boundaries_seed0(
    out_path: Path,
    seed0_frames: dict[str, dict[str, Any]],
    seed0_data: dict[str, np.ndarray],
    seed0_records: dict[str, dict[str, Any]],
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), layout="constrained", sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    X_tr = seed0_data["X_train"]
    y_tr = seed0_data["y_train"]

    im_ref = None
    panel_tags = ["(a)", "(b)", "(c)"]
    for idx, (ax, var) in enumerate(zip(axes, PARITY_VARIANTS)):
        fr = seed0_frames[var]
        rec = seed0_records[var]
        ep = int(rec["epochs"])
        te_acc = float(rec["final_metrics"]["final_test_acc"])
        title = f"{panel_tags[idx]} {VARIANT_LABELS[var]}\nseed 0, epoch {ep} (test acc = {te_acc:.3f})"
        im_ref = _plot_boundary_panel(
            ax,
            fr["grid_x"],
            fr["grid_y"],
            fr[f"frame_{ep}"],
            X_tr,
            y_tr,
            title=title,
            show_ylabel=(idx == 0),
            show_xlabel=True,
        )
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="white")

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes.tolist(), shrink=0.86)
        cbar.set_label("Predicted P(class = 1)", fontsize=9)
    fig.savefig(out_path, format="png", dpi=260, facecolor="white")
    plt.close(fig)
    return out_path


def generate_all_figures(
    run_dir: Path,
    manifest: dict[str, Any],
    raw_records: list[dict[str, Any]],
    summaries: list[VariantSummary],
    hypotheses: list[HypothesisResult],
) -> dict[str, Path]:
    fig_dir = run_dir / "figures"
    raw_dir = run_dir / "raw"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = int(manifest["run_spec"]["seeds"])
    proto = manifest["run_spec"]["protocol"]

    p1 = _draw_circuit_schematic(fig_dir / "fig1_circuit_schematic.svg", manifest)
    p3 = _draw_moons_paired_accuracy(
        fig_dir / "fig3_moons_paired_accuracy.svg",
        raw_records,
        summaries,
        hypotheses,
        margin=float(proto["equivalence_margin"]),
    )
    p4 = _draw_moons_learning_curves(
        fig_dir / "fig4_moons_learning_curves.svg",
        raw_records,
        n_seeds=n_seeds,
    )
    p6 = _draw_parity_accuracy_and_curves(
        fig_dir / "fig6_parity_accuracy_and_curves.svg",
        raw_records,
        summaries,
        additive_ceiling=float(proto["additive_ceiling"]),
    )

    p2_gif = fig_dir / "fig2_moons_training_frames.gif"
    p2_png = fig_dir / "fig2_moons_training_final.png"
    p5 = fig_dir / "fig5_moons_boundaries_seed0.png"
    p7 = fig_dir / "fig7_parity_boundaries_seed0.png"

    has_npz = (
        (raw_dir / "e2_moons__seed000_data.npz").exists()
        and (raw_dir / "e3_parity__seed000_data.npz").exists()
        and all((raw_dir / f"e2_moons__{v}__seed000_frames.npz").exists() for v in MOONS_VARIANTS)
        and all((raw_dir / f"e3_parity__{v}__seed000_frames.npz").exists() for v in PARITY_VARIANTS)
    )
    if has_npz:
        moons_seed0_data = dict(np.load(raw_dir / "e2_moons__seed000_data.npz"))
        parity_seed0_data = dict(np.load(raw_dir / "e3_parity__seed000_data.npz"))

        moons_frames: dict[str, dict[str, Any]] = {}
        moons_s0_rec: dict[str, dict[str, Any]] = {}
        for var in MOONS_VARIANTS:
            moons_frames[var] = dict(np.load(raw_dir / f"e2_moons__{var}__seed000_frames.npz"))
            moons_s0_rec[var] = next(
                r for r in raw_records if r["experiment"] == "e2_moons" and r["variant"] == var and int(r["seed"]) == 0
            )

        parity_frames: dict[str, dict[str, Any]] = {}
        parity_s0_rec: dict[str, dict[str, Any]] = {}
        for var in PARITY_VARIANTS:
            parity_frames[var] = dict(np.load(raw_dir / f"e3_parity__{var}__seed000_frames.npz"))
            parity_s0_rec[var] = next(
                r for r in raw_records if r["experiment"] == "e3_parity" and r["variant"] == var and int(r["seed"]) == 0
            )

        p2_gif, p2_png = _draw_training_frames(
            fig_dir,
            moons_frames["hybrid_entangled"],
            moons_seed0_data,
            moons_s0_rec["hybrid_entangled"],
        )
        p5 = _draw_moons_boundaries_seed0(
            p5,
            moons_frames,
            moons_seed0_data,
            moons_s0_rec,
        )
        p7 = _draw_parity_boundaries_seed0(
            p7,
            parity_frames,
            parity_seed0_data,
            parity_s0_rec,
        )
    elif not (p2_gif.exists() and p2_png.exists() and p5.exists() and p7.exists()):
        raise FileNotFoundError(
            f"Missing seed-0 boundary caches or pre-rendered boundary figures in {run_dir}"
        )

    return {
        "fig1_schematic": p1,
        "fig2_training_gif": p2_gif,
        "fig2_training_final": p2_png,
        "fig3_moons_paired": p3,
        "fig4_moons_curves": p4,
        "fig5_moons_boundaries": p5,
        "fig6_parity_acc_curves": p6,
        "fig7_parity_boundaries": p7,
    }
