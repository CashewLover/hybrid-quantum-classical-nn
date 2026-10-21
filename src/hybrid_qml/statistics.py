from dataclasses import dataclass
from typing import Any

import numpy as np

from hybrid_qml.protocol import (
    HYPOTHESIS_SPECS,
    MOONS_VARIANTS,
    PARITY_VARIANTS,
    VARIANT_LABELS,
    ProtocolConfig,
)


@dataclass(frozen=True)
class VariantSummary:
    experiment: str
    variant: str
    label: str
    n_seeds: int
    test_acc_mean: float
    test_acc_sd: float
    test_acc_ci_low: float | None
    test_acc_ci_high: float | None
    test_loss_mean: float
    test_loss_sd: float
    train_acc_mean: float
    train_acc_sd: float
    middle_params: int
    total_params: int
    entropy_mean: float | None
    entropy_sd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "variant": self.variant,
            "label": self.label,
            "n_seeds": self.n_seeds,
            "test_acc_mean": self.test_acc_mean,
            "test_acc_sd": self.test_acc_sd,
            "test_acc_ci_low": self.test_acc_ci_low,
            "test_acc_ci_high": self.test_acc_ci_high,
            "test_loss_mean": self.test_loss_mean,
            "test_loss_sd": self.test_loss_sd,
            "train_acc_mean": self.train_acc_mean,
            "train_acc_sd": self.train_acc_sd,
            "middle_params": self.middle_params,
            "total_params": self.total_params,
            "entropy_mean": self.entropy_mean,
            "entropy_sd": self.entropy_sd,
        }


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis_id: str
    experiment: str
    variant_a: str
    variant_b: str
    label_a: str
    label_b: str
    prediction: str
    n_seeds: int
    mean_diff: float
    sd_diff: float | None
    ci_low: float | None
    ci_high: float | None
    margin: float
    verdict_code: str
    verdict: str
    consistent: bool | None
    preliminary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "experiment": self.experiment,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "prediction": self.prediction,
            "n_seeds": self.n_seeds,
            "mean_diff": self.mean_diff,
            "sd_diff": self.sd_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "margin": self.margin,
            "verdict_code": self.verdict_code,
            "verdict": self.verdict,
            "consistent": self.consistent,
            "preliminary": self.preliminary,
        }


def percentile_bootstrap_ci(
    values: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return mean, sd


def summarize_variants(
    raw_records: list[dict[str, Any]],
    cfg: ProtocolConfig,
) -> list[VariantSummary]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in raw_records:
        key = (str(rec["experiment"]), str(rec["variant"]))
        grouped.setdefault(key, []).append(rec)

    ordered_keys: list[tuple[str, str]] = [
        ("e2_moons", v) for v in MOONS_VARIANTS if ("e2_moons", v) in grouped
    ] + [
        ("e3_parity", v) for v in PARITY_VARIANTS if ("e3_parity", v) in grouped
    ]
    for key in sorted(grouped):
        if key not in ordered_keys:
            ordered_keys.append(key)

    summaries: list[VariantSummary] = []
    for idx, (exp, var) in enumerate(ordered_keys):
        rows = sorted(grouped[(exp, var)], key=lambda r: int(r["seed"]))
        te_accs = [float(r["final_metrics"]["final_test_acc"]) for r in rows]
        te_losses = [float(r["final_metrics"]["final_test_loss"]) for r in rows]
        tr_accs = [float(r["final_metrics"]["final_train_acc"]) for r in rows]
        ent_vals = [
            float(r["final_metrics"]["mean_single_qubit_entropy"])
            for r in rows
            if r["final_metrics"].get("mean_single_qubit_entropy") is not None
        ]

        acc_mean, acc_sd = _mean_sd(te_accs)
        loss_mean, loss_sd = _mean_sd(te_losses)
        tr_mean, tr_sd = _mean_sd(tr_accs)

        if len(te_accs) >= 2:
            ci_lo, ci_hi = percentile_bootstrap_ci(
                np.asarray(te_accs, dtype=np.float64),
                cfg.bootstrap_resamples,
                cfg.bootstrap_seed + 100 + idx,
            )
        else:
            ci_lo, ci_hi = None, None

        ent_mean: float | None = None
        ent_sd: float | None = None
        if ent_vals:
            ent_mean, ent_sd = _mean_sd(ent_vals)

        first_meta = rows[0]["final_metrics"]
        summaries.append(
            VariantSummary(
                experiment=exp,
                variant=var,
                label=VARIANT_LABELS.get(var, var),
                n_seeds=len(rows),
                test_acc_mean=acc_mean,
                test_acc_sd=acc_sd,
                test_acc_ci_low=ci_lo,
                test_acc_ci_high=ci_hi,
                test_loss_mean=loss_mean,
                test_loss_sd=loss_sd,
                train_acc_mean=tr_mean,
                train_acc_sd=tr_sd,
                middle_params=int(first_meta["middle_params"]),
                total_params=int(first_meta["total_params"]),
                entropy_mean=ent_mean,
                entropy_sd=ent_sd,
            )
        )
    return summaries


def evaluate_hypotheses(
    raw_records: list[dict[str, Any]],
    preset: str,
    cfg: ProtocolConfig,
) -> tuple[list[HypothesisResult], dict[str, Any]]:
    by_exp_var_seed: dict[tuple[str, str], dict[int, float]] = {}
    for rec in raw_records:
        key = (str(rec["experiment"]), str(rec["variant"]))
        by_exp_var_seed.setdefault(key, {})[int(rec["seed"])] = float(
            rec["final_metrics"]["final_test_acc"]
        )

    parity_off_scores = list(
        by_exp_var_seed.get(("e3_parity", "parity_no_entanglement"), {}).values()
    )
    parity_off_mean = float(np.mean(parity_off_scores)) if parity_off_scores else None
    additive_limit = cfg.additive_ceiling + cfg.additive_leak_tolerance
    additive_ceiling_exceeded = (
        parity_off_mean is not None and parity_off_mean > additive_limit
    )

    exp_map = {"E2 (Moons)": "e2_moons", "E3 (Parity)": "e3_parity"}
    delta = cfg.equivalence_margin
    results: list[HypothesisResult] = []

    for idx, spec in enumerate(HYPOTHESIS_SPECS):
        exp_key = exp_map[spec["experiment"]]
        map_a = by_exp_var_seed.get((exp_key, spec["variant_a"]), {})
        map_b = by_exp_var_seed.get((exp_key, spec["variant_b"]), {})
        paired_seeds = sorted(set(map_a) & set(map_b))

        if not paired_seeds:
            results.append(
                HypothesisResult(
                    hypothesis_id=spec["id"],
                    experiment=spec["experiment"],
                    variant_a=spec["variant_a"],
                    variant_b=spec["variant_b"],
                    label_a=spec["label_a"],
                    label_b=spec["label_b"],
                    prediction=spec["prediction"],
                    n_seeds=0,
                    mean_diff=0.0,
                    sd_diff=None,
                    ci_low=None,
                    ci_high=None,
                    margin=delta,
                    verdict_code="missing",
                    verdict="Unavailable (missing raw files)",
                    consistent=None,
                    preliminary=True,
                )
            )
            continue

        diffs = np.array(
            [map_a[s] - map_b[s] for s in paired_seeds], dtype=np.float64
        )
        n_seeds = int(diffs.size)
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1)) if n_seeds >= 2 else None
        preliminary = n_seeds < 10

        if preset == "smoke" or n_seeds < 2:
            ci_lo: float | None = None
            ci_hi: float | None = None
            verdict_code = "smoke_no_verdict"
            verdict = "No verdict (smoke preset)" if preset == "smoke" else "No verdict (n < 2)"
            consistent: bool | None = None
        else:
            ci_lo, ci_hi = percentile_bootstrap_ci(
                diffs,
                cfg.bootstrap_resamples,
                cfg.bootstrap_seed + idx,
            )
            if spec["id"] == "H3" and additive_ceiling_exceeded:
                verdict_code = "invalid_leak"
                verdict = "Invalid (additive ceiling exceeded)"
                consistent = False
            else:
                if ci_lo > -delta and ci_hi < delta:
                    verdict_code = "equivalent"
                    base = "Practically equivalent"
                elif ci_lo > 0.0:
                    verdict_code = "a_higher"
                    base = "A higher (> margin)" if ci_lo > delta else "A higher"
                elif ci_hi < 0.0:
                    verdict_code = "b_higher"
                    base = "B higher"
                else:
                    verdict_code = "inconclusive"
                    base = "Inconclusive"

                verdict = f"Preliminary: {base}" if preliminary else base
                if spec["id"] in {"H1", "H2"}:
                    consistent = verdict_code == "equivalent"
                else:
                    consistent = verdict_code == "a_higher" and ci_lo > delta

        results.append(
            HypothesisResult(
                hypothesis_id=spec["id"],
                experiment=spec["experiment"],
                variant_a=spec["variant_a"],
                variant_b=spec["variant_b"],
                label_a=spec["label_a"],
                label_b=spec["label_b"],
                prediction=spec["prediction"],
                n_seeds=n_seeds,
                mean_diff=mean_diff,
                sd_diff=sd_diff,
                ci_low=ci_lo,
                ci_high=ci_hi,
                margin=delta,
                verdict_code=verdict_code,
                verdict=verdict,
                consistent=consistent,
                preliminary=preliminary,
            )
        )

    flags = {
        "additive_ceiling_exceeded": additive_ceiling_exceeded,
        "parity_off_mean_acc": parity_off_mean,
        "additive_leak_limit": additive_limit,
    }
    return results, flags


def generate_summary_sentences(
    summaries: list[VariantSummary],
    hypotheses: list[HypothesisResult],
    flags: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    run_spec = manifest["run_spec"]
    preset = str(run_spec["preset"])
    seeds = int(run_spec["seeds"])
    is_confirmatory = bool(run_spec["is_confirmatory"])
    is_preliminary = bool(run_spec["is_preliminary"])
    device = str(run_spec["device"]).upper()

    mode_phrase = "confirmatory" if is_confirmatory else "exploratory"
    seed_word = "seed" if seeds == 1 else "seeds"
    sentences: list[str] = [
        f"Results from a {mode_phrase} {preset} run over {seeds} paired {seed_word} on {device}."
    ]

    if preset == "smoke":
        sentences.append(
            "The smoke preset runs a single seed for 2 epochs to check that the code works end to end, so no statistical verdicts are reported."
        )
    elif is_preliminary:
        sentences.append(
            f"Since fewer than 10 seeds were used (N = {seeds}), the verdicts below are marked preliminary."
        )

    for h in hypotheses:
        if h.n_seeds == 0:
            sentences.append(
                f"No paired run files were found for {h.hypothesis_id} ({h.label_a} vs. {h.label_b})."
            )
            continue
        if h.ci_low is None or h.ci_high is None:
            sentences.append(
                f"For {h.hypothesis_id} ({h.label_a} minus {h.label_b}), the single-seed difference was {h.mean_diff:+.4f} ({h.verdict.lower()})."
            )
        else:
            match_txt = (
                "matches"
                if h.consistent
                else "does not match"
            )
            sentences.append(
                f"For {h.hypothesis_id} ({h.label_a} minus {h.label_b}, N = {h.n_seeds}), the mean paired accuracy difference was "
                f"{h.mean_diff:+.4f} with a 95% bootstrap interval of [{h.ci_low:+.4f}, {h.ci_high:+.4f}] against the \u00b1{h.margin:.2f} margin. "
                f"The resulting verdict is \"{h.verdict}\", which {match_txt} the advance prediction ({h.prediction.lower()})."
            )

    by_key = {(s.experiment, s.variant): s for s in summaries}
    p_ref = by_key.get(("e3_parity", "parity_classical_mlp"))
    p_off = by_key.get(("e3_parity", "parity_no_entanglement"))
    p_on = by_key.get(("e3_parity", "parity_entangled"))
    if p_off is not None and p_on is not None and p_ref is not None:
        leak_note = (
            f"above the {flags['additive_leak_limit']:.2f} leak check"
            if flags["additive_ceiling_exceeded"]
            else f"well below the {flags['additive_leak_limit']:.2f} additive-ceiling check"
        )
        sentences.append(
            f"On the parity task, the entangled two-qubit model averaged {p_on.test_acc_mean:.4f} test accuracy (SD {p_on.test_acc_sd:.4f}), "
            f"while turning entanglement off dropped accuracy to {p_off.test_acc_mean:.4f} (SD {p_off.test_acc_sd:.4f}, {leak_note}). "
            f"The unconstrained classical MLP reference reached {p_ref.test_acc_mean:.4f} (SD {p_ref.test_acc_sd:.4f})."
        )

    m_on = by_key.get(("e2_moons", "hybrid_entangled"))
    m_off = by_key.get(("e2_moons", "hybrid_no_entanglement"))
    if (
        m_on is not None
        and m_off is not None
        and m_on.entropy_mean is not None
        and m_off.entropy_mean is not None
    ):
        sentences.append(
            f"Single-qubit von Neumann entropy on two moons averaged {m_on.entropy_mean:.4f} bits (SD {m_on.entropy_sd:.4f}) "
            f"with the CNOT ring enabled and {m_off.entropy_mean:.4f} bits (SD {m_off.entropy_sd:.4f}) with it disabled."
        )

    return sentences
