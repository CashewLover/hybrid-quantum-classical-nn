import base64
import csv
import html
import json
from pathlib import Path
from typing import Any

from hybrid_qml.figures import generate_all_figures
from hybrid_qml.protocol import ProtocolConfig
from hybrid_qml.statistics import (
    HypothesisResult,
    VariantSummary,
    evaluate_hypotheses,
    generate_summary_sentences,
    summarize_variants,
)


VERIFIED_REFERENCES: tuple[str, ...] = (
    "Bowles, J., Ahmed, S., & Schuld, M. (2024). Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059.",
    "Bergholm, V., Izaac, J., Schuld, M., et al. (2018). PennyLane: Automatic differentiation of hybrid quantum-classical computations. arXiv:1811.04968.",
    "Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS 2019 (arXiv:1912.01703).",
    "P\u00e9rez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. Quantum, 4, 226 (arXiv:1907.02085).",
    "Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830 (arXiv:1201.0490).",
)


def load_run_folder(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw/ directory in {run_dir}")

    raw_records: list[dict[str, Any]] = []
    for json_path in sorted(raw_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            raw_records.append(json.load(f))
    if not raw_records:
        raise FileNotFoundError(f"No raw run records found in {raw_dir}")
    return manifest, raw_records


def _fmt_opt(val: float | None, digits: int = 4) -> str:
    return "-" if val is None else f"{val:.{digits}f}"


def _fmt_ci(lo: float | None, hi: float | None, signed: bool = False) -> str:
    if lo is None or hi is None:
        return "-"
    fmt = "+.4f" if signed else ".4f"
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


def _write_tables(
    run_dir: Path,
    summaries: list[VariantSummary],
    hypotheses: list[HypothesisResult],
) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    var_cols = [
        "experiment",
        "variant",
        "label",
        "n_seeds",
        "test_acc_mean",
        "test_acc_sd",
        "test_acc_ci_low",
        "test_acc_ci_high",
        "test_loss_mean",
        "test_loss_sd",
        "train_acc_mean",
        "train_acc_sd",
        "middle_params",
        "total_params",
        "entropy_mean",
        "entropy_sd",
    ]
    with (tables_dir / "variant_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=var_cols)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s.to_dict())

    md_lines = [
        "| Experiment | Variant | N | Test Acc (Mean ± SD) | 95% CI | Test Loss | Train Acc | Params (Mid / Total) | 1-Qubit Entropy (bits) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        ent_txt = (
            "-"
            if s.entropy_mean is None
            else f"{s.entropy_mean:.4f} ± {s.entropy_sd:.4f}"
        )
        md_lines.append(
            f"| {s.experiment} | {s.label} | {s.n_seeds} | "
            f"{s.test_acc_mean:.4f} ± {s.test_acc_sd:.4f} | "
            f"{_fmt_ci(s.test_acc_ci_low, s.test_acc_ci_high)} | "
            f"{s.test_loss_mean:.4f} | {s.train_acc_mean:.4f} | "
            f"{s.middle_params} / {s.total_params} | {ent_txt} |"
        )
    (tables_dir / "variant_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    hyp_cols = [
        "hypothesis_id",
        "experiment",
        "variant_a",
        "variant_b",
        "prediction",
        "n_seeds",
        "mean_diff",
        "sd_diff",
        "ci_low",
        "ci_high",
        "margin",
        "verdict",
        "consistent",
        "preliminary",
    ]
    with (tables_dir / "hypotheses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hyp_cols)
        writer.writeheader()
        for h in hypotheses:
            d = h.to_dict()
            writer.writerow({k: d[k] for k in hyp_cols})

    h_md = [
        "| ID | Comparison (A - B) | N | Prediction | Mean Diff | 95% Bootstrap CI | Margin | Verdict | Consistent |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for h in hypotheses:
        cons = "-" if h.consistent is None else ("Yes" if h.consistent else "No")
        h_md.append(
            f"| {h.hypothesis_id} | {h.label_a} - {h.label_b} | {h.n_seeds} | {h.prediction} | "
            f"{h.mean_diff:+.4f} | {_fmt_ci(h.ci_low, h.ci_high, signed=True)} | "
            f"±{h.margin:.2f} | {h.verdict} | {cons} |"
        )
    (tables_dir / "hypotheses.md").write_text("\n".join(h_md) + "\n", encoding="utf-8")


def _encode_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".gif": "image/gif",
    }[suffix]
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_threat_flags(
    manifest: dict[str, Any],
    flags: dict[str, Any],
) -> list[tuple[str, bool, str]]:
    run_spec = manifest["run_spec"]
    preset = str(run_spec["preset"])
    seeds = int(run_spec["seeds"])
    is_conf = bool(run_spec["is_confirmatory"])
    device = str(run_spec["device"])
    git_info = manifest.get("git", {})
    dirty = bool(git_info.get("dirty", False))
    commit = str(git_info.get("commit", "unavailable"))
    leak = bool(flags.get("additive_ceiling_exceeded", False))

    return [
        (
            "Sample size",
            seeds < 10,
            f"Only {seeds} seeds run (< 10), so verdicts are marked preliminary."
            if seeds < 10
            else f"{seeds} paired seeds run (meets >= 10 requirement).",
        ),
        (
            "Run mode",
            not is_conf,
            "Exploratory (non-standard preset or overridden CLI flag)."
            if not is_conf
            else "Confirmatory (unmodified standard preset).",
        ),
        (
            "Epoch schedule",
            preset in {"smoke", "quick"},
            f"Shortened '{preset}' schedule used rather than full training epochs."
            if preset in {"smoke", "quick"}
            else "Full epoch schedule (40 moons, 60 parity).",
        ),
        (
            "Compute device",
            device == "cuda",
            "Ran on CUDA; GPU reduction order can slightly shift float64 trajectories relative to CPU."
            if device == "cuda"
            else "CPU float64 statevector simulation.",
        ),
        (
            "Repository state",
            dirty or commit == "unavailable",
            f"Working tree had uncommitted changes or no git commit ({commit}, dirty={dirty})."
            if (dirty or commit == "unavailable")
            else f"Clean git tree at commit {commit[:10]}.",
        ),
        (
            "Parity additive ceiling",
            leak,
            (
                f"Entanglement-off mean accuracy ({flags['parity_off_mean_acc']:.4f}) exceeded "
                f"{flags['additive_leak_limit']:.2f}, flagging possible feature mixing (H3 marked invalid)."
            )
            if leak
            else (
                f"Entanglement-off mean accuracy ({_fmt_opt(flags.get('parity_off_mean_acc'))}) stayed within "
                f"the {flags['additive_leak_limit']:.2f} additive ceiling."
            ),
        ),
    ]


def build_report(run_dir: Path, cfg: ProtocolConfig | None = None) -> Path:
    run_dir = Path(run_dir)
    manifest, raw_records = load_run_folder(run_dir)
    proto_dict = dict(manifest["run_spec"]["protocol"])
    if "frame_epochs" in proto_dict:
        proto_dict["frame_epochs"] = tuple(proto_dict["frame_epochs"])
    if "negative_control_seeds" in proto_dict:
        proto_dict["negative_control_seeds"] = tuple(proto_dict["negative_control_seeds"])
    proto = ProtocolConfig(**proto_dict) if cfg is None else cfg
    preset = str(manifest["run_spec"]["preset"])
    n_seeds = int(manifest["run_spec"]["seeds"])

    summaries = summarize_variants(raw_records, proto)
    hypotheses, flags = evaluate_hypotheses(raw_records, preset=preset, cfg=proto)
    summary_sentences = generate_summary_sentences(summaries, hypotheses, flags, manifest)

    _write_tables(run_dir, summaries, hypotheses)
    fig_paths = generate_all_figures(run_dir, manifest, raw_records, summaries, hypotheses)

    run_spec = manifest["run_spec"]
    is_confirmatory = bool(run_spec["is_confirmatory"])
    mode_txt = "Confirmatory" if is_confirmatory else "Exploratory"
    if bool(run_spec["is_preliminary"]):
        mode_txt += f" (preliminary, N = {n_seeds})"

    hw = manifest.get("hardware", {})
    git_info = manifest.get("git", {})
    versions = manifest.get("versions", {})
    val_checks = manifest.get("validation_checks", [])

    if val_checks:
        val_rows_html = "\n".join(
            f"<tr><td>{int(c['check_id'])}</td><td>{html.escape(str(c['name']))}</td>"
            f"<td><span class='{'status-pass' if c['passed'] else 'status-fail'}'>"
            f"{'PASS' if c['passed'] else 'FAIL'}</span></td>"
            f"<td><code>{html.escape(str(c['detail']))}</code></td></tr>"
            for c in val_checks
        )
    else:
        val_rows_html = (
            "<tr><td colspan='4'>No <code>doctor</code> record found for this commit. "
            "Run <code>hybrid-qml doctor</code> before running experiments to include validation checks here.</td></tr>"
        )

    var_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(s.experiment)}</td>"
        f"<td>{html.escape(s.label)}</td>"
        f"<td class='num'>{s.n_seeds}</td>"
        f"<td class='num'>{s.test_acc_mean:.4f} &plusmn; {s.test_acc_sd:.4f}</td>"
        f"<td class='num'>{html.escape(_fmt_ci(s.test_acc_ci_low, s.test_acc_ci_high))}</td>"
        f"<td class='num'>{s.test_loss_mean:.4f}</td>"
        f"<td class='num'>{s.train_acc_mean:.4f}</td>"
        f"<td class='num'>{s.middle_params} / {s.total_params}</td>"
        f"<td class='num'>{html.escape(_fmt_opt(s.entropy_mean) if s.entropy_mean is None else f'{s.entropy_mean:.4f} ± {s.entropy_sd:.4f}')}</td>"
        "</tr>"
        for s in summaries
    )

    hyp_rows_html = "\n".join(
        "<tr>"
        f"<td><strong>{html.escape(h.hypothesis_id)}</strong></td>"
        f"<td>{html.escape(h.label_a)} - {html.escape(h.label_b)}</td>"
        f"<td class='num'>{h.n_seeds}</td>"
        f"<td>{html.escape(h.prediction)}</td>"
        f"<td class='num'>{h.mean_diff:+.4f}</td>"
        f"<td class='num'>{html.escape(_fmt_ci(h.ci_low, h.ci_high, signed=True))}</td>"
        f"<td class='num'>&plusmn;{h.margin:.2f}</td>"
        f"<td><strong>{html.escape(h.verdict)}</strong></td>"
        f"<td>{'-' if h.consistent is None else ('Yes' if h.consistent else 'No')}</td>"
        "</tr>"
        for h in hypotheses
    )

    threat_rows = _build_threat_flags(manifest, flags)
    threat_list_html = "\n".join(
        f"<li><strong>{html.escape(name)}:</strong> "
        f"<span class='{'flag-on' if active else 'flag-off'}'>{html.escape(msg)}</span></li>"
        for name, active, msg in threat_rows
    )

    summary_items_html = "\n".join(f"<p>{html.escape(s)}</p>" for s in summary_sentences)
    refs_html = "\n".join(f"<li>{html.escape(r)}</li>" for r in VERIFIED_REFERENCES)
    ver_items = ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))

    repro_cmd = (
        f"hybrid-qml run --preset {preset}"
        + (f" --seeds {n_seeds}" if n_seeds != run_spec["protocol"].get(f"{preset}_seeds", n_seeds) else "")
        + (f" --device {run_spec['device']}" if run_spec["device"] != "cpu" else "")
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(proto.title)}</title>
<style>
  body {{
    font-family: Georgia, "Times New Roman", serif;
    color: #1a1a1a;
    background: #ffffff;
    line-height: 1.58;
    max-width: 900px;
    margin: 36px auto;
    padding: 0 24px 56px 24px;
  }}
  h1, h2, .meta, table, code {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  h1 {{
    font-size: 1.5rem;
    line-height: 1.3;
    font-weight: 600;
    margin: 0 0 6px 0;
    color: #111111;
  }}
  h2 {{
    font-size: 1.08rem;
    font-weight: 600;
    margin-top: 2rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #d4d4d4;
    color: #111111;
  }}
  p {{ margin: 0.6rem 0; }}
  .run-line {{
    font-size: 0.9rem;
    color: #4a4a4a;
    margin-bottom: 16px;
  }}
  .identity-box {{
    background: #fafafa;
    border-top: 1px solid #cccccc;
    border-bottom: 1px solid #cccccc;
    padding: 10px 14px;
    margin: 14px 0 22px 0;
    font-size: 0.84rem;
    color: #333333;
  }}
  .identity-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 5px 16px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 18px 0;
    font-size: 0.83rem;
  }}
  th, td {{
    border-bottom: 1px solid #e0e0e0;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    font-weight: 600;
    border-top: 1.5px solid #222222;
    border-bottom: 1px solid #222222;
    background: #fcfcfc;
  }}
  tbody tr:last-child td {{
    border-bottom: 1.5px solid #222222;
  }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  figure {{
    margin: 20px 0 26px 0;
    padding: 8px 0;
    page-break-inside: avoid;
  }}
  figure img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
  figcaption {{ font-size: 0.85rem; color: #444444; margin-top: 8px; line-height: 1.45; }}
  .dual-media {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }}
  .status-pass {{ color: #0b6b4f; font-weight: 600; }}
  .status-fail {{ color: #b91c1c; font-weight: 600; }}
  .flag-on {{ color: #9a3412; font-weight: 600; }}
  .flag-off {{ color: #333333; }}
  code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 2px; font-size: 0.86em; }}
  @media print {{
    body {{ margin: 0; padding: 0; max-width: none; font-size: 10pt; }}
    figure, table {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <header>
    <h1>{html.escape(proto.title)}</h1>
    <div class="run-line meta">
      Run report &middot; {html.escape(mode_txt)} &middot; preset: <code>{html.escape(preset)}</code> &middot; N = {n_seeds} paired seeds
    </div>
  </header>

  <section class="identity-box meta">
    <div class="identity-grid">
      <div><strong>Mode:</strong> {html.escape(mode_txt)}</div>
      <div><strong>Device / dtype:</strong> {html.escape(str(run_spec['device']))} / float64</div>
      <div><strong>Epochs (moons / parity):</strong> {int(run_spec['moons_epochs'])} / {int(run_spec['parity_epochs'])}</div>
      <div><strong>Elapsed:</strong> {float(manifest.get('total_elapsed_seconds', 0.0)):.2f} s</div>
      <div><strong>OS:</strong> {html.escape(str(hw.get('os', 'unknown')))}</div>
      <div><strong>CPU:</strong> {html.escape(str(hw.get('cpu', 'unknown')))}</div>
      <div><strong>GPU:</strong> {html.escape(str(hw.get('gpu', 'none')))}</div>
      <div><strong>Git:</strong> {html.escape(str(git_info.get('commit', 'unavailable'))[:10])} (dirty={html.escape(str(git_info.get('dirty', 'unknown')))})</div>
      <div><strong>Started (UTC):</strong> {html.escape(str(manifest.get('started_utc', '')))}</div>
    </div>
    <div style="margin-top:6px;"><strong>Packages:</strong> <code>{html.escape(ver_items)}</code></div>
  </section>

  <h2>Summary</h2>
  {summary_items_html}

  <h2>Validation Checks</h2>
  <table>
    <thead>
      <tr><th>#</th><th>Check</th><th>Result</th><th>Measured Values</th></tr>
    </thead>
    <tbody>
      {val_rows_html}
    </tbody>
  </table>

  <h2>Results</h2>
  <p class="meta"><strong>Table 1.</strong> Variant summary over N = {n_seeds} paired seeds (evaluated at the final epoch). Accuracy intervals are 95% percentile bootstrap intervals (10,000 resamples) when N &ge; 2.</p>
  <table>
    <thead>
      <tr>
        <th>Task</th><th>Variant</th><th class="num">N</th>
        <th class="num">Test Acc (Mean &plusmn; SD)</th><th class="num">95% CI</th>
        <th class="num">Test Loss</th><th class="num">Train Acc</th>
        <th class="num">Params (Mid / Total)</th><th class="num">1-Qubit Entropy (bits)</th>
      </tr>
    </thead>
    <tbody>
      {var_rows_html}
    </tbody>
  </table>

  <figure>
    <img src="{_encode_data_uri(fig_paths['fig1_schematic'])}" alt="Quantum block circuit diagram">
    <figcaption><strong>Figure 1.</strong> Quantum middle block: per-layer Y-rotation data encoding, three-angle single-qubit rotations, optional CNOT entangling ring (identity when switched off), and Pauli-Z expectation readout across L = {int(proto.n_layers)} layers.</figcaption>
  </figure>

  <figure>
    <div class="dual-media">
      <div><img src="{_encode_data_uri(fig_paths['fig2_training_gif'])}" alt="Seed-0 moons training boundary animation"></div>
      <div><img src="{_encode_data_uri(fig_paths['fig2_training_final'])}" alt="Seed-0 moons final boundary"></div>
    </div>
    <figcaption><strong>Figure 2.</strong> Decision boundary progression on seed 0 of the entangled moons run (left: saved epochs; right: final epoch). Points show the standardized seed-0 training split (blue circles: class 0, orange triangles: class 1).</figcaption>
  </figure>

  <figure>
    <img src="{_encode_data_uri(fig_paths['fig3_moons_paired'])}" alt="Moons paired accuracy and differences">
    <figcaption><strong>Figure 3.</strong> Two-moons test accuracy across N = {n_seeds} paired seeds. (a) Per-seed accuracy for each variant with lines linking shared seeds; offset markers show the mean and 95% percentile bootstrap CI. (b) Paired differences for H1 and H2 with 95% bootstrap CIs and the &plusmn;{proto.equivalence_margin:.2f} equivalence band.</figcaption>
  </figure>

  <figure>
    <img src="{_encode_data_uri(fig_paths['fig4_moons_curves'])}" alt="Moons learning curves">
    <figcaption><strong>Figure 4.</strong> Two-moons test accuracy over training epochs (mean &plusmn; 1 SD across N = {n_seeds} seeds). (a) Entangled quantum block vs. classical MLP vs. no middle block. (b) Entangled vs. entanglement-off quantum blocks.</figcaption>
  </figure>

  <figure>
    <img src="{_encode_data_uri(fig_paths['fig5_moons_boundaries'])}" alt="Seed-0 moons decision boundaries">
    <figcaption><strong>Figure 5.</strong> Seed-0 final-epoch decision boundaries on two moons for all four model variants, with P(class = 1) shading and the 0.50 contour.</figcaption>
  </figure>

  <figure>
    <img src="{_encode_data_uri(fig_paths['fig6_parity_acc_curves'])}" alt="Parity accuracy and learning curves">
    <figcaption><strong>Figure 6.</strong> Parity/XOR results across N = {n_seeds} paired seeds. (a) Final-epoch test accuracy with paired seed lines, means and 95% bootstrap CIs, the 0.50 chance line, and the 0.75 additive ceiling. (b) Test accuracy trajectories over epochs (mean &plusmn; 1 SD across N = {n_seeds} seeds).</figcaption>
  </figure>

  <figure>
    <img src="{_encode_data_uri(fig_paths['fig7_parity_boundaries'])}" alt="Seed-0 parity decision boundaries">
    <figcaption><strong>Figure 7.</strong> Seed-0 final-epoch decision boundaries on parity/XOR for the entangled two-qubit model, the entanglement-off model, and the classical MLP reference.</figcaption>
  </figure>

  <h2>Hypothesis Comparisons</h2>
  <p class="meta"><strong>Table 2.</strong> Paired comparisons across N = {n_seeds} seeds (10,000 bootstrap resamples, equivalence margin &delta; = &plusmn;{proto.equivalence_margin:.2f}, no p-values).</p>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Comparison (A - B)</th><th class="num">N</th>
        <th>Advance Prediction</th><th class="num">Mean Diff</th>
        <th class="num">95% Bootstrap CI</th><th class="num">Margin</th>
        <th>Verdict</th><th>Matches Prediction</th>
      </tr>
    </thead>
    <tbody>
      {hyp_rows_html}
    </tbody>
  </table>

  <h2>Threats to Validity</h2>
  <p>All circuits here run as noiseless 64-bit statevector simulations on 2 to 4 simulated qubits over small two-dimensional synthetic datasets. Results apply only to this circuit family, parameter count, and training setup, and do not imply hardware resilience or any computational speed-up.</p>
  <ul>
    {threat_list_html}
  </ul>

  <h2>Reproducing This Run</h2>
  <p>To run the validation suite and regenerate this experiment folder from scratch:</p>
  <p><code>hybrid-qml doctor</code><br><code>{html.escape(repro_cmd)} --output &lt;run_dir&gt;</code></p>
  <p>To rebuild the tables, plots, and HTML report from existing raw logs in <code>&lt;run_dir&gt;/raw/</code> without retraining: <code>hybrid-qml report &lt;run_dir&gt;</code>.</p>

  <h2>References</h2>
  <ol>
    {refs_html}
  </ol>
</body>
</html>
"""

    report_path = run_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path
