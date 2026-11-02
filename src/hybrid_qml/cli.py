import argparse
from datetime import datetime, timezone
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from hybrid_qml import __version__
from hybrid_qml.protocol import (
    MOONS_VARIANTS,
    PARITY_VARIANTS,
    ProtocolConfig,
    RunSpec,
    resolve_run_spec,
)


def _git_metadata(cwd: Path) -> dict[str, Any]:
    git_exe = shutil.which("git")
    if git_exe is None:
        return {"commit": "unavailable", "dirty": True, "available": False}
    try:
        commit = (
            subprocess.check_output(
                [git_exe, "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                [git_exe, "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return {"commit": commit, "dirty": bool(status), "available": True}
    except Exception:
        return {"commit": "unavailable", "dirty": True, "available": False}


def _package_versions() -> dict[str, str]:
    pkgs = [
        "hybrid_qml",
        "torch",
        "pennylane",
        "pennylane-lightning",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pillow",
    ]
    out: dict[str, str] = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "not-installed"
    return out


def _hardware_info() -> dict[str, Any]:
    import torch

    gpu_desc = "none (CPU-only PyTorch build)"
    if torch.cuda.is_available():
        gpu_desc = torch.cuda.get_device_name(0)
    else:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            try:
                name = (
                    subprocess.check_output(
                        [nvsmi, "--query-gpu=name", "--format=csv,noheader"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode("utf-8")
                    .strip()
                    .splitlines()[0]
                )
                gpu_desc = f"{name} (driver present, PyTorch CPU build)"
            except Exception:
                pass

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu_desc,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _check_python_and_deps() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"Python >= 3.11 is required (found {platform.python_version()})."
        )
    for mod in ("torch", "pennylane", "numpy", "scipy", "sklearn", "matplotlib", "PIL"):
        try:
            __import__(mod)
        except ImportError as exc:
            raise RuntimeError(
                f"Missing dependency {mod!r}. Run 'pip install -e .' inside your environment."
            ) from exc


def _doctor_cache_path() -> Path:
    return Path("results") / ".doctor_status.json"


def _cmd_doctor() -> int:
    _check_python_and_deps()
    from hybrid_qml.validation import ALL_VALIDATION_CHECKS, run_validation_checks

    versions = _package_versions()
    hw = _hardware_info()
    git_info = _git_metadata(Path.cwd())

    print("hybrid-qml doctor")
    print(f"OS       : {hw['os']}")
    print(f"CPU      : {hw['cpu']} ({hw['cpu_count']} logical cores)")
    print(f"GPU      : {hw['gpu']}")
    print(f"Git      : {git_info['commit']} (dirty={git_info['dirty']})")
    print("Versions : " + ", ".join(f"{k}={v}" for k, v in versions.items()))
    print("Validation:")

    results = run_validation_checks(ALL_VALIDATION_CHECKS, ProtocolConfig())
    all_ok = all(r.passed for r in results)

    cache_path = _doctor_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info,
            "all_passed": all_ok,
            "checks": [
                {
                    "check_id": r.check_id,
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in results
            ],
        }
        _atomic_write_text(cache_path, json.dumps(payload, indent=2))
    except OSError:
        pass

    print(f"Result   : {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


def _load_matching_doctor_checks(git_info: dict[str, Any]) -> list[dict[str, Any]]:
    cache_path = _doctor_cache_path()
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("git", {}).get("commit") == git_info.get("commit"):
            return list(data.get("checks", []))
    except Exception:
        pass
    return []


def _ensure_writable_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Cannot write to output directory {out_dir!s}: {exc}") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    _check_python_and_deps()
    import torch
    from hybrid_qml.data import make_moons_dataset, make_parity_dataset
    from hybrid_qml.models import build_moons_model, build_parity_model
    from hybrid_qml.report import build_report
    from hybrid_qml.statistics import evaluate_hypotheses
    from hybrid_qml.training import train_and_evaluate

    spec: RunSpec = resolve_run_spec(
        preset=args.preset,
        seeds_override=args.seeds,
        device_override=args.device,
    )
    if spec.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use '--device cpu' or install a CUDA build of PyTorch."
        )

    out_dir = Path(args.output) if args.output else Path("results") / spec.preset
    _ensure_writable_dir(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    git_info = _git_metadata(Path.cwd())
    versions = _package_versions()
    hw = _hardware_info()

    manifest_path = out_dir / "manifest.json"
    spec_json_obj = json.loads(json.dumps(spec.to_dict()))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_spec") != spec_json_obj:
            raise RuntimeError(
                f"Existing manifest in {out_dir} was created with different settings. "
                "Pick a new --output directory or use matching flags to resume."
            )
        manifest = existing
    else:
        manifest = {
            "hybrid_qml_version": __version__,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "completed_utc": None,
            "total_elapsed_seconds": 0.0,
            "git": git_info,
            "versions": versions,
            "hardware": hw,
            "run_spec": spec.to_dict(),
            "validation_checks": _load_matching_doctor_checks(git_info),
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

    mode_label = "confirmatory" if spec.is_confirmatory else "exploratory"
    print(
        f"hybrid-qml {__version__} | commit={git_info['commit'][:10]} (dirty={git_info['dirty']}) | "
        f"preset={spec.preset} ({mode_label}) | device={spec.device} | seeds={spec.seeds}"
    )

    run_t0 = time.perf_counter()
    raw_records: list[dict[str, Any]] = []

    tasks = [
        ("e2_moons", MOONS_VARIANTS, make_moons_dataset, build_moons_model, spec.moons_epochs),
        ("e3_parity", PARITY_VARIANTS, make_parity_dataset, build_parity_model, spec.parity_epochs),
    ]

    for exp_name, variants, make_ds, build_m, epochs in tasks:
        for seed in range(spec.seeds):
            ds = None
            if seed == 0:
                seed0_data_path = raw_dir / f"{exp_name}__seed000_data.npz"
                if not seed0_data_path.exists():
                    ds = make_ds(seed, spec.protocol)
                    _atomic_write_npz(
                        seed0_data_path,
                        X_train=ds.X_train,
                        y_train=ds.y_train,
                    )

            for variant in variants:
                unit_prefix = f"{exp_name}__{variant}__seed{seed:03d}"
                json_path = raw_dir / f"{unit_prefix}.json"
                frames_path = raw_dir / f"{unit_prefix}_frames.npz"
                already_done = json_path.exists() and (seed != 0 or frames_path.exists())

                if already_done:
                    rec = json.loads(json_path.read_text(encoding="utf-8"))
                    raw_records.append(rec)
                    te_acc = float(rec["final_metrics"]["final_test_acc"])
                    el = float(rec["elapsed_seconds"])
                    print(f"[skip] {exp_name:9s}  {variant:22s}  seed={seed:03d}  test_acc={te_acc:.4f}  ({el:.2f}s)")
                    continue

                if ds is None:
                    ds = make_ds(seed, spec.protocol)

                model = build_m(variant, seed, spec.protocol, spec.device)
                res = train_and_evaluate(
                    model=model,
                    dataset=ds,
                    experiment=exp_name,
                    variant=variant,
                    seed=seed,
                    epochs=epochs,
                    grid_resolution=spec.grid_resolution,
                    cfg=spec.protocol,
                    device=spec.device,
                    save_frames=(seed == 0),
                )

                rec = {
                    "experiment": res.experiment,
                    "variant": res.variant,
                    "seed": res.seed,
                    "epochs": res.epochs,
                    "history": res.history,
                    "final_metrics": res.final_metrics,
                    "param_counts": res.param_counts,
                    "elapsed_seconds": res.elapsed_seconds,
                }
                _atomic_write_text(json_path, json.dumps(rec, indent=2))
                if seed == 0 and res.grid_x is not None and res.grid_y is not None:
                    frame_dict: dict[str, Any] = {
                        "grid_x": res.grid_x,
                        "grid_y": res.grid_y,
                    }
                    for ep_k, arr in res.grid_frames.items():
                        frame_dict[f"frame_{ep_k}"] = arr
                    _atomic_write_npz(frames_path, **frame_dict)

                raw_records.append(rec)
                te_acc = float(res.final_metrics["final_test_acc"])
                print(
                    f"[done] {exp_name:9s}  {variant:22s}  seed={seed:03d}  "
                    f"test_acc={te_acc:.4f}  ({res.elapsed_seconds:.2f}s)"
                )

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_elapsed_seconds"] = float(manifest.get("total_elapsed_seconds", 0.0)) + (
        time.perf_counter() - run_t0
    )
    if not manifest.get("validation_checks"):
        manifest["validation_checks"] = _load_matching_doctor_checks(git_info)
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

    report_path = build_report(out_dir, spec.protocol)
    hyps, _flags = evaluate_hypotheses(raw_records, preset=spec.preset, cfg=spec.protocol)
    print("Verdicts:")
    for h in hyps:
        ci_txt = (
            "n/a"
            if h.ci_low is None or h.ci_high is None
            else f"[{h.ci_low:+.4f}, {h.ci_high:+.4f}]"
        )
        print(
            f"  {h.hypothesis_id}: diff={h.mean_diff:+.4f}, 95% CI={ci_txt} -> {h.verdict}"
        )
    print(f"Report: {report_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    _check_python_and_deps()
    from hybrid_qml.report import build_report

    run_dir = Path(args.run_folder)
    report_path = build_report(run_dir)
    print(f"Report written to {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid-qml",
        description="Entanglement-ablation experiments on simulated qubits.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check environment and run validation checks.")

    run_p = sub.add_parser("run", help="Run experiments and build the report.")
    run_p.add_argument(
        "--preset",
        choices=("smoke", "quick", "standard"),
        default="standard",
        help="Preset configuration (default: standard).",
    )
    run_p.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of paired seeds (marks run exploratory).",
    )
    run_p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Torch device (default: cpu; cuda marks run exploratory).",
    )
    run_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder (default: results/<preset>).",
    )

    rep_p = sub.add_parser("report", help="Rebuild tables, plots, and HTML report from raw logs.")
    rep_p.add_argument("run_folder", type=str, help="Existing run folder.")

    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _cmd_doctor()
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        return 1
    except MemoryError:
        print("Error: out of memory.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
