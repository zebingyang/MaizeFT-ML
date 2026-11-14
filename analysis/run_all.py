#!/usr/bin/env python3
"""Run the complete maize candidate-gene function and mechanism workflow.

The two modules remain scientifically separate:
1. function prediction compares six supervised classifiers and predicts
   similarity to curated flowering-time genes;
2. mechanism prediction prioritizes candidate-specific mechanism hypotheses
   using public multilayer networks, expression context, directed TF evidence,
   and annotation.

No NAM genetic evidence is consumed by either prediction module.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.example.json"


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in command)


def run_step(name: str, command: list[str], log_handle, environment: dict[str, str]) -> None:
    heading = f"\n[{datetime.now(timezone.utc).isoformat()}] STEP {name}\n$ {command_text(command)}\n"
    print(heading, end="", flush=True)
    log_handle.write(heading)
    log_handle.flush()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
    return_code = process.wait()
    log_handle.flush()
    if return_code != 0:
        raise RuntimeError(f"Step {name} failed with exit code {return_code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--module", choices=["all", "function", "mechanism"], default="all")
    parser.add_argument("--validate-only", action="store_true", help="Validate direct inputs and dependencies without fitting or propagation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow known outputs to be replaced in non-empty module result directories.")
    parser.add_argument("--skip-full-network-check", action="store_true", help="Skip streaming every mechanism-network edge during input QC.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = resolve_config_path(str(config["data_root"]), config_path)
    results_root = resolve_config_path(str(config["results_root"]), config_path)
    function_config = dict(config.get("function_prediction", {}))
    mechanism_config = dict(config.get("mechanism_prediction", {}))

    run_function = args.module in {"all", "function"} and bool(function_config.get("enabled", True))
    run_mechanism = args.module in {"all", "mechanism"} and bool(mechanism_config.get("enabled", True))
    if not run_function and not run_mechanism:
        raise SystemExit("No module is enabled for this run")

    results_root.mkdir(parents=True, exist_ok=True)
    qc_dir = results_root / "QC"
    qc_dir.mkdir(exist_ok=True)
    log_path = qc_dir / ("combined_validation.log" if args.validate_only else "combined_pipeline.log")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["MPLBACKEND"] = "Agg"
    python = sys.executable
    commands: list[tuple[str, list[str]]] = []

    if run_function:
        training_file = str(function_config.get("training_file", "FT_training_feature_matrix.csv"))
        candidate_file = str(function_config.get("candidate_file", "candidate9_feature_matrix.csv"))
        function_output = results_root / str(function_config.get("output_subdir", "function_prediction"))
        command = [
            python,
            str(PACKAGE_DIR / "function_prediction" / "FT_nested_six_model_training.py"),
            "--input-dir", str(data_root),
            "--training-file", training_file,
            "--candidate-file", candidate_file,
        ]
        if args.validate_only:
            command.append("--validate-only")
        else:
            command.extend(
                [
                    "--output-dir", str(function_output),
                    "--n-jobs", str(function_config.get("n_jobs", 1)),
                    "--bootstrap-repeats", str(function_config.get("bootstrap_repeats", 1000)),
                    "--permutation-repeats", str(function_config.get("permutation_repeats", 50)),
                ]
            )
            if args.overwrite:
                command.append("--allow-nonempty-output")
        commands.append(("function_prediction_input_validation" if args.validate_only else "function_prediction_six_models", command))

    if run_mechanism:
        mechanism_input = resolve_config_path(str(mechanism_config.get("input_dir", "../Mechanism_input")), config_path)
        mechanism_output = results_root / str(mechanism_config.get("output_subdir", "mechanism_prediction"))
        full_check = bool(mechanism_config.get("full_network_check", True)) and not args.skip_full_network_check
        if args.validate_only:
            command = [
                python,
                str(PACKAGE_DIR / "mechanism_prediction" / "src" / "00_validate_inputs.py"),
                "--input-dir", str(mechanism_input),
                "--report", str(qc_dir / "mechanism_input_validation_report.json"),
            ]
            if full_check:
                command.append("--full-network-check")
            commands.append(("mechanism_prediction_input_validation", command))
        else:
            command = [
                python,
                str(PACKAGE_DIR / "mechanism_prediction" / "run_pipeline.py"),
                "--input-dir", str(mechanism_input),
                "--output-dir", str(mechanism_output),
            ]
            if full_check:
                command.append("--full-network-check")
            if args.overwrite:
                command.append("--overwrite")
            commands.append(("mechanism_prediction_public_network_inference", command))

    with log_path.open("a", encoding="utf-8", newline="") as log_handle:
        log_handle.write("Maize FT function + mechanism prediction workflow\n")
        log_handle.write(f"Started_UTC: {datetime.now(timezone.utc).isoformat()}\n")
        log_handle.write(f"Python: {sys.version}\nData_root: {data_root}\nResults_root: {results_root}\n")
        for name, command in commands:
            run_step(name, command, log_handle, environment)
        log_handle.write(f"Completed_UTC: {datetime.now(timezone.utc).isoformat()}\n")

    report = {
        "status": "PASS",
        "mode": "validation_only" if args.validate_only else "full_analysis",
        "modules": [name for name, enabled in (("function", run_function), ("mechanism", run_mechanism)) if enabled],
        "data_root": str(data_root),
        "results_root": str(results_root),
        "log": str(log_path),
        "interpretation": {
            "function_probability": "Similarity to curated flowering-time training genes; not a causal probability.",
            "mechanism_score": "Within-candidate public-data mechanism support; not a cross-gene causal ranking.",
            "nam_evidence_used": False,
        },
    }
    report_path = qc_dir / ("combined_validation_report.json" if args.validate_only else "combined_run_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
