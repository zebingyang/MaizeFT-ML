#!/usr/bin/env python3
"""Run the complete candidate-gene mechanism-prediction workflow.

The launcher uses only the 11 prepared public-data input tables. It validates the
inputs, runs mechanism inference, standardizes results, exports figures/workbook,
and performs output QC. No NAM genetic evidence is read by this workflow.
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
SRC_DIR = PACKAGE_DIR / "src"
DEFAULT_CONFIG = PACKAGE_DIR / "config.example.json"


def load_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def absolute_from_package(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PACKAGE_DIR / path).resolve()


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in command)


def run_step(name: str, command: list[str], log_handle, environment: dict[str, str]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    heading = f"\n[{timestamp}] STEP {name}\n$ {command_text(command)}\n"
    print(heading, end="")
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
    code = process.wait()
    log_handle.flush()
    if code != 0:
        raise RuntimeError(f"Step {name} failed with exit code {code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, help="Override input_dir in the JSON configuration.")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir in the JSON configuration.")
    parser.add_argument("--full-network-check", action="store_true", help="Stream and validate every network edge before analysis.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation (useful for compute-only runs).")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory; known files may be replaced.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    input_dir = (args.input_dir.resolve() if args.input_dir else absolute_from_package(str(config["input_dir"])))
    output_dir = (args.output_dir.resolve() if args.output_dir else absolute_from_package(str(config["output_dir"])))
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}\nUse a new directory or pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    qc_dir = output_dir / "QC"
    raw_dir.mkdir(exist_ok=True)
    qc_dir.mkdir(exist_ok=True)
    raw_json = raw_dir / "mechanism_analysis_results.json"
    standardized_json = output_dir / "candidate9_mechanism_results.json"
    standardizer_native_json = output_dir / "candidate9_mechanism_results_v2.json"
    workbook = output_dir / "candidate9_mechanism_results.xlsx"
    log_path = output_dir / "pipeline_run.log"

    parameters = dict(config.get("parameters", {}))
    restart = str(parameters.get("rwr_restart", 0.5))
    tolerance = str(parameters.get("rwr_tolerance", 1e-11))
    max_iterations = str(parameters.get("rwr_max_iterations", 200))
    path_depth = str(parameters.get("path_max_depth", 4))
    formats = str(config.get("figure_formats", "png,pdf,svg,tiff"))
    dpi = str(config.get("figure_dpi", 400))
    selected_genes = ",".join(str(x) for x in config.get("selected_path_genes", []))

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUTF8"] = "1"
    python = sys.executable
    commands: list[tuple[str, list[str]]] = []
    validate_command = [python, str(SRC_DIR / "00_validate_inputs.py"), "--input-dir", str(input_dir), "--report", str(qc_dir / "input_validation_report.json")]
    if args.full_network_check or bool(config.get("full_network_check", False)):
        validate_command.append("--full-network-check")
    commands.append(("validate_inputs", validate_command))
    commands.append(
        (
            "mechanism_inference",
            [python, str(SRC_DIR / "01_run_mechanism_inference.py"), "--input-dir", str(input_dir),
             "--output-json", str(raw_json), "--rwr-restart", restart, "--rwr-tolerance", tolerance,
             "--rwr-max-iterations", max_iterations, "--path-max-depth", path_depth],
        )
    )
    commands.append(
        ("standardize_results", [python, str(SRC_DIR / "02_standardize_results.py"), "--input-json", str(raw_json), "--output-dir", str(output_dir)])
    )
    if not args.skip_figures:
        plot_command = [python, str(SRC_DIR / "03_plot_results.py"), "--input-json", str(standardizer_native_json),
                        "--output-dir", str(output_dir), "--formats", formats, "--dpi", dpi]
        if selected_genes:
            plot_command.extend(["--selected-genes", selected_genes])
        commands.append(("plot_results", plot_command))
    commands.append(
        ("export_excel", [python, str(SRC_DIR / "04_export_results_excel.py"), "--input-json", str(standardizer_native_json), "--output-xlsx", str(workbook)])
    )

    with log_path.open("a", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"MaizeGeneXAI mechanism pipeline\nStarted_UTC: {datetime.now(timezone.utc).isoformat()}\n")
        log_handle.write(f"Python: {sys.version}\nInput_dir: {input_dir}\nOutput_dir: {output_dir}\n")
        for name, command in commands:
            run_step(name, command, log_handle, environment)

        if standardizer_native_json != standardized_json:
            standardized_json.write_bytes(standardizer_native_json.read_bytes())
        validate_outputs = [python, str(SRC_DIR / "05_validate_outputs.py"), "--input-json", str(standardized_json),
                            "--output-xlsx", str(workbook), "--output-dir", str(output_dir)]
        if args.skip_figures:
            validate_outputs.append("--skip-figures")
        run_step("validate_outputs", validate_outputs, log_handle, environment)
        log_handle.write(f"Completed_UTC: {datetime.now(timezone.utc).isoformat()}\n")

    print(json.dumps({
        "status": "PASS",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "standardized_json": str(standardized_json),
        "workbook": str(workbook),
        "validation_report": str(qc_dir / "input_validation_report.json"),
        "pipeline_log": str(log_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
