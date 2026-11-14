#!/usr/bin/env python3
"""Validate mechanism outputs and create a checksummed output manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook


EXPECTED_SHEETS = {
    "README", "Quick_View", "Candidate_Summary", "L1_Mechanism_Scores", "Evidence_Components",
    "Evidence_Paths", "Direct_Edges", "Tissue_Contexts", "Coverage_QC", "Mechanism_Ontology",
    "Data_Dictionary", "Input_Manifest", "Method",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the complete mechanism-prediction output set.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))

    summaries = payload.get("candidate_summary", [])
    l1_rows = payload.get("l1_scores_long", [])
    components = payload.get("top_components_long", [])
    paths = payload.get("top_evidence_paths", [])
    ontology = payload.get("mechanism_ontology_l1", [])
    genes = [str(x.get("gene_id_v5", "")) for x in summaries]
    if len(summaries) != 9:
        errors.append(f"Expected 9 candidate summaries, observed {len(summaries)}")
    if len(set(genes)) != len(genes):
        errors.append("Candidate summary contains duplicate gene IDs")
    if len(l1_rows) != len(summaries) * 6:
        errors.append(f"Expected six L1 rows per candidate; observed {len(l1_rows)} total")
    if len(components) != len(summaries) * 5:
        errors.append(f"Expected five evidence-component rows per candidate; observed {len(components)} total")
    if len(paths) != len(summaries):
        errors.append(f"Expected one top path row per candidate; observed {len(paths)}")
    mechanism_ids = {str(x.get("mechanism_l1_id", "")) for x in ontology}
    if mechanism_ids != {"M1", "M2", "M3", "M4", "M5", "M6"}:
        errors.append(f"Unexpected L1 mechanism IDs: {sorted(mechanism_ids)}")
    if payload.get("schema", {}).get("nam_evidence_included") is not False:
        errors.append("Schema does not explicitly confirm exclusion of NAM evidence")

    status_counts = Counter(str(x.get("prediction_status", "")) for x in summaries)
    score_by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in l1_rows:
        score_by_gene[str(row.get("gene_id_v5", ""))].append(row)
    for summary in summaries:
        gene = str(summary.get("gene_id_v5", ""))
        rows = score_by_gene.get(gene, [])
        ranks = sorted(int(x["within_gene_l1_rank"]) for x in rows if x.get("within_gene_l1_rank") is not None)
        if ranks and ranks != [1, 2, 3, 4, 5, 6]:
            errors.append(f"{gene}: invalid within-gene ranks {ranks}")
        if summary.get("prediction_status") == "NETWORK_RANKED":
            score = finite(summary.get("primary_score_0_100"))
            if score is None or not 0 <= score <= 100:
                errors.append(f"{gene}: reportable primary score is missing or outside 0–100")

    component_by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in components:
        component_by_gene[str(row.get("gene_id_v5", ""))].append(row)
    for summary in summaries:
        gene = str(summary.get("gene_id_v5", ""))
        if summary.get("prediction_status") == "NETWORK_RANKED":
            total = sum(finite(x.get("weighted_contribution_to_primary_score")) or 0.0 for x in component_by_gene[gene])
            target = finite(summary.get("primary_score_0_100"))
            if target is not None and abs(total - target) > 0.08:
                errors.append(f"{gene}: weighted contributions ({total:.3f}) do not reproduce primary score ({target:.3f})")

    if not args.output_xlsx.exists():
        errors.append(f"Workbook missing: {args.output_xlsx}")
    else:
        workbook = load_workbook(args.output_xlsx, read_only=True, data_only=False)
        missing_sheets = sorted(EXPECTED_SHEETS - set(workbook.sheetnames))
        if missing_sheets:
            errors.append(f"Workbook missing sheets: {missing_sheets}")
        workbook.close()

    expected_figure_stems = [
        "Fig_mechanism_landscape",
        "Fig_selected_candidate_evidence_paths",
        "Fig_candidate_mechanism_profiles_overview",
    ]
    if not args.skip_figures:
        for stem in expected_figure_stems:
            for extension in ("png", "pdf", "svg", "tiff"):
                path = args.output_dir / f"{stem}.{extension}"
                if not path.exists() or path.stat().st_size == 0:
                    errors.append(f"Missing or empty figure: {path.name}")
        profile_dir = args.output_dir / "candidate_profiles"
        for gene in genes:
            if not (profile_dir / f"{gene}_mechanism_profile.png").exists():
                errors.append(f"Missing candidate profile PNG for {gene}")

    if status_counts.get("DESCRIPTIVE_ONLY", 0):
        warnings.append("Descriptive-only candidates must not be compared numerically with network-ranked candidates.")
    if status_counts.get("UNRESOLVED", 0):
        warnings.append("Unresolved candidates represent insufficient public-data coverage, not evidence against gene function.")

    files = sorted(path for path in args.output_dir.rglob("*") if path.is_file() and path.name not in {"output_manifest_sha256.tsv", "output_validation_report.json"})
    manifest_path = args.output_dir / "output_manifest_sha256.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in files:
            writer.writerow([path.relative_to(args.output_dir).as_posix(), path.stat().st_size, sha256(path)])

    report = {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "candidate_count": len(summaries),
        "status_counts": dict(sorted(status_counts.items())),
        "l1_score_rows": len(l1_rows),
        "component_rows": len(components),
        "top_path_rows": len(paths),
        "workbook_sheet_count_expected": len(EXPECTED_SHEETS),
        "manifest_file_count": len(files),
        "interpretation_rule": "Mechanism scores prioritize mechanisms within a candidate and are not cross-gene causal probabilities.",
        "nam_evidence_included": False,
    }
    report_path = args.output_dir / "output_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
