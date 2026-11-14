#!/usr/bin/env python3
"""Validate the prepared inputs for public-data mechanism inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


REQUIRED = {
    "candidate9_mechanism_query.tsv": ["Gene_ID_v5", "Gene_ID_v4", "Trait", "Include"],
    "FT_mechanism_ontology.tsv": ["Mechanism_L1_ID", "Mechanism_L2_ID", "Mechanism_L1", "Mechanism_L2"],
    "FT_mechanism_seed_annotations_v5.tsv": ["Gene_ID_v5", "Mechanism_L1_ID", "Mechanism_L2_ID", "Seed_eligible_L1"],
    "maize_gene_ID_mapping_v5.tsv": ["Gene_ID_v5"],
    "maize_FT_multilayer_network_edges_v5.tsv": ["Gene1_ID_v5", "Gene2_ID_v5", "ChIA_PET_weight", "RNA_coexpression_weight", "Ribo_cotranslation_weight", "PPI_weight"],
    "maize_TF_target_regulatory_edges_v5.tsv": ["TF_ID_v5", "Target_ID_v5", "Direction", "Confidence_score"],
    "maize_chromatin_regulatory_edges_v5.tsv": ["Gene1_ID_v5", "Gene2_ID_v5"],
    "maize_expression_sample_metadata.tsv": ["Context_ID"],
    "maize_multilayer_network_nodes_v5.tsv": ["Gene_ID_v5"],
    "maize_gene_function_annotations_v5.tsv": ["Gene_ID_v5"],
    "candidate9_tissue_context_evidence.tsv": ["Gene_ID_v5", "Context_ID", "Candidate_tissue_evidence_QC"],
}

V5 = re.compile(r"^Zm00001eb\d{6}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--full-network-check", action="store_true", help="Stream all multilayer-network rows and validate weights/IDs.")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()

    errors: list[str] = []
    warnings: list[str] = []
    manifest = []
    for filename, required_columns in REQUIRED.items():
        path = input_dir / filename
        if not path.is_file():
            errors.append(f"Missing input: {filename}")
            continue
        observed = header(path)
        missing = [x for x in required_columns if x not in observed]
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
        manifest.append({"file": filename, "bytes": path.stat().st_size, "sha256": sha256(path), "columns": len(observed)})

    candidates: list[dict[str, str]] = []
    if not errors:
        candidates = [x for x in rows(input_dir / "candidate9_mechanism_query.tsv") if x.get("Include", "1") == "1"]
        candidate_ids = [x["Gene_ID_v5"] for x in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            errors.append("Candidate table contains duplicate Gene_ID_v5 values")
        invalid = [x for x in candidate_ids if not V5.fullmatch(x)]
        if invalid:
            errors.append(f"Invalid v5 candidate IDs: {invalid}")
        traits = {x.get("Trait", "") for x in candidates}
        if not traits.issubset({"DTA", "DTS", "ASI"}):
            errors.append(f"Unexpected trait values: {sorted(traits)}")

        ontology = rows(input_dir / "FT_mechanism_ontology.tsv")
        l1 = {x["Mechanism_L1_ID"] for x in ontology}
        l2 = {x["Mechanism_L2_ID"] for x in ontology}
        if l1 != {"M1", "M2", "M3", "M4", "M5", "M6"}:
            errors.append(f"Ontology must contain exactly M1-M6; observed {sorted(l1)}")
        if len(l2) < 6:
            errors.append("Ontology has too few L2 mechanism classes")

        seeds = rows(input_dir / "FT_mechanism_seed_annotations_v5.tsv")
        eligible = [x for x in seeds if x.get("Seed_eligible_L1") == "1"]
        eligible_ids = {x["Gene_ID_v5"] for x in eligible}
        overlap = sorted(set(candidate_ids) & eligible_ids)
        if overlap:
            errors.append(f"Candidate genes must not be included as mechanism seeds: {overlap}")
        missing_l1 = sorted(l1 - {x["Mechanism_L1_ID"] for x in eligible})
        if missing_l1:
            errors.append(f"L1 classes without eligible seeds: {missing_l1}")

        nodes = {x["Gene_ID_v5"] for x in rows(input_dir / "maize_multilayer_network_nodes_v5.tsv")}
        missing_nodes = sorted(set(candidate_ids) - nodes)
        if missing_nodes:
            errors.append(f"Candidates missing from node table: {missing_nodes}")

        tissue_genes = {x["Gene_ID_v5"] for x in rows(input_dir / "candidate9_tissue_context_evidence.tsv")}
        missing_tissue = sorted(set(candidate_ids) - tissue_genes)
        if missing_tissue:
            errors.append(f"Candidates missing from tissue-context evidence: {missing_tissue}")

        if args.full_network_check:
            edge_path = input_dir / "maize_FT_multilayer_network_edges_v5.tsv"
            edge_count = 0
            bad_ids = 0
            bad_weights = 0
            with edge_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    edge_count += 1
                    if not V5.fullmatch(row["Gene1_ID_v5"]) or not V5.fullmatch(row["Gene2_ID_v5"]):
                        bad_ids += 1
                    try:
                        weights = [float(row[x] or 0) for x in ["ChIA_PET_weight", "RNA_coexpression_weight", "Ribo_cotranslation_weight", "PPI_weight"]]
                    except ValueError:
                        bad_weights += 1
                        continue
                    if any(x < 0 for x in weights) or not any(x > 0 for x in weights):
                        bad_weights += 1
            if edge_count == 0:
                errors.append("Multilayer network contains no edges")
            if bad_ids:
                errors.append(f"Multilayer network contains {bad_ids} rows with invalid v5 IDs")
            if bad_weights:
                errors.append(f"Multilayer network contains {bad_weights} invalid weight rows")
        else:
            warnings.append("Full network row validation was skipped; use --full-network-check for archival QC")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "input_dir": str(input_dir),
        "candidate_count": len(candidates),
        "errors": errors,
        "warnings": warnings,
        "input_manifest": manifest,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
