from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "ML_prediction_results_reproduced" / "mechanism_prediction"
DEFAULT_INPUT_JSON = DEFAULT_OUTPUT_DIR / "raw" / "mechanism_analysis_results.json"
INPUT_JSON = DEFAULT_INPUT_JSON
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PROFILE_DIR = OUTPUT_DIR / "candidate_profiles"
OUTPUT_JSON = OUTPUT_DIR / "candidate9_mechanism_results_v2.json"

SCHEMA_VERSION = "MaizeGeneXAI_mechanism_output_schema_v2.0"
DISPLAY_VERSION = "Maize_candidate_mechanism_display_v2.0_2026-08-27"

MECHANISM_EN = {
    "M1": "Environmental and light signaling",
    "M2": "Florigen and floral transition",
    "M3": "Age and autonomous pathways",
    "M4": "Hormone and stress response",
    "M5": "Meristem and reproductive development",
    "M6": "Chromatin and RNA regulation",
}
MECHANISM_SHORT = {
    "M1": "Environment /\nlight",
    "M2": "Florigen /\ntransition",
    "M3": "Age /\nautonomous",
    "M4": "Hormone /\nstress",
    "M5": "Meristem /\nreproduction",
    "M6": "Chromatin /\nRNA",
}
TRAIT_COLORS = {"DTA": "#2F6B9A", "DTS": "#D87520", "ASI": "#7A4E9D"}
MECHANISM_COLORS = {
    "M1": "#4477AA",
    "M2": "#66CCEE",
    "M3": "#228833",
    "M4": "#CCBB44",
    "M5": "#EE6677",
    "M6": "#AA3377",
}
LAYER_COLORS = {"CHIA": "#7A5195", "RNA": "#2F6B9A", "RIBO": "#E07A2D", "PPI": "#3B8F5A"}
COMPONENT_WEIGHTS = {"network": 0.60, "tf_directed": 0.15, "tissue": 0.15, "annotation": 0.05, "trait": 0.05}
COMPONENT_LABELS = {
    "network": "Network",
    "tf_directed": "TF-directed",
    "tissue": "Tissue context",
    "annotation": "Annotation",
    "trait": "Trait match",
}


def finite(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def r2(value: object) -> float | None:
    x = finite(value)
    return None if x is None else round(x, 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def specificity_flag(status: str, margin: float | None) -> str:
    if status == "UNRESOLVED_INSUFFICIENT_PUBLIC_DATA":
        return "UNRESOLVED"
    if margin is None:
        return "NOT_ASSESSABLE"
    if abs(margin) < 1e-9:
        return "TIED_TOP_TWO"
    if margin < 5:
        return "LOW_TOP_TWO_SEPARATION"
    return "TOP_MECHANISM_SEPARATED"


def status_short(status: str) -> str:
    return {
        "RANKED_L1_NETWORK_PROPAGATION": "NETWORK_RANKED",
        "DESCRIPTIVE_ONLY_NO_MULTILAYER_NETWORK": "DESCRIPTIVE_ONLY",
        "UNRESOLVED_INSUFFICIENT_PUBLIC_DATA": "UNRESOLVED",
    }.get(status, status)


def tier_short(tier: str) -> str:
    return tier.replace("EVIDENCE_TIER_", "Tier ").replace("DESCRIPTIVE_ONLY", "Descriptive").replace("UNRESOLVED", "Unresolved")


def parse_contexts(details: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, token in enumerate(str(details or "").split(";"), start=1):
        if not token:
            continue
        parts = token.split("|")
        if len(parts) < 4:
            continue
        rows.append(
            {
                "context_rank": rank,
                "context_id": parts[0],
                "biological_tissue": parts[1],
                "developmental_stage": parts[2],
                "effective_expression_percentile": r2((finite(parts[3]) or 0.0) * 100.0),
            }
        )
    return rows


def build_standardized_payload(raw: dict[str, object]) -> dict[str, object]:
    summaries = raw["summary"]
    l1_rows = raw["l1_scores"]
    paths = raw["paths"]
    summary_by_gene = {x["Gene_ID_v5"]: x for x in summaries}
    l1_lookup = {(x["Gene_ID_v5"], x["Mechanism_L1_ID"]): x for x in l1_rows}
    path_lookup = {(x["Candidate_Gene_ID_v5"], x["Mechanism_L1_ID"]): x for x in paths}

    trait_order = {"DTA": 0, "DTS": 1, "ASI": 2}
    display_gene_order = sorted(
        [x["Gene_ID_v5"] for x in summaries],
        key=lambda g: (trait_order.get(summary_by_gene[g]["Trait"], 9), g),
    )
    display_order = {g: i + 1 for i, g in enumerate(display_gene_order)}

    standardized_summary: list[dict[str, object]] = []
    standardized_l1: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    top_paths: list[dict[str, object]] = []
    tissue_contexts: list[dict[str, object]] = []

    for s in summaries:
        gene = s["Gene_ID_v5"]
        status = s["Prediction_status"]
        top_id = s["Top_Mechanism_L1_ID"] or ""
        top = l1_lookup.get((gene, top_id))
        margin = finite(s["Top_vs_runner_up_margin"])
        if status == "RANKED_L1_NETWORK_PROPAGATION":
            primary_score = finite(s["Top_integrated_score_0_100"])
            score_type = "INTEGRATED_NETWORK_AUGMENTED"
        elif status == "DESCRIPTIVE_ONLY_NO_MULTILAYER_NETWORK":
            primary_score = finite(s["Top_descriptive_score_noncomparable_0_100"])
            score_type = "DESCRIPTIVE_NONCOMPARABLE"
        else:
            primary_score = None
            score_type = "NOT_REPORTABLE"

        raw_components = {
            "network": finite(top.get("Network_score_equal_layer_0_100")) if top else None,
            "tf_directed": finite(top.get("TF_direct_score_0_100")) if top else None,
            "tissue": finite(top.get("Tissue_context_score_0_100")) if top else None,
            "annotation": finite(top.get("Annotation_support_score_0_100")) if top else None,
            "trait": finite(top.get("Trait_compatibility_score_0_100")) if top else None,
        }
        if status != "RANKED_L1_NETWORK_PROPAGATION":
            raw_components["network"] = None
        if status == "UNRESOLVED_INSUFFICIENT_PUBLIC_DATA":
            raw_components = {k: None for k in raw_components}
        denom = sum(COMPONENT_WEIGHTS[k] for k, v in raw_components.items() if v is not None)
        available_names = [k for k, v in raw_components.items() if v is not None]
        for component, value in raw_components.items():
            available = value is not None
            effective_weight = COMPONENT_WEIGHTS[component] / denom if available and denom else None
            components.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "display_order": display_order[gene],
                    "gene_id_v5": gene,
                    "trait": s["Trait"],
                    "prediction_status": status_short(status),
                    "top_mechanism_l1_id": top_id,
                    "top_mechanism_l1_en": MECHANISM_EN.get(top_id, ""),
                    "evidence_component": component,
                    "evidence_component_label": COMPONENT_LABELS[component],
                    "component_available": int(available),
                    "component_score_0_100": r2(value),
                    "declared_weight": COMPONENT_WEIGHTS[component],
                    "effective_weight_after_missing_renormalization": round(effective_weight, 6) if effective_weight is not None else None,
                    "weighted_contribution_to_primary_score": r2(value * effective_weight) if effective_weight is not None else None,
                    "score_type": score_type,
                    "interpretation_limit": "Component values support the top mechanism within this candidate; they are not causal probabilities.",
                }
            )

        layer_fields = {
            "CHIA": "CHIA_degree_adjusted_RWR_percentile",
            "RNA": "RNA_degree_adjusted_RWR_percentile",
            "RIBO": "RIBO_degree_adjusted_RWR_percentile",
            "PPI": "PPI_degree_adjusted_RWR_percentile",
        }
        assessable_layers = [layer for layer, field in layer_fields.items() if top and finite(top.get(field)) is not None]
        top_path = path_lookup.get((gene, top_id), {}) if top_id else {}
        top_path_record = {
            "schema_version": SCHEMA_VERSION,
            "display_order": display_order[gene],
            "gene_id_v5": gene,
            "trait": s["Trait"],
            "top_mechanism_l1_id": top_id,
            "top_mechanism_l1_en": MECHANISM_EN.get(top_id, ""),
            "path_available": int(finite(top_path.get("Shortest_path_distance_edges")) is not None),
            "shortest_path_distance_edges": finite(top_path.get("Shortest_path_distance_edges")),
            "path_node_sequence": top_path.get("Path_node_sequence", ""),
            "path_edge_layer_sequence": top_path.get("Path_edge_layer_sequence", ""),
            "terminal_seed_gene_id_v5": top_path.get("Terminal_seed_Gene_ID_v5", ""),
            "terminal_seed_symbol": top_path.get("Terminal_seed_symbol", ""),
            "path_directionality": "UNDIRECTED_ASSOCIATION_PATH",
            "interpretation_limit": "Shortest public-network association path for explanation only; it is not a regulatory cascade or proof of causality.",
        }
        top_paths.append(top_path_record)

        contexts = parse_contexts(top.get("Top_tissue_context_details", "") if top else "")
        for ctx in contexts:
            tissue_contexts.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "display_order": display_order[gene],
                    "gene_id_v5": gene,
                    "trait": s["Trait"],
                    "top_mechanism_l1_id": top_id,
                    "top_mechanism_l1_en": MECHANISM_EN.get(top_id, ""),
                    **ctx,
                    "interpretation_limit": "Expression context is supporting background, not evidence that expression changes flowering time.",
                }
            )

        spec = specificity_flag(status, margin)
        sensitivity = finite(s["Network_layer_weight_sensitivity_top_stability"])
        summary_row = {
            "schema_version": SCHEMA_VERSION,
            "display_order": display_order[gene],
            "gene_id_v5": gene,
            "gene_id_v4": s["Gene_ID_v4"],
            "preferred_gene_symbol": s["Preferred_gene_symbol"],
            "trait": s["Trait"],
            "prediction_status": status_short(status),
            "evidence_tier": s["Evidence_tier"],
            "top_mechanism_specificity": spec,
            "top_mechanism_l1_id": top_id,
            "top_mechanism_l1_en": MECHANISM_EN.get(top_id, ""),
            "top_mechanism_l1_cn": s["Top_Mechanism_L1_CN"],
            "primary_score_0_100": r2(primary_score),
            "primary_score_type": score_type,
            "recommended_comparison_scope": "WITHIN_GENE_ONLY",
            "runner_up_mechanism_l1_id": s["Runner_up_Mechanism_L1_ID"],
            "runner_up_mechanism_l1_en": MECHANISM_EN.get(s["Runner_up_Mechanism_L1_ID"], ""),
            "top_vs_runner_up_margin": r2(margin),
            "sensitivity_top_mechanism_stability": round(sensitivity, 3) if sensitivity is not None else None,
            "network_layers_assessable_n": int(s["Network_layers_assessable"] or 0),
            "network_layers_assessable": ";".join(assessable_layers),
            "available_evidence_components_n": len(available_names),
            "available_evidence_components": ";".join(available_names),
            "network_score_0_100": r2(raw_components["network"]),
            "tf_directed_score_0_100": r2(raw_components["tf_directed"]),
            "tissue_context_score_0_100": r2(raw_components["tissue"]),
            "annotation_score_0_100": r2(raw_components["annotation"]),
            "trait_match_score_0_100": r2(raw_components["trait"]),
            "direct_tf_seed_edges_n": int(top.get("N_direct_TF_seed_edges", 0)) if top else 0,
            "direct_multilayer_seed_edges_n": int(top.get("N_direct_multilayer_seed_edges", 0)) if top else 0,
            "direct_chromatin_seed_edges_n": int(top.get("N_direct_chromatin_seed_edges", 0)) if top else 0,
            "top_l2_descriptive_id": s["Top_L2_descriptive_ID"],
            "top_l2_descriptive_en": next((x["Mechanism_L2"] for x in raw["l2_support"] if x["Gene_ID_v5"] == gene and x["Mechanism_L2_ID"] == s["Top_L2_descriptive_ID"]), ""),
            "shortest_path_distance_edges": top_path_record["shortest_path_distance_edges"],
            "terminal_seed_gene_id_v5": top_path_record["terminal_seed_gene_id_v5"],
            "terminal_seed_symbol": top_path_record["terminal_seed_symbol"],
            "top_tissue_context_ids": ";".join(x["context_id"] for x in contexts),
            "mechanism_hypothesis_cn": s["Mechanism_hypothesis_CN"],
            "interpretation_limit": "Public-data mechanism hypothesis only; no NAM genetic evidence enters this module, and the result is not experimental validation.",
        }
        standardized_summary.append(summary_row)

        coverage.append(
            {
                "schema_version": SCHEMA_VERSION,
                "display_order": display_order[gene],
                "gene_id_v5": gene,
                "trait": s["Trait"],
                "prediction_status": status_short(status),
                "network_layer_coverage_n_of_4": int(s["Network_layers_assessable"] or 0),
                "network_layer_coverage_fraction": round(int(s["Network_layers_assessable"] or 0) / 4, 3),
                "network_layers_assessable": ";".join(assessable_layers),
                "tf_evidence_assessable": int(top is not None and finite(top.get("TF_direct_score_0_100")) is not None),
                "tissue_evidence_assessable": int(top is not None and finite(top.get("Tissue_context_score_0_100")) is not None),
                "annotation_evidence_assessable": int(top is not None and finite(top.get("Annotation_support_score_0_100")) is not None),
                "trait_match_assessable": int(top is not None and finite(top.get("Trait_compatibility_score_0_100")) is not None),
                "shortest_path_available": top_path_record["path_available"],
                "top_mechanism_sensitivity_stability": round(sensitivity, 3) if sensitivity is not None else None,
                "top_vs_runner_up_margin": r2(margin),
                "top_mechanism_specificity": spec,
                "network_qc": top.get("Network_QC", "") if top else "NOT_ASSESSABLE",
                "tissue_qc": top.get("Tissue_QC", "") if top else "NOT_ASSESSABLE",
                "function_annotation_qc": top.get("Function_annotation_QC", "") if top else "NOT_ASSESSABLE",
            }
        )

    top_score_by_gene = {x["gene_id_v5"]: finite(x["primary_score_0_100"]) for x in standardized_summary}
    for row in l1_rows:
        gene = row["Gene_ID_v5"]
        status = row["Prediction_status"]
        if status == "RANKED_L1_NETWORK_PROPAGATION":
            score = finite(row["Integrated_mechanism_support_score_0_100"])
            score_type = "INTEGRATED_NETWORK_AUGMENTED"
        elif status == "DESCRIPTIVE_ONLY_NO_MULTILAYER_NETWORK":
            score = finite(row["Descriptive_support_score_noncomparable_0_100"])
            score_type = "DESCRIPTIVE_NONCOMPARABLE"
        else:
            score = None
            score_type = "NOT_REPORTABLE"
        top_score = top_score_by_gene[gene]
        standardized_l1.append(
            {
                "schema_version": SCHEMA_VERSION,
                "display_order": display_order[gene],
                "gene_id_v5": gene,
                "gene_id_v4": row["Gene_ID_v4"],
                "preferred_gene_symbol": row["Preferred_gene_symbol"],
                "trait": row["Trait"],
                "prediction_status": status_short(status),
                "mechanism_l1_id": row["Mechanism_L1_ID"],
                "mechanism_l1_en": MECHANISM_EN[row["Mechanism_L1_ID"]],
                "mechanism_l1_cn": row["Mechanism_L1_CN"],
                "within_gene_l1_rank": int(row["Within_gene_L1_rank"]),
                "primary_display_score_0_100": r2(score),
                "primary_score_type": score_type,
                "delta_from_top_score": r2(score - top_score) if score is not None and top_score is not None else None,
                "recommended_comparison_scope": "WITHIN_GENE_ONLY",
                "network_score_equal_layer_0_100": r2(row["Network_score_equal_layer_0_100"]),
                "network_sensitivity_min_0_100": r2(row["Network_score_sensitivity_min_0_100"]),
                "network_sensitivity_max_0_100": r2(row["Network_score_sensitivity_max_0_100"]),
                "network_sensitivity_range": r2(row["Network_score_sensitivity_range"]),
                "network_layers_assessable_n": int(row["Network_layers_assessable"] or 0),
                "chia_degree_adjusted_rwr_percentile": r2((finite(row["CHIA_degree_adjusted_RWR_percentile"]) or 0) * 100) if finite(row["CHIA_degree_adjusted_RWR_percentile"]) is not None else None,
                "rna_degree_adjusted_rwr_percentile": r2((finite(row["RNA_degree_adjusted_RWR_percentile"]) or 0) * 100) if finite(row["RNA_degree_adjusted_RWR_percentile"]) is not None else None,
                "ribo_degree_adjusted_rwr_percentile": r2((finite(row["RIBO_degree_adjusted_RWR_percentile"]) or 0) * 100) if finite(row["RIBO_degree_adjusted_RWR_percentile"]) is not None else None,
                "ppi_degree_adjusted_rwr_percentile": r2((finite(row["PPI_degree_adjusted_RWR_percentile"]) or 0) * 100) if finite(row["PPI_degree_adjusted_RWR_percentile"]) is not None else None,
                "tissue_context_score_0_100": r2(row["Tissue_context_score_0_100"]),
                "tf_directed_score_0_100": r2(row["TF_direct_score_0_100"]),
                "annotation_score_0_100": r2(row["Annotation_support_score_0_100"]),
                "trait_match_score_0_100": r2(row["Trait_compatibility_score_0_100"]),
                "direct_tf_seed_edges_n": int(row["N_direct_TF_seed_edges"] or 0),
                "direct_multilayer_seed_edges_n": int(row["N_direct_multilayer_seed_edges"] or 0),
                "direct_chromatin_seed_edges_n": int(row["N_direct_chromatin_seed_edges"] or 0),
                "eligible_seed_genes_n": int(row["Eligible_seed_genes"] or 0),
                "top_tissue_context_ids": row["Top_tissue_contexts"],
                "direct_tf_seed_genes": row["Direct_TF_seed_genes"],
                "annotation_matched_terms": row["Annotation_matched_terms"],
                "network_qc": row["Network_QC"],
                "tissue_qc": row["Tissue_QC"],
                "function_annotation_qc": row["Function_annotation_QC"],
                "interpretation_limit": "Use scores to compare mechanisms within the same candidate. Cross-gene ranking is not recommended because evidence coverage differs.",
            }
        )

    enriched_edges = []
    top_by_gene = {x["gene_id_v5"]: x["top_mechanism_l1_id"] for x in standardized_summary}
    for edge in raw["direct_edges"]:
        gene = edge["Candidate_Gene_ID_v5"]
        enriched_edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "display_order": display_order[gene],
                "gene_id_v5": gene,
                "trait": summary_by_gene[gene]["Trait"],
                "is_top_mechanism_edge": int(edge["Mechanism_L1_ID"] == top_by_gene[gene]),
                "evidence_source": edge["Evidence_source"],
                "seed_gene_id_v5": edge["Seed_Gene_ID_v5"],
                "seed_symbol": edge["Seed_symbol"],
                "mechanism_l1_id": edge["Mechanism_L1_ID"],
                "mechanism_l1_en": MECHANISM_EN.get(edge["Mechanism_L1_ID"], ""),
                "mechanism_l2_ids": edge["Mechanism_L2_IDs"],
                "direction": edge["Direction"],
                "regulatory_sign": edge["Regulatory_sign"],
                "confidence_score": finite(edge.get("Confidence_score")),
                "evidence_layers": edge["Evidence_layers"],
                "source_accession": edge["Source_accession"],
                "doi": edge.get("DOI", ""),
                "source_url": edge["Source_URL"],
                "interpretation_limit": edge["Interpretation_limit"],
            }
        )

    l1_ontology = []
    for mid in sorted(MECHANISM_EN):
        example = next(x for x in l1_rows if x["Mechanism_L1_ID"] == mid)
        l1_ontology.append(
            {
                "mechanism_l1_id": mid,
                "mechanism_l1_en": MECHANISM_EN[mid],
                "mechanism_l1_cn": example["Mechanism_L1_CN"],
                "eligible_seed_genes_n": int(example["Eligible_seed_genes"]),
            }
        )

    dictionary = [
        {"table": "candidate_summary", "field": "prediction_status", "type": "categorical", "definition": "NETWORK_RANKED, DESCRIPTIVE_ONLY, or UNRESOLVED according to public-data assessability."},
        {"table": "candidate_summary", "field": "primary_score_0_100", "type": "numeric", "definition": "Top within-candidate support score. Integrated and descriptive scores are not interchangeable."},
        {"table": "candidate_summary", "field": "recommended_comparison_scope", "type": "categorical", "definition": "WITHIN_GENE_ONLY: use the score to distinguish mechanisms for one candidate; do not rank causal genes across candidates."},
        {"table": "candidate_summary", "field": "top_vs_runner_up_margin", "type": "numeric", "definition": "Top score minus runner-up score on the same score type; <5 indicates weak mechanism specificity and 0 indicates a tie."},
        {"table": "candidate_summary", "field": "sensitivity_top_mechanism_stability", "type": "numeric", "definition": "Fraction of three network-layer weighting scenarios retaining the same top mechanism."},
        {"table": "l1_scores_long", "field": "primary_display_score_0_100", "type": "numeric", "definition": "Integrated score for network-ranked genes or separate descriptive score for descriptive-only genes."},
        {"table": "top_components_long", "field": "effective_weight_after_missing_renormalization", "type": "numeric", "definition": "Declared component weight divided by the sum of weights for components assessable for that candidate."},
        {"table": "top_components_long", "field": "weighted_contribution_to_primary_score", "type": "numeric", "definition": "Component score multiplied by its effective weight; contributions sum to the primary score subject to rounding."},
        {"table": "top_evidence_paths", "field": "path_directionality", "type": "categorical", "definition": "UNDIRECTED_ASSOCIATION_PATH; never interpret the shortest path as a regulatory cascade."},
        {"table": "direct_edges", "field": "direction", "type": "categorical", "definition": "Only TF-target evidence can be directed. Multilayer PPI/RNA/Ribo/ChIA edges remain undirected associations."},
        {"table": "tissue_contexts", "field": "effective_expression_percentile", "type": "numeric", "definition": "Expression support percentile in the reported mechanism-relevant tissue context."},
        {"table": "coverage_qc", "field": "network_layer_coverage_n_of_4", "type": "integer", "definition": "Number of assessable public-network layers among ChIA-PET, RNA, Ribo, and PPI."},
    ]

    payload = {
        "schema": {
            "schema_version": SCHEMA_VERSION,
            "display_version": DISPLAY_VERSION,
            "source_analysis_version": raw["method"].get("Analysis_version", ""),
            "generated_from": INPUT_JSON.name,
            "input_sha256": sha256(INPUT_JSON),
            "primary_use": "Explainable public-data mechanism hypotheses for candidate genes",
            "comparison_rule": "Mechanism scores are for within-gene mechanism prioritization; cross-gene causal ranking is not recommended.",
            "nam_evidence_included": False,
        },
        "candidate_summary": sorted(standardized_summary, key=lambda x: x["display_order"]),
        "l1_scores_long": sorted(standardized_l1, key=lambda x: (x["display_order"], x["mechanism_l1_id"])),
        "top_components_long": sorted(components, key=lambda x: (x["display_order"], list(COMPONENT_WEIGHTS).index(x["evidence_component"]))),
        "top_evidence_paths": sorted(top_paths, key=lambda x: x["display_order"]),
        "direct_edges": sorted(enriched_edges, key=lambda x: (x["display_order"], -x["is_top_mechanism_edge"], x["mechanism_l1_id"], x["seed_gene_id_v5"])),
        "tissue_contexts": sorted(tissue_contexts, key=lambda x: (x["display_order"], x["context_rank"])),
        "coverage_qc": sorted(coverage, key=lambda x: x["display_order"]),
        "mechanism_ontology_l1": l1_ontology,
        "data_dictionary": dictionary,
        "method": raw["method"],
        "input_manifest": raw["input_manifest"],
        "validation": raw["validation"],
    }
    return payload


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def gene_label(row: dict[str, object]) -> str:
    symbol = str(row.get("preferred_gene_symbol") or "").strip()
    symbol_text = f" / {symbol}" if symbol else ""
    return f"{row['gene_id_v5']}{symbol_text}  [{row['trait']}]"


def score_matrix(payload: dict[str, object], genes: list[str]) -> np.ndarray:
    lookup = {(x["gene_id_v5"], x["mechanism_l1_id"]): x["primary_display_score_0_100"] for x in payload["l1_scores_long"]}
    return np.array([[np.nan if lookup[(g, m)] is None else lookup[(g, m)] for m in sorted(MECHANISM_EN)] for g in genes], dtype=float)


def draw_heatmap(ax, matrix: np.ndarray, row_labels: list[str], title: str, cmap_name: str, top_ids: list[str], traits: list[str], panel: str) -> None:
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#E5E7EB")
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(range(6), [f"{m}\n{MECHANISM_SHORT[m]}" for m in sorted(MECHANISM_EN)])
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.text(-0.12, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    for label, trait in zip(ax.get_yticklabels(), traits):
        label.set_color(TRAIT_COLORS.get(trait, "#333333"))
        label.set_fontweight("bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value >= 67 else "#202020"
                weight = "bold" if sorted(MECHANISM_EN)[j] == top_ids[i] else "normal"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=6.8, fontweight=weight)
        if top_ids[i] in MECHANISM_EN:
            j = sorted(MECHANISM_EN).index(top_ids[i])
            ax.add_patch(patches.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=TRAIT_COLORS[traits[i]], linewidth=2.0))
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def figure_landscape(payload: dict[str, object]) -> None:
    summary = payload["candidate_summary"]
    by_gene = {x["gene_id_v5"]: x for x in summary}
    network = [x for x in summary if x["prediction_status"] == "NETWORK_RANKED"]
    descriptive = [x for x in summary if x["prediction_status"] == "DESCRIPTIVE_ONLY"]
    unresolved = [x for x in summary if x["prediction_status"] == "UNRESOLVED"]

    fig = plt.figure(figsize=(7.35, 10.1), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.65, 1.15, 3.7], hspace=0.65)
    ax_a = fig.add_subplot(gs[0])
    genes_a = [x["gene_id_v5"] for x in network]
    im_a = draw_heatmap(
        ax_a,
        score_matrix(payload, genes_a),
        [gene_label(x) for x in network],
        "Network-assessable candidates: integrated L1 mechanism support",
        "YlGnBu",
        [x["top_mechanism_l1_id"] for x in network],
        [x["trait"] for x in network],
        "A",
    )
    cbar = fig.colorbar(im_a, ax=ax_a, fraction=0.025, pad=0.015)
    cbar.set_label("Support score (0–100)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_b = fig.add_subplot(gs[1])
    genes_b = [x["gene_id_v5"] for x in descriptive]
    if genes_b:
        im_b = draw_heatmap(
            ax_b,
            score_matrix(payload, genes_b),
            [gene_label(x) for x in descriptive],
            "Descriptive-only candidates (separate scale; not comparable with panel A)",
            "Purples",
            [x["top_mechanism_l1_id"] for x in descriptive],
            [x["trait"] for x in descriptive],
            "B",
        )
        cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.025, pad=0.015)
        cbar_b.set_label("Descriptive score", fontsize=7)
        cbar_b.ax.tick_params(labelsize=6)
    unresolved_text = "; ".join(x["gene_id_v5"] for x in unresolved) or "None"
    ax_b.text(1.0, -0.45, f"Unresolved because public data were insufficient: {unresolved_text}", transform=ax_b.transAxes, ha="right", va="top", fontsize=7, color="#555555")

    ax_c = fig.add_subplot(gs[2])
    genes = [x["gene_id_v5"] for x in summary]
    comp_order = list(COMPONENT_WEIGHTS)
    comp_lookup = {(x["gene_id_v5"], x["evidence_component"]): x for x in payload["top_components_long"]}
    y = np.arange(len(genes))
    ax_c.set_xlim(-0.65, len(comp_order) - 0.35)
    ax_c.set_ylim(len(genes) - 0.5, -0.5)
    ax_c.set_xticks(range(len(comp_order)), [COMPONENT_LABELS[x] for x in comp_order], rotation=25, ha="right")
    ax_c.set_yticks(y, [gene_label(by_gene[g]) for g in genes])
    ax_c.tick_params(length=0)
    ax_c.set_title("Evidence components supporting each candidate's top hypothesis", loc="left", fontweight="bold", pad=7)
    ax_c.text(-0.12, 1.04, "C", transform=ax_c.transAxes, fontweight="bold", fontsize=10, va="bottom")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap("viridis")
    for i, gene in enumerate(genes):
        row = by_gene[gene]
        ax_c.get_yticklabels()[i].set_color(TRAIT_COLORS.get(row["trait"], "#333333"))
        ax_c.get_yticklabels()[i].set_fontweight("bold")
        for j, component in enumerate(comp_order):
            rec = comp_lookup[(gene, component)]
            value = rec["component_score_0_100"]
            if value is None:
                ax_c.scatter(j, i, s=58, marker="x", color="#A7A7A7", linewidth=1.2)
            else:
                ax_c.scatter(j, i, s=92, c=[cmap(norm(value))], edgecolor="white", linewidth=0.7)
                ax_c.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.8, color="white" if value > 48 else "#101010", fontweight="bold")
        mech = row["top_mechanism_l1_id"] or "NA"
        margin = row["top_vs_runner_up_margin"]
        margin_text = "NA" if margin is None else f"Δ={margin:.1f}"
        meta = f"{mech}; {tier_short(row['evidence_tier'])}; {margin_text}; layers={row['network_layers_assessable_n']}/4"
        ax_c.text(len(comp_order) - 0.15, i, meta, va="center", ha="left", fontsize=6.2, clip_on=False)
    ax_c.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_c = fig.colorbar(sm, ax=ax_c, fraction=0.025, pad=0.22)
    cbar_c.set_label("Component score", fontsize=7)
    cbar_c.ax.tick_params(labelsize=6)
    fig.suptitle("Explainable public-data mechanism hypotheses for maize flowering-time candidates", y=0.993, fontsize=11, fontweight="bold")
    fig.text(0.01, 0.006, "Outlined cells mark the top mechanism within each candidate. Scores support within-gene mechanism prioritization; they are not causal probabilities or cross-gene rankings. NAM evidence was not used.", fontsize=6.8)
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUTPUT_DIR / f"Fig_mechanism_landscape_v2.{ext}", **kwargs)
    plt.close(fig)


def draw_path_axis(ax, summary_row: dict[str, object], path_row: dict[str, object], panel: str) -> None:
    ax.axis("off")
    gene = summary_row["gene_id_v5"]
    trait = summary_row["trait"]
    top = summary_row["top_mechanism_l1_id"]
    runner = summary_row["runner_up_mechanism_l1_id"]
    margin = summary_row["top_vs_runner_up_margin"]
    ax.text(0.0, 0.96, panel, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax.text(0.045, 0.96, f"{gene}  ({trait})", transform=ax.transAxes, fontsize=9, fontweight="bold", color=TRAIT_COLORS[trait], va="top")
    ax.text(0.045, 0.80, f"Top: {top} {MECHANISM_EN.get(top, '')} | runner-up: {runner} | margin: {margin:.2f}" if margin is not None else f"Top: {top}", transform=ax.transAxes, fontsize=7.2, va="top")
    nodes = str(path_row.get("path_node_sequence") or "").split(">") if path_row.get("path_node_sequence") else []
    layers = str(path_row.get("path_edge_layer_sequence") or "").split(">") if path_row.get("path_edge_layer_sequence") else []
    if not nodes:
        ax.text(0.5, 0.42, "No public-network path was assessable", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#666666")
        return
    xs = np.linspace(0.09, 0.72, len(nodes))
    y = 0.43
    for idx in range(len(nodes) - 1):
        layer = layers[idx] if idx < len(layers) else ""
        ax.plot([xs[idx] + 0.025, xs[idx + 1] - 0.025], [y, y], transform=ax.transAxes, color=LAYER_COLORS.get(layer, "#777777"), linewidth=2.2, solid_capstyle="round")
        ax.text((xs[idx] + xs[idx + 1]) / 2, y - 0.12, layer, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=LAYER_COLORS.get(layer, "#555555"), fontweight="bold")
    for idx, node in enumerate(nodes):
        if idx == 0:
            color = TRAIT_COLORS[trait]
            label = node
        elif idx == len(nodes) - 1:
            color = MECHANISM_COLORS.get(top, "#555555")
            symbol = path_row.get("terminal_seed_symbol") or node
            label = f"{symbol}\n{node}"
        else:
            color = "#8B8B8B"
            label = node
        circle = patches.Circle((xs[idx], y), 0.028, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(xs[idx], y + 0.10, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3)
    ax.text(0.78, 0.59, f"Public association path\n{int(path_row['shortest_path_distance_edges'])} edge(s)", transform=ax.transAxes, fontsize=7, va="top")
    stability = summary_row.get("sensitivity_top_mechanism_stability")
    stability_text = "NA" if stability is None else f"{stability:.2f}"
    ax.text(0.78, 0.39, f"Evidence tier: {tier_short(summary_row['evidence_tier'])}\nWeight sensitivity: {stability_text}\nNetwork layers: {summary_row['network_layers_assessable_n']}/4", transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.045, 0.08, "Colored lines are undirected association layers; no arrow implies no regulatory direction.", transform=ax.transAxes, fontsize=6.3, color="#555555")


def figure_top3_paths(payload: dict[str, object]) -> None:
    selected = ["Zm00001eb349860", "Zm00001eb325180", "Zm00001eb341290"]
    summary = {x["gene_id_v5"]: x for x in payload["candidate_summary"]}
    paths = {x["gene_id_v5"]: x for x in payload["top_evidence_paths"]}
    fig, axes = plt.subplots(3, 1, figsize=(7.35, 6.6))
    for ax, gene, panel in zip(axes, selected, ["A", "B", "C"]):
        draw_path_axis(ax, summary[gene], paths[gene], panel)
    handles = [plt.Line2D([0], [0], color=color, lw=2.2, label=layer) for layer, color in LAYER_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.002), fontsize=7)
    fig.suptitle("Public-network evidence paths for the three candidates advanced to locus-level analysis", y=0.995, fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(top=0.94, bottom=0.08, hspace=0.28)
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUTPUT_DIR / f"Fig_top3_mechanism_evidence_paths_v2.{ext}", **kwargs)
    plt.close(fig)


def draw_profile(fig, summary_row: dict[str, object], l1_rows: list[dict[str, object]], component_rows: list[dict[str, object]], path_row: dict[str, object]) -> None:
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.85], width_ratios=[1.25, 1.0], hspace=0.48, wspace=0.48)
    ax_l1 = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_path = fig.add_subplot(gs[1, :])
    gene = summary_row["gene_id_v5"]
    trait = summary_row["trait"]
    symbol = summary_row.get("preferred_gene_symbol") or ""
    fig.suptitle(f"{gene}{' / ' + symbol if symbol else ''} — {trait}", x=0.02, ha="left", fontsize=11, fontweight="bold", color=TRAIT_COLORS[trait])

    rows = sorted(l1_rows, key=lambda x: x["mechanism_l1_id"], reverse=True)
    values = [x["primary_display_score_0_100"] for x in rows]
    mids = [x["mechanism_l1_id"] for x in rows]
    if any(v is not None for v in values):
        colors = [MECHANISM_COLORS[m] if m == summary_row["top_mechanism_l1_id"] else "#B8BDC5" for m in mids]
        numeric = [0 if v is None else v for v in values]
        bars = ax_l1.barh(range(6), numeric, color=colors, height=0.62)
        ax_l1.set_yticks(range(6), [f"{m}  {MECHANISM_EN[m]}" for m in mids])
        ax_l1.set_xlim(0, 100)
        ax_l1.set_xlabel("Support score (0–100)")
        for bar, value in zip(bars, values):
            if value is not None:
                ax_l1.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=7)
        ax_l1.set_title("L1 mechanism profile", loc="left", fontweight="bold")
        ax_l1.grid(axis="x", color="#E1E1E1", linewidth=0.5)
    else:
        ax_l1.axis("off")
        ax_l1.text(0.5, 0.5, "Unresolved: insufficient public data", transform=ax_l1.transAxes, ha="center", va="center", fontsize=9, color="#666666")

    comp_names = list(COMPONENT_WEIGHTS)
    comp_lookup = {x["evidence_component"]: x for x in component_rows}
    comp_values = [comp_lookup[x]["component_score_0_100"] for x in comp_names]
    ax_comp.set_xlim(0, 100)
    ax_comp.set_ylim(len(comp_names) - 0.5, -0.5)
    ax_comp.set_yticks(range(len(comp_names)), [COMPONENT_LABELS[x] for x in comp_names])
    ax_comp.set_xlabel("Component score")
    ax_comp.set_title("Top-hypothesis evidence", loc="left", fontweight="bold")
    for i, value in enumerate(comp_values):
        if value is None:
            ax_comp.scatter(4, i, marker="x", color="#A7A7A7", s=40)
            ax_comp.text(9, i, "NA", va="center", fontsize=7, color="#777777")
        else:
            ax_comp.hlines(i, 0, value, color="#D4D4D4", linewidth=1.2)
            ax_comp.scatter(value, i, s=48, color=TRAIT_COLORS[trait], edgecolor="white", linewidth=0.6)
            ax_comp.text(min(value + 3, 96), i, f"{value:.1f}", va="center", fontsize=7)
    ax_comp.grid(axis="x", color="#E1E1E1", linewidth=0.5)
    for ax in [ax_l1, ax_comp]:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    draw_path_axis(ax_path, summary_row, path_row, "")
    status = summary_row["prediction_status"].replace("_", " ").title()
    margin = summary_row["top_vs_runner_up_margin"]
    margin_text = "NA" if margin is None else f"{margin:.2f}"
    fig.text(0.985, 0.965, f"{status} | {tier_short(summary_row['evidence_tier'])} | top–runner margin {margin_text}", ha="right", va="top", fontsize=7)
    fig.text(0.02, 0.012, "Public-data mechanism hypothesis only. Compare mechanism scores within this candidate; do not interpret as a causal probability or experimental validation.", fontsize=6.2, color="#555555")


def figure_profiles(payload: dict[str, object]) -> None:
    l1_by_gene: dict[str, list[dict[str, object]]] = {}
    comp_by_gene: dict[str, list[dict[str, object]]] = {}
    for row in payload["l1_scores_long"]:
        l1_by_gene.setdefault(row["gene_id_v5"], []).append(row)
    for row in payload["top_components_long"]:
        comp_by_gene.setdefault(row["gene_id_v5"], []).append(row)
    paths = {x["gene_id_v5"]: x for x in payload["top_evidence_paths"]}
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "candidate9_mechanism_profiles_v2.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in payload["candidate_summary"]:
            fig = plt.figure(figsize=(7.35, 5.15))
            draw_profile(fig, row, l1_by_gene[row["gene_id_v5"]], comp_by_gene[row["gene_id_v5"]], paths[row["gene_id_v5"]])
            fig.savefig(PROFILE_DIR / f"{row['gene_id_v5']}_mechanism_profile_v2.png", dpi=400)
            pdf.savefig(fig)
            plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 9.0))
    for ax, s in zip(axes.flat, payload["candidate_summary"]):
        gene = s["gene_id_v5"]
        rows = sorted(l1_by_gene[gene], key=lambda x: x["mechanism_l1_id"])
        values = [x["primary_display_score_0_100"] for x in rows]
        mids = [x["mechanism_l1_id"] for x in rows]
        if any(v is not None for v in values):
            colors = [MECHANISM_COLORS[m] if m == s["top_mechanism_l1_id"] else "#C4C8CD" for m in mids]
            ax.barh(range(6), [0 if v is None else v for v in values], color=colors, height=0.64)
            ax.set_yticks(range(6), mids)
            ax.set_xlim(0, 100)
            ax.grid(axis="x", color="#E4E4E4", linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Unresolved", transform=ax.transAxes, ha="center", va="center", color="#777777")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"{gene}\n{s['trait']} | {s['top_mechanism_l1_id'] or 'NA'}", fontsize=8, color=TRAIT_COLORS[s["trait"]], fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Candidate-specific L1 mechanism profiles", fontsize=11, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "Profiles are interpreted within candidate. Grey bars are alternative mechanisms; colored bars mark the top hypothesis.", ha="center", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "Fig_candidate9_mechanism_profiles_overview_v2.png", dpi=500)
    fig.savefig(OUTPUT_DIR / "Fig_candidate9_mechanism_profiles_overview_v2.pdf")
    plt.close(fig)


def write_caption(payload: dict[str, object]) -> None:
    text = f"""# Figure legends and interpretation notes (v2)

## Fig_mechanism_landscape_v2

Explainable public-data mechanism hypotheses for nine co-localized maize flowering-time candidate genes. (A) Integrated L1 mechanism-support scores for candidates assessable in at least one public multilayer-network layer. The outlined cell marks the highest-scoring mechanism within each candidate. (B) Descriptive-only scores for candidates lacking multilayer-network connectivity; these scores use a separate scale and must not be compared with panel A. The unresolved candidate is reported without a mechanism assignment. (C) Evidence components supporting each candidate's top hypothesis. Crosses indicate unavailable components. Values are evidence-prioritization scores, not causal probabilities. The analysis excludes NAM GWAS, QTL, LD, haplotype, founder-allele, and additive-effect evidence.

## Fig_top3_mechanism_evidence_paths_v2

Shortest public-network association paths linking the three candidates advanced to locus-level analysis with flowering-time mechanism seed genes. Edge colors denote ChIA-PET, RNA coexpression, Ribo cotranslation, or protein-interaction layers. Lines have no arrowheads because these public multilayer-network paths are undirected associations; they are explanatory paths rather than regulatory cascades. The mechanism label, top-versus-runner-up margin, evidence tier, layer-weight sensitivity, and network coverage are shown for each candidate.

## Reporting rules

- Use “prioritized as” or “public-data evidence supports” rather than “proved” or “causes.”
- Compare L1 mechanisms within the same candidate. Do not rank causal genes across candidates from these mechanism scores.
- Keep network-ranked, descriptive-only, and unresolved candidates visually and verbally separate.
- L2 assignments are descriptive refinements only; the formal propagation level is L1.
- Only TF-target edges can be drawn with arrows. PPI, coexpression, cotranslation, and ChIA-PET edges remain undirected.

Schema: {payload['schema']['schema_version']}
"""
    (OUTPUT_DIR / "mechanism_figure_legends_v2.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    payload = build_standardized_payload(raw)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_caption(payload)
    print(json.dumps({
        "output_json": str(OUTPUT_JSON),
        "candidate_rows": len(payload["candidate_summary"]),
        "l1_rows": len(payload["l1_scores_long"]),
        "component_rows": len(payload["top_components_long"]),
        "path_rows": len(payload["top_evidence_paths"]),
        "direct_edge_rows": len(payload["direct_edges"]),
        "tissue_context_rows": len(payload["tissue_contexts"]),
        "figure_data_ready": True,
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize raw mechanism-inference JSON for tables and figures.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON, help="Raw mechanism-inference JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for standardized JSON and figure legends.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_JSON = args.input_json.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    PROFILE_DIR = OUTPUT_DIR / "candidate_profiles"
    OUTPUT_JSON = OUTPUT_DIR / "candidate9_mechanism_results_v2.json"
    main()
