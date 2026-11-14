from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "Mechanism_input"
DEFAULT_OUTPUT_JSON = PROJECT_DIR / "ML_prediction_results_reproduced" / "mechanism_prediction" / "raw" / "mechanism_analysis_results.json"
INPUT_DIR = DEFAULT_INPUT_DIR
OUTPUT_JSON = DEFAULT_OUTPUT_JSON
WORK_DIR = OUTPUT_JSON.parent

VERSION = "Maize_candidate_mechanism_inference_v1.0_2026-08-27"
RWR_RESTART = 0.50
RWR_TOL = 1e-11
RWR_MAX_ITER = 200
PATH_MAX_DEPTH = 4

LAYER_COLUMNS = {
    "CHIA": "ChIA_PET_weight",
    "RNA": "RNA_coexpression_weight",
    "RIBO": "Ribo_cotranslation_weight",
    "PPI": "PPI_weight",
}
LAYER_LABELS = {
    "CHIA": "ChIA-PET chromatin contact",
    "RNA": "RNA coexpression",
    "RIBO": "Ribo cotranslation",
    "PPI": "protein interaction",
}
SCENARIO_WEIGHTS = {
    "EQUAL_LAYER_PRIMARY": {"CHIA": 0.25, "RNA": 0.25, "RIBO": 0.25, "PPI": 0.25},
    "INTERACTION_HEAVY_SENSITIVITY": {"CHIA": 0.20, "RNA": 0.20, "RIBO": 0.20, "PPI": 0.40},
    "EXPRESSION_HEAVY_SENSITIVITY": {"CHIA": 0.10, "RNA": 0.40, "RIBO": 0.40, "PPI": 0.10},
}
COMPONENT_WEIGHTS = {
    "network": 0.60,
    "tf_direct": 0.15,
    "tissue": 0.15,
    "annotation": 0.05,
    "trait": 0.05,
}
PRIORITY_WEIGHT = {"CORE": 1.0, "SUPPORTING": 0.75, "DOWNSTREAM": 0.50}


FILES = {
    "query": "candidate9_mechanism_query.tsv",
    "ontology": "FT_mechanism_ontology.tsv",
    "seeds": "FT_mechanism_seed_annotations_v5.tsv",
    "mapping": "maize_gene_ID_mapping_v5.tsv",
    "network": "maize_FT_multilayer_network_edges_v5.tsv",
    "tf": "maize_TF_target_regulatory_edges_v5.tsv",
    "chromatin": "maize_chromatin_regulatory_edges_v5.tsv",
    "expression_metadata": "maize_expression_sample_metadata.tsv",
    "nodes": "maize_multilayer_network_nodes_v5.tsv",
    "annotations": "maize_gene_function_annotations_v5.tsv",
    "tissue": "candidate9_tissue_context_evidence.tsv",
}


def clean(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"", "na", "nan", "none"} else s


def number(value: object) -> float | None:
    s = clean(value)
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def as_int(value: object) -> int:
    x = number(value)
    return int(x) if x is not None else 0


def norm_text(value: object) -> str:
    s = clean(value).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        INPUT_DIR / FILES[name],
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        **kwargs,
    )


def empirical_midrank_percentile(reference: np.ndarray, value: float) -> tuple[float, int]:
    if reference.size == 0 or not math.isfinite(value):
        return (math.nan, 0)
    ref = np.sort(reference)
    left = int(np.searchsorted(ref, value, side="left"))
    right = int(np.searchsorted(ref, value, side="right"))
    return ((left + 0.5 * (right - left)) / ref.size, int(ref.size))


def degree_bin(degree: int) -> int:
    return int(math.floor(math.log2(max(1, degree))))


def degree_bin_label(bin_id: int) -> str:
    low = 2**bin_id
    high = 2 ** (bin_id + 1) - 1
    return f"{low}-{high}"


def weighted_available(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    available = [(k, v) for k, v in values.items() if v is not None and math.isfinite(v)]
    denom = sum(weights[k] for k, _ in available)
    if denom <= 0:
        return None
    return sum(weights[k] * float(v) for k, v in available) / denom


def run_rwr(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    norm_weight: np.ndarray,
    seed_indices: np.ndarray,
    seed_weights: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    restart = np.zeros(n_nodes, dtype=np.float64)
    restart[seed_indices] = seed_weights / seed_weights.sum()
    p = restart.copy()
    delta = math.inf
    for iteration in range(1, RWR_MAX_ITER + 1):
        walk = np.bincount(
            dst,
            weights=p[src] * norm_weight,
            minlength=n_nodes,
        )
        updated = RWR_RESTART * restart + (1.0 - RWR_RESTART) * walk
        delta = float(np.abs(updated - p).sum())
        p = updated
        if delta < RWR_TOL:
            break
    return p, iteration, delta


def context_token_matches(token: str, row: pd.Series) -> tuple[bool, str]:
    token_n = norm_text(token)
    tissue = norm_text(row.get("Biological_tissue", ""))
    subregion = norm_text(row.get("Tissue_subregion", ""))
    organ = norm_text(row.get("Organ_category", ""))
    stage = norm_text(row.get("Developmental_stage", ""))
    phase = norm_text(row.get("Developmental_phase", ""))
    priority = clean(row.get("FT_context_priority", ""))
    haystack = " ".join([tissue, subregion, organ, stage, phase])
    if token_n and token_n in haystack:
        return True, "EXACT_TEXT"

    leaf = organ == "leaf" or "leaf" in tissue
    sam = organ == "shoot apical meristem" or "apical meristem" in tissue or subregion == "sam"
    female = organ == "female inflorescence" or tissue == "ear"
    male = organ == "male inflorescence" or tissue == "tassel"
    infl = organ == "inflorescence" or tissue == "inflorescence" or female or male
    stem = organ in {"stem", "internode"} or tissue in {"stem", "internode"}
    seedling = organ == "whole seedling" or tissue == "seedling"

    if any(x in token_n for x in ["leaf", "phloem", "source sink", "source", "sink"]):
        if leaf or ("phloem" in token_n and stem):
            return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["shoot apical meristem", "shoot apex", "sam", "floral commitment"]):
        if sam:
            return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["transition", "vegetative phase change", "juvenile to adult"]):
        if sam or leaf:
            return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["ear", "female", "ovary", "silk", "stigma", "pre pollination"]):
        if female:
            return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["tassel", "male", "anther", "pollen", "anthesis", "meiocyte", "tapetum"]):
        if male:
            return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["inflorescence", "branch meristem", "spikelet meristem", "meristem determinacy"]):
        if infl:
            return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["stem", "vascular"]):
        if stem:
            return True, "ONTOLOGY_PROXY"
    if "seedling" in token_n and (seedling or (leaf and stage in {"v3", "6das", "14das"})):
        return True, "ONTOLOGY_PROXY"
    if any(x in token_n for x in ["flowering stage", "flowering stress"]):
        if priority == "CORE":
            return True, "BROAD_FT_CONTEXT_PROXY"
    return False, ""


def trait_score(trait: str, relevance_values: list[str]) -> tuple[float, str]:
    best = 0.0
    basis = "NO_EXPLICIT_TRAIT_MATCH"
    for relevance in relevance_values:
        tokens = [x.strip() for x in relevance.split(";") if x.strip()]
        if trait in tokens:
            return 100.0, "EXACT_TRAIT"
        if f"{trait}_indirect" in tokens and best < 60.0:
            best = 60.0
            basis = "INDIRECT_TRAIT"
    return best, basis


def annotation_support(annotation_text: str, terms: list[str]) -> tuple[float, list[str]]:
    text_n = norm_text(annotation_text)
    matched: list[str] = []
    for term in terms:
        term_n = norm_text(term)
        if len(term_n) < 3:
            continue
        if re.search(rf"\b{re.escape(term_n)}\b", text_n):
            matched.append(term)
    matched = sorted(set(matched), key=lambda x: (-len(x), x.lower()))
    score = min(100.0, len(matched) * (100.0 / 3.0))
    return score, matched


def tissue_support(
    tissue_rows: pd.DataFrame,
    relevant_tokens: list[str],
) -> dict[str, object]:
    matched_rows: list[tuple[pd.Series, list[str], list[str]]] = []
    for _, row in tissue_rows.iterrows():
        matched_tokens: list[str] = []
        bases: list[str] = []
        for token in relevant_tokens:
            yes, basis = context_token_matches(token, row)
            if yes:
                matched_tokens.append(token)
                bases.append(basis)
        if matched_tokens:
            matched_rows.append((row, matched_tokens, bases))

    scored = []
    for row, tokens, bases in matched_rows:
        pct_values = [
            number(row.get("RNA_genome_background_percentile", "")),
            number(row.get("Ribo_genome_background_percentile", "")),
        ]
        pct_values = [x for x in pct_values if x is not None]
        if not pct_values:
            continue
        raw = float(np.mean(pct_values))
        pweight = PRIORITY_WEIGHT.get(clean(row.get("FT_context_priority", "")), 0.50)
        effective = raw * pweight
        scored.append({
            "Context_ID": clean(row.get("Context_ID", "")),
            "Biological_tissue": clean(row.get("Biological_tissue", "")),
            "Developmental_stage": clean(row.get("Developmental_stage", "")),
            "Priority": clean(row.get("FT_context_priority", "")),
            "Raw_percentile": raw,
            "Effective_percentile": effective,
            "Matched_tokens": sorted(set(tokens)),
            "Match_basis": sorted(set(bases)),
        })
    scored.sort(key=lambda x: (-x["Effective_percentile"], x["Context_ID"]))
    top = scored[:3]
    score = float(np.mean([x["Effective_percentile"] for x in top]) * 100.0) if top else None
    return {
        "score": score,
        "matched_contexts": len(matched_rows),
        "assessable_contexts": len(scored),
        "top_contexts": top,
    }


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(INPUT_DIR / value) for value in FILES.values() if not (INPUT_DIR / value).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + "; ".join(missing))

    query = read_tsv("query")
    query = query.loc[query["Include"] == "1"].copy()
    if len(query) != 9 or query["Gene_ID_v5"].nunique() != 9:
        raise ValueError("candidate query must contain exactly nine unique Include=1 genes")
    candidates = query["Gene_ID_v5"].tolist()
    candidate_set = set(candidates)
    candidate_info = query.set_index("Gene_ID_v5").to_dict("index")

    ontology = read_tsv("ontology")
    seeds = read_tsv("seeds")
    if set(ontology["Primary_RWR_level"]) != {"L1"}:
        raise ValueError("This implementation requires Primary_RWR_level=L1 for every ontology row")
    l1_ids = ontology["Mechanism_L1_ID"].drop_duplicates().tolist()
    if len(l1_ids) != 6:
        raise ValueError(f"Expected six L1 mechanisms, observed {len(l1_ids)}")
    l2_ids = ontology["Mechanism_L2_ID"].tolist()

    l1_meta: dict[str, dict[str, object]] = {}
    l2_meta: dict[str, dict[str, object]] = {}
    for l1, group in ontology.groupby("Mechanism_L1_ID", sort=False):
        first = group.iloc[0]
        l1_meta[l1] = {
            "Mechanism_L1": clean(first["Mechanism_L1"]),
            "Mechanism_L1_CN": clean(first["Mechanism_L1_CN"]),
            "Ontology_roles": sorted(set(group["Ontology_role"])),
            "Relevant_tissue_tokens": sorted({x.strip() for s in group["Relevant_tissue_stage"] for x in s.split(";") if x.strip()}),
            "Trait_relevance": group["Trait_relevance"].tolist(),
            "Include_terms": sorted({x.strip() for s in group["Include_terms"] for x in s.split(";") if x.strip()}),
            "L2_IDs": group["Mechanism_L2_ID"].tolist(),
        }
    for _, row in ontology.iterrows():
        l2 = row["Mechanism_L2_ID"]
        l2_meta[l2] = {
            "Mechanism_L1_ID": row["Mechanism_L1_ID"],
            "Mechanism_L1": clean(row["Mechanism_L1"]),
            "Mechanism_L1_CN": clean(row["Mechanism_L1_CN"]),
            "Mechanism_L2": clean(row["Mechanism_L2"]),
            "Mechanism_L2_CN": clean(row["Mechanism_L2_CN"]),
            "Ontology_role": clean(row["Ontology_role"]),
            "Relevant_tissue_tokens": [x.strip() for x in row["Relevant_tissue_stage"].split(";") if x.strip()],
            "Trait_relevance": [row["Trait_relevance"]],
            "Include_terms": [x.strip() for x in row["Include_terms"].split(";") if x.strip()],
            "Min_primary_seed_count": as_int(row["Min_primary_seed_count"]),
            "Primary_RWR_level": row["Primary_RWR_level"],
            "Sparse_seed_action": row["Sparse_seed_action"],
        }

    eligible_l1 = seeds.loc[seeds["Seed_eligible_L1"] == "1"].copy()
    l1_seed_map: dict[str, dict[str, float]] = {}
    gene_to_l1: dict[str, set[str]] = defaultdict(set)
    gene_to_l2: dict[str, set[str]] = defaultdict(set)
    seed_details: dict[str, dict[str, str]] = {}
    for l1, group in eligible_l1.groupby("Mechanism_L1_ID", sort=False):
        per_gene: dict[str, float] = {}
        for _, row in group.iterrows():
            gene = clean(row["Gene_ID_v5"])
            if not gene:
                continue
            per_gene[gene] = max(per_gene.get(gene, 0.0), number(row["Seed_weight"]) or 0.0)
            gene_to_l1[gene].add(l1)
            l2 = clean(row["Mechanism_L2_ID"])
            if l2:
                gene_to_l2[gene].add(l2)
            seed_details.setdefault(gene, {
                "Gene_symbol": clean(row["Gene_symbol"]),
                "Seed_confidence": clean(row["Seed_confidence"]),
                "Evidence_type": clean(row["Evidence_type"]),
                "Source_URL": clean(row["Source_URL"]),
            })
        l1_seed_map[l1] = per_gene
    if set(l1_seed_map) != set(l1_ids):
        raise ValueError("Every L1 mechanism must have eligible seeds")

    # All annotated L2 assignments are retained for descriptive direct evidence;
    # they are not used for L2 propagation because the ontology declares L1 as primary.
    for _, row in seeds.iterrows():
        gene = clean(row["Gene_ID_v5"])
        l1 = clean(row["Mechanism_L1_ID"])
        l2 = clean(row["Mechanism_L2_ID"])
        if gene and l1:
            gene_to_l1[gene].add(l1)
        if gene and l2:
            gene_to_l2[gene].add(l2)
        if gene:
            seed_details.setdefault(gene, {
                "Gene_symbol": clean(row["Gene_symbol"]),
                "Seed_confidence": clean(row["Seed_confidence"]),
                "Evidence_type": clean(row["Evidence_type"]),
                "Source_URL": clean(row["Source_URL"]),
            })
    all_seed_genes = set(gene_to_l1)

    node_cols = [
        "Gene_ID_v5", "Gene_symbol", "In_integrated_undirected_network",
        "Degree_integrated_unique_pairs", "Degree_ChIA_PET", "Degree_RNA_coexpression",
        "Degree_Ribo_cotranslation", "Degree_PPI", "N_undirected_layers_present",
        "Undirected_layers_present", "In_TF_regulatory_network", "TF_out_degree",
        "TF_in_degree", "In_any_network", "Is_network_isolate", "Node_QC",
    ]
    nodes = read_tsv("nodes", usecols=node_cols)
    if nodes["Gene_ID_v5"].duplicated().any():
        raise ValueError("node table contains duplicated Gene_ID_v5")
    genes = nodes["Gene_ID_v5"].tolist()
    gene_index = {gene: i for i, gene in enumerate(genes)}
    n_nodes = len(genes)
    if n_nodes != 39801:
        raise ValueError(f"Expected 39,801 network nodes, observed {n_nodes}")
    node_lookup = nodes.set_index("Gene_ID_v5").to_dict("index")

    function_cols = [
        "Gene_ID_v5", "Preferred_gene_symbol", "Consensus_function_description",
        "GO_BP_terms", "GO_MF_terms", "GO_CC_terms", "InterPro_short_descriptions",
        "Plant_Reactome_Pathway_IDs", "Arabidopsis_ortholog_symbols", "Rice_ortholog_symbols",
        "Function_annotation_level", "Function_annotation_QC",
    ]
    annotations_all = read_tsv("annotations", usecols=function_cols)
    annotations = annotations_all.loc[annotations_all["Gene_ID_v5"].isin(candidate_set)].set_index("Gene_ID_v5").to_dict("index")
    if set(annotations) != candidate_set:
        raise ValueError("functional annotation table does not cover all candidates")

    tissue_cols = [
        "Gene_ID_v5", "Context_ID", "Biological_tissue", "Tissue_subregion",
        "Organ_category", "Developmental_stage", "Developmental_phase",
        "FT_context_class", "FT_context_priority", "RNA_genome_background_percentile",
        "Ribo_genome_background_percentile", "Context_expression_evidence_class",
        "Candidate_tissue_evidence_QC",
    ]
    tissue = read_tsv("tissue", usecols=tissue_cols)
    if len(tissue) != 297 or tissue.groupby("Gene_ID_v5")["Context_ID"].nunique().min() != 33:
        raise ValueError("tissue evidence must contain 33 contexts for every candidate")

    # Read the public undirected network once and materialize each layer separately.
    layer_parts: dict[str, dict[str, list[np.ndarray]]] = {
        layer: {"u": [], "v": [], "w": []} for layer in LAYER_COLUMNS
    }
    union_u_parts: list[np.ndarray] = []
    union_v_parts: list[np.ndarray] = []
    direct_multilayer_edges: list[dict[str, object]] = []
    network_cols = [
        "Gene1_ID_v5", "Gene2_ID_v5", "Direction", *LAYER_COLUMNS.values(),
        "N_evidence_layers", "Evidence_layers", "Integrated_raw_weight_sum",
        "Edge_QC", "Source_accession", "Source_DOI", "Source_URL", "Network_version",
    ]
    for chunk in pd.read_csv(
        INPUT_DIR / FILES["network"], sep="\t", usecols=network_cols,
        dtype=str, keep_default_na=False, na_filter=False, chunksize=150_000,
    ):
        u = chunk["Gene1_ID_v5"].map(gene_index).to_numpy()
        v = chunk["Gene2_ID_v5"].map(gene_index).to_numpy()
        if pd.isna(u).any() or pd.isna(v).any():
            raise ValueError("network endpoint absent from node table")
        u = u.astype(np.int32)
        v = v.astype(np.int32)
        union_u_parts.append(u)
        union_v_parts.append(v)
        for layer, column in LAYER_COLUMNS.items():
            w = pd.to_numeric(chunk[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            mask = w > 0
            if mask.any():
                layer_parts[layer]["u"].append(u[mask])
                layer_parts[layer]["v"].append(v[mask])
                layer_parts[layer]["w"].append(w[mask])
        pair_mask = (
            (chunk["Gene1_ID_v5"].isin(candidate_set) & chunk["Gene2_ID_v5"].isin(all_seed_genes))
            | (chunk["Gene2_ID_v5"].isin(candidate_set) & chunk["Gene1_ID_v5"].isin(all_seed_genes))
        )
        for _, row in chunk.loc[pair_mask].iterrows():
            g1, g2 = row["Gene1_ID_v5"], row["Gene2_ID_v5"]
            candidate = g1 if g1 in candidate_set else g2
            seed = g2 if candidate == g1 else g1
            for l1 in sorted(gene_to_l1[seed]):
                direct_multilayer_edges.append({
                    "Evidence_source": "MULTILAYER_DIRECT_EDGE",
                    "Candidate_Gene_ID_v5": candidate,
                    "Seed_Gene_ID_v5": seed,
                    "Seed_symbol": seed_details.get(seed, {}).get("Gene_symbol", ""),
                    "Mechanism_L1_ID": l1,
                    "Mechanism_L2_IDs": ";".join(sorted(gene_to_l2.get(seed, set()))),
                    "Direction": "undirected",
                    "Regulatory_sign": "not_applicable",
                    "Confidence_score": "",
                    "Evidence_layers": row["Evidence_layers"],
                    "ChIA_PET_weight": row["ChIA_PET_weight"],
                    "RNA_coexpression_weight": row["RNA_coexpression_weight"],
                    "Ribo_cotranslation_weight": row["Ribo_cotranslation_weight"],
                    "PPI_weight": row["PPI_weight"],
                    "Source_accession": row["Source_accession"],
                    "DOI": row["Source_DOI"],
                    "Source_URL": row["Source_URL"],
                    "Interpretation_limit": "Functional association; an undirected edge does not establish regulatory direction or causality.",
                })

    union_u = np.concatenate(union_u_parts).astype(np.int32, copy=False)
    union_v = np.concatenate(union_v_parts).astype(np.int32, copy=False)
    if union_u.size != 2_549_138:
        raise ValueError(f"Unexpected integrated edge count: {union_u.size}")

    l1_layer_results: dict[tuple[str, str], dict[str, object]] = {}
    layer_qc: list[dict[str, object]] = []
    layer_degrees: dict[str, np.ndarray] = {}
    for layer in LAYER_COLUMNS:
        u = np.concatenate(layer_parts[layer]["u"]).astype(np.int32, copy=False)
        v = np.concatenate(layer_parts[layer]["v"]).astype(np.int32, copy=False)
        w = np.concatenate(layer_parts[layer]["w"]).astype(np.float64, copy=False)
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        weights = np.concatenate([w, w])
        strength = np.bincount(src, weights=weights, minlength=n_nodes)
        degree = np.bincount(src, minlength=n_nodes).astype(np.int32)
        norm_weight = weights / strength[src]
        layer_degrees[layer] = degree
        layer_seed_union = {gene for genes_by_l1 in l1_seed_map.values() for gene in genes_by_l1 if gene in gene_index and degree[gene_index[gene]] > 0}
        for l1 in l1_ids:
            seeds_connected = [g for g in l1_seed_map[l1] if g in gene_index and degree[gene_index[g]] > 0]
            seed_idx = np.array([gene_index[g] for g in seeds_connected], dtype=np.int32)
            seed_w = np.array([l1_seed_map[l1][g] for g in seeds_connected], dtype=np.float64)
            if seed_idx.size == 0:
                p = np.zeros(n_nodes, dtype=np.float64)
                iterations, delta = 0, math.nan
            else:
                p, iterations, delta = run_rwr(n_nodes, src, dst, norm_weight, seed_idx, seed_w)
            mechanism_seed_indices = {gene_index[g] for g in l1_seed_map[l1] if g in gene_index}
            reference_mask = degree > 0
            if mechanism_seed_indices:
                reference_mask[list(mechanism_seed_indices)] = False
            global_ref = p[reference_mask]
            for candidate in candidates:
                idx = gene_index[candidate]
                deg = int(degree[idx])
                raw = float(p[idx])
                if deg == 0:
                    global_pct = degree_pct = None
                    ref_n = bin_n = 0
                    reachability = "LAYER_NOT_ASSESSABLE"
                    bin_id = None
                elif raw <= 0:
                    global_pct = degree_pct = 0.0
                    ref_n = int(global_ref.size)
                    bin_id = degree_bin(deg)
                    bin_mask = reference_mask & (np.floor(np.log2(np.maximum(degree, 1))).astype(np.int32) == bin_id)
                    bin_n = int(bin_mask.sum())
                    reachability = "NO_SEED_REACHABILITY"
                else:
                    global_pct, ref_n = empirical_midrank_percentile(global_ref, raw)
                    bin_id = degree_bin(deg)
                    all_bins = np.floor(np.log2(np.maximum(degree, 1))).astype(np.int32)
                    bin_mask = reference_mask & (all_bins == bin_id)
                    degree_pct, bin_n = empirical_midrank_percentile(p[bin_mask], raw)
                    reachability = "REACHABLE_FROM_SEEDS"
                l1_layer_results[(candidate, l1, layer)] = {
                    "raw_rwr": raw if deg > 0 else None,
                    "global_percentile": global_pct,
                    "degree_adjusted_percentile": degree_pct,
                    "degree": deg,
                    "degree_bin": degree_bin_label(bin_id) if bin_id is not None else "",
                    "degree_bin_reference_n": bin_n,
                    "global_reference_n": ref_n,
                    "reachability": reachability,
                }
            layer_qc.append({
                "Layer": layer,
                "Layer_label": LAYER_LABELS[layer],
                "Mechanism_L1_ID": l1,
                "Undirected_edges": int(u.size),
                "Connected_nodes": int((degree > 0).sum()),
                "Eligible_seed_genes_total": len(l1_seed_map[l1]),
                "Eligible_seed_genes_connected": len(seeds_connected),
                "Seed_weight_connected_sum": round(float(seed_w.sum()), 6),
                "RWR_restart_probability": RWR_RESTART,
                "Iterations": iterations,
                "Final_L1_delta": delta,
                "Converged": bool(iterations < RWR_MAX_ITER or delta < RWR_TOL),
            })
        del src, dst, weights, strength, norm_weight, u, v, w

    # Multi-source, unweighted shortest paths are explanatory only and capped at four edges.
    src_union = np.concatenate([union_u, union_v])
    dst_union = np.concatenate([union_v, union_u])
    order = np.argsort(src_union, kind="stable")
    src_sorted = src_union[order]
    neighbors = dst_union[order]
    offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(src_sorted, minlength=n_nodes))
    path_records: list[dict[str, object]] = []
    path_pairs_needed: set[tuple[str, str]] = set()
    for l1 in l1_ids:
        seed_genes = [g for g in l1_seed_map[l1] if g in gene_index]
        seed_idx = [gene_index[g] for g in seed_genes]
        distance = np.full(n_nodes, -1, dtype=np.int16)
        parent = np.full(n_nodes, -1, dtype=np.int32)
        root_seed = np.full(n_nodes, -1, dtype=np.int32)
        queue: deque[int] = deque()
        for idx in seed_idx:
            distance[idx] = 0
            root_seed[idx] = idx
            queue.append(idx)
        while queue:
            current = queue.popleft()
            if distance[current] >= PATH_MAX_DEPTH:
                continue
            for neighbor in neighbors[offsets[current]:offsets[current + 1]]:
                neighbor = int(neighbor)
                if distance[neighbor] != -1:
                    continue
                distance[neighbor] = distance[current] + 1
                parent[neighbor] = current
                root_seed[neighbor] = root_seed[current]
                queue.append(neighbor)
        for candidate in candidates:
            idx = gene_index[candidate]
            d = int(distance[idx])
            path_nodes: list[str] = []
            if d >= 0:
                cur = idx
                path_nodes.append(genes[cur])
                while parent[cur] >= 0:
                    cur = int(parent[cur])
                    path_nodes.append(genes[cur])
                for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                    path_pairs_needed.add(tuple(sorted((a, b))))
            seed_gene = genes[int(root_seed[idx])] if root_seed[idx] >= 0 else ""
            path_records.append({
                "Candidate_Gene_ID_v5": candidate,
                "Mechanism_L1_ID": l1,
                "Mechanism_L1_CN": l1_meta[l1]["Mechanism_L1_CN"],
                "Shortest_path_distance_edges": d if d >= 0 else None,
                "Path_within_max_depth": 1 if d >= 0 else 0,
                "Path_node_sequence": ">".join(path_nodes),
                "Terminal_seed_Gene_ID_v5": seed_gene,
                "Terminal_seed_symbol": seed_details.get(seed_gene, {}).get("Gene_symbol", ""),
                "Path_edge_layer_sequence": "",
                "Interpretation_limit": "Unweighted shortest association path for explanation only; intermediate edges may be undirected and non-causal.",
            })

    # Recover layer details for only the explanatory path edges.
    path_edge_details: dict[tuple[str, str], str] = {}
    if path_pairs_needed:
        for chunk in pd.read_csv(
            INPUT_DIR / FILES["network"], sep="\t",
            usecols=["Gene1_ID_v5", "Gene2_ID_v5", "Evidence_layers"],
            dtype=str, keep_default_na=False, na_filter=False, chunksize=200_000,
        ):
            for _, row in chunk.iterrows():
                pair = tuple(sorted((row["Gene1_ID_v5"], row["Gene2_ID_v5"])))
                if pair in path_pairs_needed:
                    path_edge_details[pair] = row["Evidence_layers"]
    for record in path_records:
        path_nodes = record["Path_node_sequence"].split(">") if record["Path_node_sequence"] else []
        record["Path_edge_layer_sequence"] = ">".join(
            path_edge_details.get(tuple(sorted((a, b))), "UNKNOWN")
            for a, b in zip(path_nodes[:-1], path_nodes[1:])
        )

    # Directed TF evidence: only one-edge candidate--mechanism-seed relations.
    direct_tf_edges: list[dict[str, object]] = []
    tf_candidate_presence = {candidate: False for candidate in candidates}
    tf_cols = [
        "TF_ID_v5", "Target_ID_v5", "TF_symbol", "Target_symbol", "Direction",
        "Regulatory_sign", "Regulatory_sign_basis", "Evidence_type", "Binding_evidence",
        "Expression_support", "Directness_class", "Tissue_stage", "Confidence",
        "Confidence_score", "N_total_experiments", "PMID", "DOI", "Database",
        "Source_URL", "Edge_QC", "Network_version",
    ]
    for chunk in pd.read_csv(
        INPUT_DIR / FILES["tf"], sep="\t", usecols=tf_cols,
        dtype=str, keep_default_na=False, na_filter=False, chunksize=100_000,
    ):
        for candidate in candidates:
            tf_candidate_presence[candidate] = tf_candidate_presence[candidate] or bool(
                ((chunk["TF_ID_v5"] == candidate) | (chunk["Target_ID_v5"] == candidate)).any()
            )
        mask = (
            (chunk["TF_ID_v5"].isin(all_seed_genes) & chunk["Target_ID_v5"].isin(candidate_set))
            | (chunk["TF_ID_v5"].isin(candidate_set) & chunk["Target_ID_v5"].isin(all_seed_genes))
        )
        for _, row in chunk.loc[mask].iterrows():
            candidate = row["Target_ID_v5"] if row["Target_ID_v5"] in candidate_set else row["TF_ID_v5"]
            seed = row["TF_ID_v5"] if candidate == row["Target_ID_v5"] else row["Target_ID_v5"]
            relation = "SEED_TF_TO_CANDIDATE" if row["TF_ID_v5"] == seed else "CANDIDATE_TF_TO_SEED"
            for l1 in sorted(gene_to_l1[seed]):
                direct_tf_edges.append({
                    "Evidence_source": "TF_TARGET_DIRECTED_EDGE",
                    "Candidate_Gene_ID_v5": candidate,
                    "Seed_Gene_ID_v5": seed,
                    "Seed_symbol": seed_details.get(seed, {}).get("Gene_symbol", ""),
                    "Mechanism_L1_ID": l1,
                    "Mechanism_L2_IDs": ";".join(sorted(gene_to_l2.get(seed, set()))),
                    "Direction": relation,
                    "Regulatory_sign": row["Regulatory_sign"],
                    "Confidence_score": number(row["Confidence_score"]),
                    "Confidence": row["Confidence"],
                    "Directness_class": row["Directness_class"],
                    "Evidence_layers": ";".join(filter(None, [row["Binding_evidence"], row["Expression_support"]])),
                    "Tissue_stage": row["Tissue_stage"],
                    "N_total_experiments": as_int(row["N_total_experiments"]),
                    "ChIA_PET_weight": "",
                    "RNA_coexpression_weight": "",
                    "Ribo_cotranslation_weight": "",
                    "PPI_weight": "",
                    "Source_accession": row["Database"],
                    "PMID": row["PMID"],
                    "DOI": row["DOI"],
                    "Source_URL": row["Source_URL"],
                    "Interpretation_limit": "Directed public TF-target evidence supports regulatory orientation but does not prove that the edge controls flowering in the study population.",
                })

    chromatin = read_tsv("chromatin")
    direct_chromatin_edges: list[dict[str, object]] = []
    pair_mask = (
        (chromatin["Gene1_ID_v5"].isin(candidate_set) & chromatin["Gene2_ID_v5"].isin(all_seed_genes))
        | (chromatin["Gene2_ID_v5"].isin(candidate_set) & chromatin["Gene1_ID_v5"].isin(all_seed_genes))
    )
    for _, row in chromatin.loc[pair_mask].iterrows():
        g1, g2 = row["Gene1_ID_v5"], row["Gene2_ID_v5"]
        candidate = g1 if g1 in candidate_set else g2
        seed = g2 if candidate == g1 else g1
        for l1 in sorted(gene_to_l1[seed]):
            direct_chromatin_edges.append({
                "Evidence_source": "CHROMATIN_CONTACT_DIRECT_EDGE",
                "Candidate_Gene_ID_v5": candidate,
                "Seed_Gene_ID_v5": seed,
                "Seed_symbol": seed_details.get(seed, {}).get("Gene_symbol", ""),
                "Mechanism_L1_ID": l1,
                "Mechanism_L2_IDs": ";".join(sorted(gene_to_l2.get(seed, set()))),
                "Direction": row["Direction"],
                "Regulatory_sign": "not_established",
                "Confidence_score": number(row["Propagation_input_weight"]),
                "Confidence": "CONTACT_SUPPORT",
                "Directness_class": row["Contact_interpretation"],
                "Evidence_layers": row["Other_omics_support"],
                "Tissue_stage": "not_reported_in_processed_edge",
                "N_total_experiments": as_int(row["N_source_v4_edges"]),
                "ChIA_PET_weight": row["ChIA_PET_weight"],
                "RNA_coexpression_weight": row["RNA_coexpression_weight"],
                "Ribo_cotranslation_weight": row["Ribo_cotranslation_weight"],
                "PPI_weight": row["PPI_weight"],
                "Source_accession": row["Source_accession"],
                "PMID": row["PMID"],
                "DOI": row["DOI"],
                "Source_URL": row["Source_URL"],
                "Interpretation_limit": "Chromatin contact is undirected and does not by itself identify an enhancer, target gene, or causal regulatory direction.",
            })

    tf_by_candidate_l1: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    tf_by_candidate_l2: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for edge in direct_tf_edges:
        tf_by_candidate_l1[(edge["Candidate_Gene_ID_v5"], edge["Mechanism_L1_ID"])].append(edge)
        for l2 in str(edge["Mechanism_L2_IDs"]).split(";"):
            if l2:
                tf_by_candidate_l2[(edge["Candidate_Gene_ID_v5"], l2)].append(edge)
    chrom_by_candidate_l1: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for edge in direct_chromatin_edges:
        chrom_by_candidate_l1[(edge["Candidate_Gene_ID_v5"], edge["Mechanism_L1_ID"])].append(edge)
    multi_by_candidate_l1: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for edge in direct_multilayer_edges:
        multi_by_candidate_l1[(edge["Candidate_Gene_ID_v5"], edge["Mechanism_L1_ID"])].append(edge)

    l1_scores: list[dict[str, object]] = []
    l2_support: list[dict[str, object]] = []
    for candidate in candidates:
        trait = candidate_info[candidate]["Trait"]
        ann = annotations[candidate]
        annotation_text = ";".join(clean(ann.get(col, "")) for col in function_cols if col != "Gene_ID_v5")
        tissue_rows = tissue.loc[tissue["Gene_ID_v5"] == candidate]
        node = node_lookup[candidate]
        for l1 in l1_ids:
            layer_values: dict[str, float | None] = {}
            layer_details = {}
            for layer in LAYER_COLUMNS:
                detail = l1_layer_results[(candidate, l1, layer)]
                layer_details[layer] = detail
                layer_values[layer] = detail["degree_adjusted_percentile"]
            scenario_scores = {
                scenario: weighted_available(layer_values, weights)
                for scenario, weights in SCENARIO_WEIGHTS.items()
            }
            network_primary = scenario_scores["EQUAL_LAYER_PRIMARY"]
            network_available_layers = sum(v is not None for v in layer_values.values())
            scenario_finite = [x for x in scenario_scores.values() if x is not None]
            network_min = min(scenario_finite) if scenario_finite else None
            network_max = max(scenario_finite) if scenario_finite else None

            tissue_result = tissue_support(tissue_rows, l1_meta[l1]["Relevant_tissue_tokens"])
            ann_score, matched_terms = annotation_support(annotation_text, l1_meta[l1]["Include_terms"])
            tr_score, tr_basis = trait_score(trait, l1_meta[l1]["Trait_relevance"])
            tf_edges = tf_by_candidate_l1[(candidate, l1)]
            tf_score = max((number(x.get("Confidence_score")) or 0.0) for x in tf_edges) * 100.0 if tf_edges else 0.0
            tf_available = tf_candidate_presence[candidate]
            annotation_available = clean(ann.get("Function_annotation_level", "")) != "N_IDENTITY_ONLY"
            tissue_available = clean(tissue_rows.iloc[0]["Candidate_tissue_evidence_QC"]) != "NO_UNAMBIGUOUS_EXPRESSION_MAPPING"

            components = {
                "network": network_primary * 100.0 if network_primary is not None else None,
                "tf_direct": tf_score if tf_available else None,
                "tissue": tissue_result["score"] if tissue_available else None,
                "annotation": ann_score if annotation_available else None,
                "trait": tr_score,
            }
            if network_primary is not None:
                composite = weighted_available(components, COMPONENT_WEIGHTS)
                prediction_status = "RANKED_L1_NETWORK_PROPAGATION"
            else:
                composite = None
                prediction_status = "DESCRIPTIVE_ONLY_NO_MULTILAYER_NETWORK"
            descriptive_components = {k: v for k, v in components.items() if k != "network"}
            descriptive_weights = {k: COMPONENT_WEIGHTS[k] for k in descriptive_components}
            nontrait_available = sum(
                components[k] is not None
                for k in ["tf_direct", "tissue", "annotation"]
            )
            descriptive_score = weighted_available(descriptive_components, descriptive_weights) if nontrait_available >= 2 else None
            if composite is None and descriptive_score is None:
                prediction_status = "UNRESOLVED_INSUFFICIENT_PUBLIC_DATA"

            top_contexts = tissue_result["top_contexts"]
            l1_scores.append({
                "Gene_ID_v5": candidate,
                "Gene_ID_v4": candidate_info[candidate]["Gene_ID_v4"],
                "Preferred_gene_symbol": clean(ann.get("Preferred_gene_symbol", "")),
                "Trait": trait,
                "Mechanism_L1_ID": l1,
                "Mechanism_L1": l1_meta[l1]["Mechanism_L1"],
                "Mechanism_L1_CN": l1_meta[l1]["Mechanism_L1_CN"],
                "Ontology_roles": ";".join(l1_meta[l1]["Ontology_roles"]),
                "Eligible_seed_genes": len(l1_seed_map[l1]),
                "Network_score_equal_layer_0_100": network_primary * 100.0 if network_primary is not None else None,
                "Network_score_sensitivity_min_0_100": network_min * 100.0 if network_min is not None else None,
                "Network_score_sensitivity_max_0_100": network_max * 100.0 if network_max is not None else None,
                "Network_score_sensitivity_range": (network_max - network_min) * 100.0 if network_min is not None else None,
                "Network_layers_assessable": network_available_layers,
                "CHIA_degree": layer_details["CHIA"]["degree"],
                "CHIA_degree_adjusted_RWR_percentile": layer_details["CHIA"]["degree_adjusted_percentile"],
                "RNA_degree": layer_details["RNA"]["degree"],
                "RNA_degree_adjusted_RWR_percentile": layer_details["RNA"]["degree_adjusted_percentile"],
                "RIBO_degree": layer_details["RIBO"]["degree"],
                "RIBO_degree_adjusted_RWR_percentile": layer_details["RIBO"]["degree_adjusted_percentile"],
                "PPI_degree": layer_details["PPI"]["degree"],
                "PPI_degree_adjusted_RWR_percentile": layer_details["PPI"]["degree_adjusted_percentile"],
                "Tissue_context_score_0_100": tissue_result["score"] if tissue_available else None,
                "Tissue_matched_contexts": tissue_result["matched_contexts"],
                "Tissue_assessable_contexts": tissue_result["assessable_contexts"],
                "Top_tissue_contexts": ";".join(x["Context_ID"] for x in top_contexts),
                "Top_tissue_context_details": ";".join(
                    f"{x['Context_ID']}|{x['Biological_tissue']}|{x['Developmental_stage']}|{x['Effective_percentile']:.6f}"
                    for x in top_contexts
                ),
                "TF_direct_score_0_100": tf_score if tf_available else None,
                "N_direct_TF_seed_edges": len(tf_edges),
                "Direct_TF_seed_genes": ";".join(sorted({str(x["Seed_Gene_ID_v5"]) for x in tf_edges})),
                "N_direct_chromatin_seed_edges": len(chrom_by_candidate_l1[(candidate, l1)]),
                "N_direct_multilayer_seed_edges": len(multi_by_candidate_l1[(candidate, l1)]),
                "Annotation_support_score_0_100": ann_score if annotation_available else None,
                "Annotation_matched_terms": ";".join(matched_terms),
                "Trait_compatibility_score_0_100": tr_score,
                "Trait_compatibility_basis": tr_basis,
                "Integrated_mechanism_support_score_0_100": composite,
                "Descriptive_support_score_noncomparable_0_100": descriptive_score,
                "Prediction_status": prediction_status,
                "Network_QC": clean(node.get("Node_QC", "")),
                "Tissue_QC": clean(tissue_rows.iloc[0]["Candidate_tissue_evidence_QC"]),
                "Function_annotation_QC": clean(ann.get("Function_annotation_QC", "")),
                "Score_interpretation": "Relative public-data support for an L1 mechanism hypothesis; not a causal probability.",
            })

        for l2 in l2_ids:
            meta = l2_meta[l2]
            tissue_result = tissue_support(tissue_rows, meta["Relevant_tissue_tokens"])
            ann_score, matched_terms = annotation_support(annotation_text, meta["Include_terms"])
            tr_score, tr_basis = trait_score(trait, meta["Trait_relevance"])
            tf_edges = tf_by_candidate_l2[(candidate, l2)]
            tf_score = max((number(x.get("Confidence_score")) or 0.0) for x in tf_edges) * 100.0 if tf_edges else 0.0
            l2_seed_rows = seeds.loc[(seeds["Mechanism_L2_ID"] == l2)]
            eligible_l2_count = l2_seed_rows.loc[l2_seed_rows["Seed_eligible_L2"] == "1", "Gene_ID_v5"].nunique()
            descriptive_values = {
                "tf_direct": tf_score if tf_candidate_presence[candidate] else None,
                "tissue": tissue_result["score"] if clean(tissue_rows.iloc[0]["Candidate_tissue_evidence_QC"]) != "NO_UNAMBIGUOUS_EXPRESSION_MAPPING" else None,
                "annotation": ann_score if clean(ann.get("Function_annotation_level", "")) != "N_IDENTITY_ONLY" else None,
                "trait": tr_score,
            }
            descriptive_weights = {k: COMPONENT_WEIGHTS[k] for k in descriptive_values}
            l2_nontrait_available = sum(
                descriptive_values[k] is not None
                for k in ["tf_direct", "tissue", "annotation"]
            )
            l2_descriptive = (
                weighted_available(descriptive_values, descriptive_weights)
                if l2_nontrait_available >= 2
                else None
            )
            l2_support.append({
                "Gene_ID_v5": candidate,
                "Preferred_gene_symbol": clean(ann.get("Preferred_gene_symbol", "")),
                "Trait": trait,
                "Mechanism_L1_ID": meta["Mechanism_L1_ID"],
                "Mechanism_L1_CN": meta["Mechanism_L1_CN"],
                "Mechanism_L2_ID": l2,
                "Mechanism_L2": meta["Mechanism_L2"],
                "Mechanism_L2_CN": meta["Mechanism_L2_CN"],
                "Ontology_role": meta["Ontology_role"],
                "Annotated_seed_rows": len(l2_seed_rows),
                "Eligible_L2_seed_genes": eligible_l2_count,
                "Minimum_primary_seed_count": meta["Min_primary_seed_count"],
                "L2_inference_status": "DESCRIPTIVE_ONLY_PRIMARY_RWR_IS_L1",
                "Tissue_context_score_0_100": descriptive_values["tissue"],
                "Top_tissue_contexts": ";".join(x["Context_ID"] for x in tissue_result["top_contexts"]),
                "TF_direct_score_0_100": descriptive_values["tf_direct"],
                "N_direct_TF_seed_edges": len(tf_edges),
                "Direct_TF_seed_genes": ";".join(sorted({str(x["Seed_Gene_ID_v5"]) for x in tf_edges})),
                "Annotation_support_score_0_100": descriptive_values["annotation"],
                "Annotation_matched_terms": ";".join(matched_terms),
                "Trait_compatibility_score_0_100": tr_score,
                "Trait_compatibility_basis": tr_basis,
                "Descriptive_support_score_noncomparable_0_100": l2_descriptive,
                "Sparse_seed_action": meta["Sparse_seed_action"],
                "Interpretation_limit": "L2 is a descriptive refinement only; no L2 RWR score is reported because the ontology prespecifies L1 as the primary propagation level.",
            })

    # Rank L1 mechanisms within each gene and summarize sensitivity.
    for candidate in candidates:
        rows = [x for x in l1_scores if x["Gene_ID_v5"] == candidate]
        use_integrated = any(x["Integrated_mechanism_support_score_0_100"] is not None for x in rows)
        key = "Integrated_mechanism_support_score_0_100" if use_integrated else "Descriptive_support_score_noncomparable_0_100"
        rows.sort(key=lambda x: (-(x[key] if x[key] is not None else -1.0), x["Mechanism_L1_ID"]))
        for rank, row in enumerate(rows, start=1):
            row["Within_gene_L1_rank"] = rank
            row["Ranking_basis"] = key

    # Sensitivity top-mechanism stability recomputes the integrated score after
    # replacing only the layer-weighted network component.  This keeps the
    # comparison aligned with the reported integrated-mechanism ranking.
    scenario_tops: dict[str, dict[str, str]] = {candidate: {} for candidate in candidates}
    for candidate in candidates:
        for scenario, weights in SCENARIO_WEIGHTS.items():
            scored = []
            for l1 in l1_ids:
                vals = {layer: l1_layer_results[(candidate, l1, layer)]["degree_adjusted_percentile"] for layer in LAYER_COLUMNS}
                network_score = weighted_available(vals, weights)
                reported = next(
                    x for x in l1_scores
                    if x["Gene_ID_v5"] == candidate and x["Mechanism_L1_ID"] == l1
                )
                if network_score is not None:
                    components = {
                        "network": network_score * 100.0,
                        "tf_direct": reported["TF_direct_score_0_100"],
                        "tissue": reported["Tissue_context_score_0_100"],
                        "annotation": reported["Annotation_support_score_0_100"],
                        "trait": reported["Trait_compatibility_score_0_100"],
                    }
                    score = weighted_available(components, COMPONENT_WEIGHTS)
                    if score is not None:
                        scored.append((score, l1))
            if scored:
                scenario_tops[candidate][scenario] = sorted(scored, key=lambda x: (-x[0], x[1]))[0][1]

    path_lookup = {(x["Candidate_Gene_ID_v5"], x["Mechanism_L1_ID"]): x for x in path_records}
    summaries: list[dict[str, object]] = []
    for candidate in candidates:
        ranked = sorted(
            [x for x in l1_scores if x["Gene_ID_v5"] == candidate],
            key=lambda x: x["Within_gene_L1_rank"],
        )
        top = ranked[0]
        runner = ranked[1]
        unresolved = top["Prediction_status"] == "UNRESOLVED_INSUFFICIENT_PUBLIC_DATA"
        ranking_key = top["Ranking_basis"]
        top_score = top[ranking_key]
        runner_score = runner[ranking_key]
        margin = (top_score - runner_score) if top_score is not None and runner_score is not None else None
        scenario_values = list(scenario_tops[candidate].values())
        stability = (
            scenario_values.count(top["Mechanism_L1_ID"]) / len(scenario_values)
            if scenario_values and not unresolved
            else None
        )
        path = path_lookup[(candidate, top["Mechanism_L1_ID"])]
        independent_supports = sum([
            (top["N_direct_TF_seed_edges"] or 0) > 0,
            (top["Tissue_context_score_0_100"] or 0) >= 75,
            (top["Annotation_support_score_0_100"] or 0) > 0,
            (top["N_direct_chromatin_seed_edges"] or 0) > 0,
        ])
        if top["Prediction_status"] == "UNRESOLVED_INSUFFICIENT_PUBLIC_DATA":
            tier = "UNRESOLVED"
        elif top["Prediction_status"] != "RANKED_L1_NETWORK_PROPAGATION":
            tier = "DESCRIPTIVE_ONLY"
        elif (
            (top["Network_score_equal_layer_0_100"] or 0) >= 80
            and top["Network_layers_assessable"] >= 2
            and independent_supports >= 1
            and (top["Network_score_sensitivity_range"] or 0) <= 15
        ):
            tier = "EVIDENCE_TIER_A"
        elif (
            (top["Network_score_equal_layer_0_100"] or 0) >= 65
            and top["Network_layers_assessable"] >= 1
            and independent_supports >= 1
        ):
            tier = "EVIDENCE_TIER_B"
        else:
            tier = "EVIDENCE_TIER_C"

        l2_rows = [
            x for x in l2_support
            if x["Gene_ID_v5"] == candidate
            and x["Mechanism_L1_ID"] == top["Mechanism_L1_ID"]
            and x["Descriptive_support_score_noncomparable_0_100"] is not None
        ]
        l2_rows.sort(key=lambda x: (-(x["Descriptive_support_score_noncomparable_0_100"] or -1.0), x["Mechanism_L2_ID"]))
        top_l2 = l2_rows[0] if l2_rows and not unresolved else None
        support_phrases = []
        if top["N_direct_TF_seed_edges"]:
            support_phrases.append(f"{top['N_direct_TF_seed_edges']}条种子TF有向边")
        if (top["Tissue_context_score_0_100"] or 0) >= 75:
            support_phrases.append("相关组织高表达背景")
        if top["Annotation_matched_terms"]:
            support_phrases.append(f"功能关键词{top['Annotation_matched_terms']}")
        if top["N_direct_chromatin_seed_edges"]:
            support_phrases.append(f"{top['N_direct_chromatin_seed_edges']}条直接染色质接触")
        if not support_phrases:
            support_phrases.append("暂无强独立支持层")
        if top["Prediction_status"] == "RANKED_L1_NETWORK_PROPAGATION":
            narrative = (
                f"公共多层网络优先支持{top['Mechanism_L1_CN']}（综合支持分{top_score:.2f}，"
                f"等权网络分{top['Network_score_equal_layer_0_100']:.2f}，可评估网络层{top['Network_layers_assessable']}）；"
                f"独立支持包括{'、'.join(support_phrases)}。"
            )
        elif top["Prediction_status"] == "DESCRIPTIVE_ONLY_NO_MULTILAYER_NETWORK":
            narrative = (
                f"因候选基因缺少可用多层网络连接，仅描述性提示{top['Mechanism_L1_CN']}；"
                f"依据为{'、'.join(support_phrases)}，不能与网络可评估基因的综合分直接比较。"
            )
        else:
            narrative = "公共网络、表达与功能注释不足，当前不能可靠提出机制类别。"
        summaries.append({
            "Gene_ID_v5": candidate,
            "Gene_ID_v4": candidate_info[candidate]["Gene_ID_v4"],
            "Preferred_gene_symbol": top["Preferred_gene_symbol"],
            "Trait": candidate_info[candidate]["Trait"],
            "Prediction_status": top["Prediction_status"],
            "Evidence_tier": tier,
            "Top_Mechanism_L1_ID": "" if unresolved else top["Mechanism_L1_ID"],
            "Top_Mechanism_L1_CN": "" if unresolved else top["Mechanism_L1_CN"],
            "Top_integrated_score_0_100": None if unresolved else top["Integrated_mechanism_support_score_0_100"],
            "Top_descriptive_score_noncomparable_0_100": None if unresolved else top["Descriptive_support_score_noncomparable_0_100"],
            "Runner_up_Mechanism_L1_ID": "" if unresolved else runner["Mechanism_L1_ID"],
            "Runner_up_Mechanism_L1_CN": "" if unresolved else runner["Mechanism_L1_CN"],
            "Top_vs_runner_up_margin": None if unresolved else margin,
            "Network_layer_weight_sensitivity_top_stability": stability,
            "Network_layers_assessable": top["Network_layers_assessable"],
            "Top_network_score_0_100": top["Network_score_equal_layer_0_100"],
            "Top_tissue_context_score_0_100": None if unresolved else top["Tissue_context_score_0_100"],
            "Top_TF_direct_score_0_100": None if unresolved else top["TF_direct_score_0_100"],
            "Top_annotation_support_score_0_100": None if unresolved else top["Annotation_support_score_0_100"],
            "Top_trait_compatibility_score_0_100": None if unresolved else top["Trait_compatibility_score_0_100"],
            "Top_L2_descriptive_ID": top_l2["Mechanism_L2_ID"] if top_l2 else "",
            "Top_L2_descriptive_CN": top_l2["Mechanism_L2_CN"] if top_l2 else "",
            "Top_L2_descriptive_score_noncomparable_0_100": top_l2["Descriptive_support_score_noncomparable_0_100"] if top_l2 else None,
            "Shortest_path_to_top_L1_seed_edges": None if unresolved else path["Shortest_path_distance_edges"],
            "Shortest_path_to_top_L1_seed": "" if unresolved else path["Path_node_sequence"],
            "Top_tissue_contexts": "" if unresolved else top["Top_tissue_contexts"],
            "Direct_TF_seed_genes": "" if unresolved else top["Direct_TF_seed_genes"],
            "Annotation_matched_terms": "" if unresolved else top["Annotation_matched_terms"],
            "Mechanism_hypothesis_CN": narrative,
            "Interpretation_limit": "Mechanism hypothesis generated from public network, expression, directed regulation and annotation evidence; it is not experimental validation and contains no NAM genetic evidence.",
        })

    # Input manifest is hashed only after successful analysis so every reported input is auditable.
    input_manifest = []
    for role, filename in FILES.items():
        path = INPUT_DIR / filename
        input_manifest.append({
            "Input_role": role,
            "File_name": filename,
            "Bytes": path.stat().st_size,
            "SHA256": sha256_file(path),
        })

    method = {
        "Analysis_version": VERSION,
        "Primary_inference_level": "L1",
        "L2_policy": "DESCRIPTIVE_ONLY; no L2 RWR because all ontology rows specify Primary_RWR_level=L1 and sparse classes are not comparable",
        "RWR_formula": "p(t+1)=0.5*s+0.5*P_transpose*p(t); each layer row-normalized separately",
        "RWR_restart_probability": RWR_RESTART,
        "RWR_tolerance_L1": RWR_TOL,
        "RWR_max_iterations": RWR_MAX_ITER,
        "Network_score": "Degree-bin-adjusted empirical midrank RWR percentiles combined across assessable layers",
        "Degree_bins": "floor(log2(layer_degree)); zero RWR caused by no seed reachability receives percentile 0",
        "Primary_layer_weights": SCENARIO_WEIGHTS["EQUAL_LAYER_PRIMARY"],
        "Sensitivity_layer_weights": {k: v for k, v in SCENARIO_WEIGHTS.items() if k != "EQUAL_LAYER_PRIMARY"},
        "Integrated_score_weights": COMPONENT_WEIGHTS,
        "Missing_component_rule": "Renormalize declared weights over assessable components; integrated score is not reported when multilayer network propagation is unavailable",
        "Tissue_score": "Mean of top three mechanism-matched RNA/Ribo mean-percentiles after CORE=1.0, SUPPORTING=0.75, DOWNSTREAM=0.5 context weighting",
        "TF_direct_score": "Maximum public TF-target confidence score for a direct edge between candidate and a mechanism seed",
        "Annotation_score": "100/3 points per exact normalized ontology include-term match, capped at 100",
        "Chromatin_policy": "Reported as direct support but not added again because ChIA-PET is already a propagation layer",
        "Shortest_path_policy": f"Unweighted explanatory paths capped at {PATH_MAX_DEPTH} edges; not causal paths",
        "NAM_evidence_used": False,
        "Interpretation": "Scores are relative evidence-prioritization measures, not causal or posterior probabilities.",
    }

    direct_edges = direct_multilayer_edges + direct_tf_edges + direct_chromatin_edges
    result = {
        "method": method,
        "summary": summaries,
        "l1_scores": l1_scores,
        "l2_support": l2_support,
        "paths": path_records,
        "direct_edges": direct_edges,
        "layer_qc": layer_qc,
        "input_manifest": input_manifest,
        "ontology": ontology.to_dict("records"),
        "seed_qc": [
            {
                "Mechanism_L1_ID": l1,
                "Mechanism_L1_CN": l1_meta[l1]["Mechanism_L1_CN"],
                "Eligible_seed_genes": len(l1_seed_map[l1]),
                "Seed_weight_sum": sum(l1_seed_map[l1].values()),
            }
            for l1 in l1_ids
        ],
        "validation": {
            "Candidate_count": len(candidates),
            "L1_score_rows": len(l1_scores),
            "L2_support_rows": len(l2_support),
            "Path_rows": len(path_records),
            "Direct_evidence_rows": len(direct_edges),
            "Expected_L1_rows": len(candidates) * len(l1_ids),
            "Expected_L2_rows": len(candidates) * len(l2_ids),
            "No_NAM_evidence": True,
        },
    }

    if len(l1_scores) != 54 or len(l2_support) != 207 or len(path_records) != 54:
        raise ValueError("Unexpected result dimensions")
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_json": str(OUTPUT_JSON),
        "summary_rows": len(summaries),
        "l1_rows": len(l1_scores),
        "l2_rows": len(l2_support),
        "path_rows": len(path_records),
        "direct_edges": len(direct_edges),
        "layer_qc_rows": len(layer_qc),
        "candidate_status": {x["Gene_ID_v5"]: x["Prediction_status"] for x in summaries},
        "candidate_top_mechanism": {x["Gene_ID_v5"]: x["Top_Mechanism_L1_ID"] for x in summaries},
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer candidate-gene flowering-time mechanism hypotheses from prepared public-data inputs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing the 11 prepared mechanism-input files.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Path for the raw inference JSON.")
    parser.add_argument("--rwr-restart", type=float, default=RWR_RESTART, help="Random-walk restart probability.")
    parser.add_argument("--rwr-tolerance", type=float, default=RWR_TOL, help="L1 convergence tolerance.")
    parser.add_argument("--rwr-max-iterations", type=int, default=RWR_MAX_ITER, help="Maximum RWR iterations.")
    parser.add_argument("--path-max-depth", type=int, default=PATH_MAX_DEPTH, help="Maximum explanatory path depth.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0 < args.rwr_restart < 1:
        raise ValueError("--rwr-restart must be between 0 and 1")
    if args.rwr_tolerance <= 0 or args.rwr_max_iterations <= 0 or args.path_max_depth <= 0:
        raise ValueError("Tolerance, iteration count, and path depth must be positive")
    INPUT_DIR = args.input_dir.resolve()
    OUTPUT_JSON = args.output_json.resolve()
    WORK_DIR = OUTPUT_JSON.parent
    RWR_RESTART = args.rwr_restart
    RWR_TOL = args.rwr_tolerance
    RWR_MAX_ITER = args.rwr_max_iterations
    PATH_MAX_DEPTH = args.path_max_depth
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    main()
