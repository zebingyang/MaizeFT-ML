#!/usr/bin/env python3
"""Create publication-ready figures from the standardized mechanism JSON.

The figures deliberately separate network-ranked, descriptive-only, and unresolved
candidates. Mechanism scores are visualized within candidates and are not presented
as causal probabilities or as a cross-gene causal ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


MECHANISMS = ["M1", "M2", "M3", "M4", "M5", "M6"]
MECHANISM_LABELS = {
    "M1": "Environment / light",
    "M2": "Florigen / transition",
    "M3": "Age / autonomous",
    "M4": "Hormone / stress",
    "M5": "Meristem / reproduction",
    "M6": "Chromatin / RNA",
}
MECHANISM_COLORS = {
    "M1": "#4477AA",
    "M2": "#66CCEE",
    "M3": "#228833",
    "M4": "#CCBB44",
    "M5": "#EE6677",
    "M6": "#AA3377",
}
TRAIT_COLORS = {"DTA": "#2F6B9A", "DTS": "#D87520", "ASI": "#7A4E9D"}
LAYER_COLORS = {"CHIA": "#7A5195", "RNA": "#2F6B9A", "RIBO": "#E07A2D", "PPI": "#3B8F5A"}
COMPONENT_ORDER = ["network", "tf_directed", "tissue", "annotation", "trait"]
COMPONENT_LABELS = {
    "network": "Network",
    "tf_directed": "TF-directed",
    "tissue": "Tissue",
    "annotation": "Annotation",
    "trait": "Trait match",
}


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def gene_label(row: dict[str, object]) -> str:
    symbol = str(row.get("preferred_gene_symbol") or "").strip()
    return f"{row['gene_id_v5']} ({symbol})" if symbol else str(row["gene_id_v5"])


def save_figure(fig: plt.Figure, stem: Path, formats: list[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        extension = extension.lower().lstrip(".")
        kwargs = {"dpi": dpi} if extension in {"png", "tif", "tiff"} else {}
        fig.savefig(stem.with_suffix(f".{extension}"), **kwargs)


def score_lookup(payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(row["gene_id_v5"]), str(row["mechanism_l1_id"])): row
        for row in payload["l1_scores_long"]
    }


def draw_heatmap(
    ax: plt.Axes,
    summaries: list[dict[str, object]],
    lookup: dict[tuple[str, str], dict[str, object]],
    title: str,
    panel: str,
    cmap: LinearSegmentedColormap,
) -> None:
    matrix = np.full((len(summaries), len(MECHANISMS)), np.nan)
    for i, summary in enumerate(summaries):
        gene = str(summary["gene_id_v5"])
        for j, mechanism in enumerate(MECHANISMS):
            row = lookup.get((gene, mechanism), {})
            value = finite(row.get("primary_display_score_0_100"))
            if value is not None:
                matrix[i, j] = value
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    ax.set_xticks(range(6), [MECHANISM_LABELS[m] for m in MECHANISMS], rotation=35, ha="right")
    ax.set_yticks(range(len(summaries)), [gene_label(x) for x in summaries])
    for i, summary in enumerate(summaries):
        top = str(summary.get("top_mechanism_l1_id") or "")
        if summary.get("prediction_status") == "UNRESOLVED":
            ax.text(2.5, i, "Insufficient public data", ha="center", va="center",
                    fontsize=7.2, color="#666666", fontstyle="italic")
        for j, mechanism in enumerate(MECHANISMS):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.8,
                        color="white" if value >= 58 else "#222222")
            if mechanism == top:
                ax.add_patch(plt.Rectangle((j - 0.49, i - 0.49), 0.98, 0.98,
                                           fill=False, edgecolor="#111111", linewidth=1.4))
    ax.set_title(f"{panel}  {title}", loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def figure_landscape(payload: dict[str, object], output_dir: Path, formats: list[str], dpi: int) -> None:
    summaries = sorted(payload["candidate_summary"], key=lambda x: int(x["display_order"]))
    ranked = [x for x in summaries if x["prediction_status"] == "NETWORK_RANKED"]
    descriptive = [x for x in summaries if x["prediction_status"] == "DESCRIPTIVE_ONLY"]
    unresolved = [x for x in summaries if x["prediction_status"] == "UNRESOLVED"]
    lookup = score_lookup(payload)
    component_lookup: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in payload["top_components_long"]:
        component_lookup[str(row["gene_id_v5"])][str(row["evidence_component"])] = row

    fig = plt.figure(figsize=(14.5, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.45, 1.0], hspace=0.72, wspace=0.72)
    cmap_ranked = LinearSegmentedColormap.from_list("ranked", ["#F7FBFF", "#6BAED6", "#08306B"])
    cmap_desc = LinearSegmentedColormap.from_list("descriptive", ["#FFF8E7", "#F3C36B", "#8C510A"])
    ax_a = fig.add_subplot(grid[0, 0])
    im = draw_heatmap(ax_a, ranked, lookup, "Network-ranked candidates", "A", cmap_ranked)
    colorbar = fig.colorbar(im, ax=ax_a, fraction=0.025, pad=0.025)
    colorbar.set_label("Integrated support (0–100)")

    ax_b = fig.add_subplot(grid[0, 1])
    secondary_panel = descriptive + unresolved
    if secondary_panel:
        im2 = draw_heatmap(ax_b, secondary_panel, lookup, "Descriptive-only and unresolved", "B", cmap_desc)
        colorbar2 = fig.colorbar(im2, ax=ax_b, fraction=0.045, pad=0.035)
        colorbar2.set_label("Descriptive support (0–100)")
    else:
        ax_b.axis("off")
        ax_b.set_title("B  No descriptive-only candidates", loc="left")
    ax_c = fig.add_subplot(grid[1, :])
    y = np.arange(len(summaries))
    offsets = np.linspace(-0.26, 0.26, len(COMPONENT_ORDER))
    markers = ["o", "s", "D", "^", "P"]
    colors = ["#3E6D9C", "#B04B55", "#4F8A62", "#9A6FB0", "#D18B31"]
    for offset, marker, color, component in zip(offsets, markers, colors, COMPONENT_ORDER):
        xs, ys = [], []
        missing_y = []
        for i, summary in enumerate(summaries):
            row = component_lookup.get(str(summary["gene_id_v5"]), {}).get(component, {})
            value = finite(row.get("component_score_0_100"))
            if value is None:
                missing_y.append(i + offset)
            else:
                xs.append(value)
                ys.append(i + offset)
        ax_c.scatter(xs, ys, s=28, marker=marker, color=color, edgecolor="white", linewidth=0.4,
                     label=COMPONENT_LABELS[component], zorder=3)
        if missing_y:
            ax_c.scatter([2] * len(missing_y), missing_y, marker="x", s=22, color="#888888", linewidth=0.8)
    ax_c.set_yticks(y, [gene_label(x) for x in summaries])
    ax_c.set_xlim(-2, 102)
    ax_c.set_xlabel("Evidence component score (0–100); × = unavailable")
    ax_c.set_title("C  Evidence components supporting each candidate’s top hypothesis", loc="left")
    ax_c.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    ax_c.spines[["top", "right", "left"]].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    ax_c.invert_yaxis()
    ax_c.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)

    fig.suptitle("Public-data evidence supports candidate-specific flowering-time mechanism hypotheses",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.012,
             "Scores prioritize mechanisms within a gene. They are not causal probabilities and should not be used to rank causal genes across candidates.",
             fontsize=7.5, color="#444444")
    save_figure(fig, output_dir / "Fig_mechanism_landscape", formats, dpi)
    plt.close(fig)


def path_nodes(path: str) -> list[str]:
    return [x for x in str(path or "").split(">") if x]


def path_layers(path: str) -> list[str]:
    return [x for x in str(path or "").split(">") if x]


def draw_path(ax: plt.Axes, summary: dict[str, object], path: dict[str, object], panel: str) -> None:
    nodes = path_nodes(str(path.get("path_node_sequence") or ""))
    layers = path_layers(str(path.get("path_edge_layer_sequence") or ""))
    gene = str(summary["gene_id_v5"])
    if len(nodes) < 2:
        ax.text(0.5, 0.5, "No network path available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x = np.arange(len(nodes), dtype=float)
    y = np.zeros(len(nodes))
    for i in range(len(nodes) - 1):
        layer = layers[i] if i < len(layers) else "UNKNOWN"
        ax.plot(x[i:i + 2], y[i:i + 2], color=LAYER_COLORS.get(layer, "#777777"), linewidth=3.0, zorder=1)
        ax.text((x[i] + x[i + 1]) / 2, 0.13, layer, ha="center", va="bottom", fontsize=7,
                color=LAYER_COLORS.get(layer, "#555555"))
    for i, node in enumerate(nodes):
        if i == 0:
            face, edge, label = TRAIT_COLORS.get(str(summary["trait"]), "#444444"), "#222222", f"Candidate\n{node}"
        elif i == len(nodes) - 1:
            face, edge = MECHANISM_COLORS.get(str(summary["top_mechanism_l1_id"]), "#AA3377"), "#222222"
            symbol = str(path.get("terminal_seed_symbol") or "")
            label = f"Mechanism seed\n{node}" + (f"\n({symbol})" if symbol else "")
        else:
            face, edge, label = "#EEEEEE", "#777777", f"Network node\n{node}"
        ax.scatter(x[i], y[i], s=300, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=2)
        ax.text(x[i], -0.17, label, ha="center", va="top", fontsize=6.7)
    margin = finite(summary.get("top_vs_runner_up_margin"))
    subtitle = (
        f"{summary['trait']} | {summary.get('top_mechanism_l1_id', 'NA')}: {summary.get('top_mechanism_l1_en', '')} | "
        f"margin={margin:.2f}" if margin is not None else
        f"{summary['trait']} | {summary.get('top_mechanism_l1_id', 'NA')}: {summary.get('top_mechanism_l1_en', '')}"
    )
    ax.set_title(f"{panel}  {gene}\n{subtitle}", loc="left", fontsize=9)
    ax.set_xlim(-0.35, len(nodes) - 0.65)
    ax.set_ylim(-0.65, 0.5)
    ax.axis("off")


def figure_selected_paths(
    payload: dict[str, object], output_dir: Path, formats: list[str], dpi: int, selected_genes: list[str]
) -> None:
    summary_by_gene = {str(x["gene_id_v5"]): x for x in payload["candidate_summary"]}
    path_by_gene = {str(x["gene_id_v5"]): x for x in payload["top_evidence_paths"]}
    selected = [g for g in selected_genes if g in summary_by_gene]
    if not selected:
        selected = [str(x["gene_id_v5"]) for x in payload["candidate_summary"] if x["prediction_status"] == "NETWORK_RANKED"][:3]
    fig, axes = plt.subplots(len(selected), 1, figsize=(11.5, 2.45 * len(selected)), squeeze=False)
    for i, gene in enumerate(selected):
        draw_path(axes[i, 0], summary_by_gene[gene], path_by_gene.get(gene, {}), chr(65 + i))
    legend = [Line2D([0], [0], color=color, lw=3, label=layer) for layer, color in LAYER_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.052), ncol=4, frameon=False)
    fig.suptitle("Shortest public-network association paths to flowering-time mechanism seeds",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.014,
             "Association paths are undirected explanatory links, not regulatory cascades or proof of causality.",
             ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=[0, 0.11, 1, 0.95])
    save_figure(fig, output_dir / "Fig_selected_candidate_evidence_paths", formats, dpi)
    plt.close(fig)


def draw_profile(
    fig: plt.Figure,
    summary: dict[str, object],
    l1_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    path: dict[str, object],
) -> None:
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.88], width_ratios=[1.15, 1.0], hspace=0.48, wspace=0.42)
    ax_score = fig.add_subplot(grid[:, 0])
    ordered = sorted(l1_rows, key=lambda x: MECHANISMS.index(str(x["mechanism_l1_id"])))
    values = [finite(x.get("primary_display_score_0_100")) for x in ordered]
    if any(v is not None for v in values):
        colors = [MECHANISM_COLORS[str(x["mechanism_l1_id"])] if str(x["mechanism_l1_id"]) == summary.get("top_mechanism_l1_id") else "#C8CCD0" for x in ordered]
        bars = ax_score.barh(np.arange(6), [0 if v is None else v for v in values], color=colors, height=0.65)
        for bar, value in zip(bars, values):
            if value is not None:
                ax_score.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=7)
        ax_score.set_yticks(np.arange(6), [f"{m}  {MECHANISM_LABELS[m]}" for m in MECHANISMS])
        ax_score.set_xlim(0, 106)
        ax_score.set_xlabel("Within-gene mechanism support (0–100)")
        ax_score.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        ax_score.invert_yaxis()
    else:
        ax_score.text(0.5, 0.5, "Unresolved owing to insufficient public data", ha="center", va="center", transform=ax_score.transAxes)
        ax_score.set_axis_off()
    ax_score.set_title("A  L1 mechanism profile", loc="left")
    ax_score.spines[["top", "right", "left"]].set_visible(False)
    ax_score.tick_params(axis="y", length=0)

    ax_comp = fig.add_subplot(grid[0, 1])
    rows = {str(x["evidence_component"]): x for x in component_rows}
    comp_values = [finite(rows.get(c, {}).get("component_score_0_100")) for c in COMPONENT_ORDER]
    bars = ax_comp.barh(np.arange(5), [0 if v is None else v for v in comp_values], color="#6C8EBF", height=0.62)
    for i, (bar, value) in enumerate(zip(bars, comp_values)):
        if value is None:
            ax_comp.text(2, i, "NA", va="center", ha="left", color="#777777", fontsize=7)
        else:
            ax_comp.text(value + 1, i, f"{value:.1f}", va="center", fontsize=7)
    ax_comp.set_yticks(np.arange(5), [COMPONENT_LABELS[x] for x in COMPONENT_ORDER])
    ax_comp.set_xlim(0, 106)
    ax_comp.invert_yaxis()
    ax_comp.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    ax_comp.set_title("B  Top-hypothesis evidence", loc="left")
    ax_comp.spines[["top", "right", "left"]].set_visible(False)
    ax_comp.tick_params(axis="y", length=0)

    ax_text = fig.add_subplot(grid[1, 1])
    distance = path.get("shortest_path_distance_edges")
    terminal = str(path.get("terminal_seed_symbol") or path.get("terminal_seed_gene_id_v5") or "NA")
    score = summary.get("primary_score_0_100")
    margin = summary.get("top_vs_runner_up_margin")
    lines = [
        f"Status: {summary.get('prediction_status', '')}",
        f"Evidence tier: {summary.get('evidence_tier', '')}",
        f"Top hypothesis: {summary.get('top_mechanism_l1_id', 'NA')} — {summary.get('top_mechanism_l1_en', '')}",
        f"Primary score: {score if score is not None else 'NA'}",
        f"Top-vs-runner-up margin: {margin if margin is not None else 'NA'}",
        f"Network layers: {summary.get('network_layers_assessable', '') or 'none'}",
        f"Nearest top-mechanism seed: {terminal} ({distance if distance is not None else 'NA'} edges)",
        "Interpretation: public-data hypothesis; no NAM evidence included.",
    ]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=7.5, linespacing=1.45)
    ax_text.set_title("C  Evidence summary", loc="left")
    ax_text.axis("off")

    fig.suptitle(f"{gene_label(summary)} | {summary['trait']} candidate mechanism profile",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=TRAIT_COLORS.get(str(summary["trait"]), "#222222"))


def figure_profiles(payload: dict[str, object], output_dir: Path, formats: list[str], dpi: int) -> None:
    summaries = sorted(payload["candidate_summary"], key=lambda x: int(x["display_order"]))
    l1_by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
    comp_by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in payload["l1_scores_long"]:
        l1_by_gene[str(row["gene_id_v5"])].append(row)
    for row in payload["top_components_long"]:
        comp_by_gene[str(row["gene_id_v5"])].append(row)
    path_by_gene = {str(x["gene_id_v5"]): x for x in payload["top_evidence_paths"]}

    profile_dir = output_dir / "candidate_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        gene = str(summary["gene_id_v5"])
        fig = plt.figure(figsize=(9.1, 5.8))
        draw_profile(fig, summary, l1_by_gene[gene], comp_by_gene[gene], path_by_gene.get(gene, {}))
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.10, top=0.86, wspace=0.48, hspace=0.54)
        save_figure(fig, profile_dir / f"{safe_name(gene)}_mechanism_profile", formats, dpi)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11.4, 9.0), squeeze=False)
    for ax, summary in zip(axes.flat, summaries):
        gene = str(summary["gene_id_v5"])
        rows = sorted(l1_by_gene[gene], key=lambda x: MECHANISMS.index(str(x["mechanism_l1_id"])))
        values = [finite(x.get("primary_display_score_0_100")) for x in rows]
        if any(v is not None for v in values):
            colors = [MECHANISM_COLORS[m] if m == summary.get("top_mechanism_l1_id") else "#C7C9CC" for m in MECHANISMS]
            ax.barh(range(6), [0 if v is None else v for v in values], color=colors, height=0.63)
            ax.set_yticks(range(6), MECHANISMS)
            ax.set_xlim(0, 100)
            ax.grid(axis="x", color="#E5E5E5", linewidth=0.5)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "Unresolved", ha="center", va="center", transform=ax.transAxes, color="#777777")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"{gene}\n{summary['trait']} | {summary.get('top_mechanism_l1_id') or 'NA'}",
                     fontsize=8, color=TRAIT_COLORS.get(str(summary["trait"]), "#222222"))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes.flat[len(summaries):]:
        ax.axis("off")
    fig.suptitle("Candidate-specific L1 mechanism profiles", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.008,
             "Colored bars mark the top mechanism within each candidate; grey bars are alternatives. Do not compare candidates as causal ranks.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    save_figure(fig, output_dir / "Fig_candidate_mechanism_profiles_overview", formats, dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot standardized candidate-gene mechanism results.")
    parser.add_argument("--input-json", type=Path, required=True, help="Standardized mechanism-results JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory receiving publication figures.")
    parser.add_argument("--formats", default="png,pdf,svg,tiff", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=400, help="Raster output resolution.")
    parser.add_argument(
        "--selected-genes",
        default="Zm00001eb349860,Zm00001eb325180,Zm00001eb341290",
        help="Comma-separated genes for the selected-candidate path figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    selected = [x.strip() for x in args.selected_genes.split(",") if x.strip()]
    setup_style()
    figure_landscape(payload, args.output_dir, formats, args.dpi)
    figure_selected_paths(payload, args.output_dir, formats, args.dpi, selected)
    figure_profiles(payload, args.output_dir, formats, args.dpi)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "formats": formats, "selected_genes": selected}, indent=2))


if __name__ == "__main__":
    main()
