#!/usr/bin/env python3
"""Export standardized mechanism results as a publication-ready Excel workbook."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


SHEETS = [
    ("Candidate_Summary", "candidate_summary"),
    ("L1_Mechanism_Scores", "l1_scores_long"),
    ("Evidence_Components", "top_components_long"),
    ("Evidence_Paths", "top_evidence_paths"),
    ("Direct_Edges", "direct_edges"),
    ("Tissue_Contexts", "tissue_contexts"),
    ("Coverage_QC", "coverage_qc"),
    ("Mechanism_Ontology", "mechanism_ontology_l1"),
    ("Data_Dictionary", "data_dictionary"),
    ("Input_Manifest", "input_manifest"),
]

HEADER_FILL = PatternFill("solid", fgColor="17365D")
SUBHEADER_FILL = PatternFill("solid", fgColor="D9EAF7")
WARNING_FILL = PatternFill("solid", fgColor="FFF2CC")
WHITE_FONT = Font(color="FFFFFF", bold=True)
THIN_GREY = Side(style="thin", color="D9D9D9")


def stringify(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def ordered_headers(rows: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers


def add_table_sheet(workbook: Workbook, name: str, rows: list[dict[str, object]]) -> None:
    ws = workbook.create_sheet(name)
    if not rows:
        ws.append(["No records"])
        return
    headers = ordered_headers(rows)
    ws.append(headers)
    for row in rows:
        ws.append([stringify(row.get(header)) for header in headers])
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = WHITE_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    table_name = "T_" + "".join(x for x in name if x.isalnum())[:220]
    table = Table(displayName=table_name, ref=ws.dimensions)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showFirstColumn=False, showLastColumn=False)
    ws.add_table(table)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
    for column_index, header in enumerate(headers, start=1):
        values = [str(header)] + [str(row.get(header, "") or "") for row in rows[:250]]
        width = min(max(10, max(len(value) for value in values) + 2), 52)
        if "interpretation" in header.lower() or "description" in header.lower() or "sequence" in header.lower():
            width = 46
        ws.column_dimensions[get_column_letter(column_index)].width = width
    ws.row_dimensions[1].height = 34
    if name == "L1_Mechanism_Scores":
        for target in ("primary_display_score_0_100", "network_score_equal_layer_0_100"):
            if target in headers and ws.max_row > 2:
                column = get_column_letter(headers.index(target) + 1)
                ws.conditional_formatting.add(
                    f"{column}2:{column}{ws.max_row}",
                    ColorScaleRule(start_type="num", start_value=0, start_color="F7FBFF",
                                   mid_type="num", mid_value=50, mid_color="6BAED6",
                                   end_type="num", end_value=100, end_color="08306B"),
                )


def add_readme(workbook: Workbook, payload: dict[str, object]) -> None:
    ws = workbook.create_sheet("README", 0)
    ws["A1"] = "Candidate-gene flowering-time mechanism prediction"
    ws["A1"].font = Font(size=16, bold=True, color="17365D")
    ws.merge_cells("A1:D1")
    rows = [
        ("Purpose", "Explainable mechanism hypotheses inferred from public multilayer networks, directed TF evidence, tissue expression context, and functional annotation."),
        ("Scope", "Nine co-localized maize flowering-time candidate genes."),
        ("NAM evidence", "Not included in this module."),
        ("Primary rule", "Compare mechanisms within the same candidate. Do not use mechanism scores as cross-gene causal probabilities."),
        ("Prediction classes", "NETWORK_RANKED; DESCRIPTIVE_ONLY; UNRESOLVED. These classes must remain separate in interpretation."),
        ("L2 rule", "L2 mechanism labels are descriptive refinements only; formal network propagation is performed at L1."),
        ("Causality", "Results are public-data-supported hypotheses, not experimental validation or proof of a regulatory cascade."),
        ("Schema", str(payload.get("schema", {}).get("schema_version", ""))),
        ("Source analysis", str(payload.get("schema", {}).get("source_analysis_version", ""))),
    ]
    ws.append([])
    ws.append(["Item", "Description"])
    for cell in ws[3]:
        cell.fill = HEADER_FILL
        cell.font = WHITE_FONT
    for item, description in rows:
        ws.append([item, description])
    ws[6][0].fill = WARNING_FILL
    ws[6][1].fill = WARNING_FILL
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 110
    for row in ws.iter_rows(min_row=3, max_col=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A4"


def add_quick_view(workbook: Workbook, payload: dict[str, object]) -> None:
    ws = workbook.create_sheet("Quick_View", 1)
    summaries = sorted(payload["candidate_summary"], key=lambda x: int(x["display_order"]))
    headers = [
        "gene_id_v5", "gene_id_v4", "preferred_gene_symbol", "trait", "prediction_status", "evidence_tier",
        "top_mechanism_l1_id", "top_mechanism_l1_en", "primary_score_0_100", "primary_score_type",
        "runner_up_mechanism_l1_id", "top_vs_runner_up_margin", "top_mechanism_specificity",
        "sensitivity_top_mechanism_stability", "network_layers_assessable", "network_score_0_100",
        "tf_directed_score_0_100", "tissue_context_score_0_100", "annotation_score_0_100",
        "trait_match_score_0_100", "shortest_path_distance_edges", "terminal_seed_symbol",
        "top_tissue_context_ids", "recommended_comparison_scope", "interpretation_limit",
    ]
    ws.append(headers)
    for row in summaries:
        ws.append([stringify(row.get(header)) for header in headers])
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = WHITE_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    table = Table(displayName="T_QuickView", ref=ws.dimensions)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(table)
    for col in range(1, ws.max_column + 1):
        header = str(ws.cell(1, col).value)
        width = 18
        if header in {"gene_id_v5", "gene_id_v4"}:
            width = 20
        elif "interpretation" in header:
            width = 55
        elif "mechanism" in header or "components" in header:
            width = 30
        ws.column_dimensions[get_column_letter(col)].width = width
    for row in ws.iter_rows(min_row=2):
        status = row[headers.index("prediction_status")].value
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if status == "UNRESOLVED":
                cell.fill = WARNING_FILL
    score_col = get_column_letter(headers.index("primary_score_0_100") + 1)
    if ws.max_row > 2:
        ws.conditional_formatting.add(
            f"{score_col}2:{score_col}{ws.max_row}",
            ColorScaleRule(start_type="num", start_value=0, start_color="F7FBFF",
                           mid_type="num", mid_value=50, mid_color="6BAED6",
                           end_type="num", end_value=100, end_color="08306B"),
        )


def add_method(workbook: Workbook, payload: dict[str, object]) -> None:
    ws = workbook.create_sheet("Method")
    ws.append(["Parameter", "Value"])
    for key, value in payload.get("method", {}).items():
        ws.append([key, stringify(value)])
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = WHITE_FONT
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 110
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standardized mechanism results to Excel.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_readme(workbook, payload)
    add_quick_view(workbook, payload)
    for sheet_name, key in SHEETS:
        add_table_sheet(workbook, sheet_name, list(payload.get(key, [])))
    add_method(workbook, payload)
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    workbook.save(args.output_xlsx)

    reopened = load_workbook(args.output_xlsx, read_only=True, data_only=False)
    expected = {"README", "Quick_View", "Method", *[x[0] for x in SHEETS]}
    missing = sorted(expected - set(reopened.sheetnames))
    reopened.close()
    if missing:
        raise RuntimeError(f"Workbook validation failed; missing sheets: {missing}")
    print(json.dumps({"output_xlsx": str(args.output_xlsx.resolve()), "sheet_count": len(expected)}, indent=2))


if __name__ == "__main__":
    main()
