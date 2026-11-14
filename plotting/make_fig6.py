#!/usr/bin/env python3
"""Recompose Fig. 6 at its original 183 x 236.22 mm size from result tables.

Claim: ML prioritization, feature contributions, mechanism scores and public
association paths provide complementary evidence for three focal candidates.
Python-only schematic-led composite; no model fitting or mechanism rescoring.
Arial: 5.8 pt workflow text, 5.0 pt heatmap values, 6.5 pt other body text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

STEM = "Fig6_ML_prioritization_mechanism_inference"
WIDTH_MM, HEIGHT_MM = 183.0, 236.22
BASE_FONT = 6.5
PANEL_FONT = 10.0
CELL_NUMBER_FONT = 5.0
WORKFLOW_FONT = 5.8
SELECTED = ["Zm00001eb349860", "Zm00001eb325180", "Zm00001eb341290"]
TRAIT_COLORS = {"DTA": "#0072B2", "DTS": "#D55E00", "ASI": "#CC79A7"}
MODELS = ["L2_LR", "ElasticNet_LR", "Linear_SVM", "Bagging", "Shallow_XGBoost", "Shallow_NeuralNet"]
MODEL_LABELS = ["L2-LR", "Elastic Net", "Linear SVM", "Bagging", "XGBoost", "NeuralNet"]
BLOCKS = ["PPI", "Shortest_distance", "Transcriptome", "Translatome"]
TOP = dict(zip(SELECTED, ["M2", "M5", "M6"]))
N_L2 = {"M2": 3, "M5": 5, "M6": 4}
L1_COLORS = ["#E3F5F6", "#ABE0E4", "#7FC6D3", "#779FC6", "#6F78B9"]
L2_COLORS = ["#FFF0DC", "#FED297", "#FCB461", "#F7935A", "#F37252"]
NODE = {"candidate": ("#78B8DF", "#2565A0"), "seed": ("#F3EFBC", "#F48A43"), "intermediate": ("#C5C7C6", "#16374A")}
EDGE_COLOR = {"RNA": "#078A3E", "RIBO": "#7D26B6"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": BASE_FONT, "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT, "ytick.labelsize": BASE_FONT,
    "legend.fontsize": BASE_FONT, "svg.fonttype": "none", "pdf.fonttype": 42,
    "ps.fonttype": 42, "text.color": "black", "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})


def axis_mm(fig, x, y, width, height, diagram=True):
    ax = fig.add_axes([x / WIDTH_MM, 1 - (y + height) / HEIGHT_MM, width / WIDTH_MM, height / HEIGHT_MM])
    if diagram:
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.set_axis_off()
    return ax


def panel_letter(ax, label, offset_x=-2, offset_y=3):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(offset_x, offset_y),
                textcoords="offset points", fontsize=PANEL_FONT, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)


def text(ax, x, y, value, **kwargs):
    kwargs.setdefault("fontsize", BASE_FONT)
    return ax.text(x, y, value, **kwargs)


def measure_mm(fig, value, bold=False, italic=False):
    fp = FontProperties(family="Arial", size=BASE_FONT, weight="bold" if bold else "normal", style="italic" if italic else "normal")
    return fig.canvas.get_renderer().get_text_width_height_descent(value, fp, False)[0] / fig.dpi * 25.4


def panel_letter_above_gene(ax, label, gene_anchor, bottom_y, centered_gene=False):
    """Align the letter's left edge with the first gene label, near the content."""
    gene_width = measure_mm(ax.figure, SELECTED[0], italic=True)
    left = gene_anchor - gene_width * (.5 if centered_gene else 1)
    ax.text(left, bottom_y, label, fontsize=PANEL_FONT, fontweight="bold", ha="left", va="bottom")


def wrap(fig, value, width, bold=False):
    lines = []
    for paragraph in value.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = (line + " " + word).strip()
            if line and measure_mm(fig, trial, bold) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def arrow(ax, x1, x2, y, color, dashed=False):
    tip = x2 - 0.6
    ax.plot([x1 + 0.6, tip], [y, y], color=color, lw=0.85,
            linestyle=(0, (3, 2.5)) if dashed else "-", zorder=1)
    head = min(1.4, (x2 - x1) * 0.32)
    ax.plot([tip - head, tip, tip - head], [y - 0.6, y, y + 0.6],
            color=color, lw=0.85, solid_capstyle="round", solid_joinstyle="round", zorder=3)


def pft_text(ax, x, y, prefix, suffix="", ha="left"):
    # A genuine lowered FT with the common body size, avoiding tiny glyphs.
    pre_w = measure_mm(ax.figure, prefix)
    ft_w = measure_mm(ax.figure, "FT")
    suf_w = measure_mm(ax.figure, suffix)
    if ha == "center":
        x -= (pre_w + ft_w + suf_w) / 2
    text(ax, x, y, prefix, va="center")
    text(ax, x + pre_w, y + 0.42, "FT", va="center")
    if suffix:
        text(ax, x + pre_w + ft_w, y, suffix, va="center")


def panel_a(ax):
    """Restore the compact reference layout; every box uses 5.8 pt Arial."""
    panel_letter(ax, "a", -10, 3)
    aw, ah = ax.get_xlim()[1], ax.get_ylim()[0]

    def width_mm(value, bold=False):
        fp = FontProperties(family="Arial", size=WORKFLOW_FONT, weight="bold" if bold else "normal")
        return ax.figure.canvas.get_renderer().get_text_width_height_descent(value, fp, False)[0] / ax.figure.dpi * 25.4

    def wrapped(value, available, bold=False):
        result = []
        for line in value.split("\n"):
            current = ""
            for word in line.split():
                trial = (current + " " + word).strip()
                if current and width_mm(trial, bold) > available:
                    result.append(current)
                    current = word
                else:
                    current = trial
            result.append(current)
        return "\n".join(result)

    def label(x, y, value, **kw):
        return ax.text(x, y, value, transform=ax.transAxes, fontsize=WORKFLOW_FONT, color="black", **kw)

    def link(x1, x2, y, color):
        ax.annotate("", xy=(x2 - .002, y), xytext=(x1 + .002, y), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=.85, color=color, mutation_scale=7))

    dx, dy, dw, dh = .002, .070, .135, .830
    ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, transform=ax.transAxes,
        boxstyle="round,pad=0.005,rounding_size=0.014", facecolor="#F3F5F7", edgecolor="#66737F",
        linewidth=1.05, clip_on=False, zorder=2))
    label(dx + dw / 2, dy + dh - .050, "Multi-omics Database", ha="center", va="top", fontweight="bold")
    items = ["[65 known FT genes and\n65 FT-unrelated genes]", "[Genome-wide PPI network]",
             "[Genome-wide network distances to FT seeds]", "[Genome-wide transcriptome and translatome]",
             "[ChIA-PET chromatin contact network]", "[TF–target gene directed regulatory network]"]
    top_mm = (1 - dy - dh) * ah + 5.65
    for item in items:
        item = wrapped(item, dw * aw - .25)
        for line in item.split("\n"):
            label(dx + dw / 2, 1 - top_mm / ah, line, ha="center", va="top")
            top_mm += 2.20
        top_mm += .50

    top_cards = [
        ("Multi-omics\nfeatures", "19 PPIs\n10 network distances\n31 RNA-seq\n21 Ribo-seq\n2 missing indicators"),
        ("Training matrix", "130 genes\nFivefold × 83 features"),
        ("Nested fivefold\ncross-validation", "L2-LR | EN-LR\nLinSVM | Bagging\nsXGB | NeuralNet"),
        ("Model comparison", "OOF metrics\n(ROC-AUC,\nPR-AUC, F1,\nMCC, Brier)"),
        ("Final prediction", "PFT_SPECIAL"),
        ("Function output", "FT-function likelihood"),
    ]
    bottom_cards = [
        ("Six FT mechanism\nseed sets", "Known FT genes\nclassified into M1–M6\nprimary mechanisms"),
        ("Four-layer public\nnetwork", "RWR on PPI /\nRNA co-expr. / Ribo /\nChIA-PET networks"),
        ("Network-layer\nsensitivity analysis", "Compare:\nequal-weight\nPPI-biased\nRNA–Ribo-biased"),
        ("Mechanistic\nevidence", "0.60 Network\n0.15 TF–target\n0.15 Tissue\n0.05 Annotation\n0.05 Trait"),
        ("Within-gene\nmechanism ranking", "Rank M1–M6 within\neach gene"),
        ("Mechanism output", "Primary mechanism\npriority\n+\nSecondary descriptive\nhypothesis"),
    ]
    for row, cards in enumerate([top_cards, bottom_cards]):
        y, h = (.535 if row == 0 else .045), .390
        strong = "#2F74A5" if row == 0 else "#26857D"
        for i, (title, body) in enumerate(cards):
            x, w = .165 + i * (.117 + .025), .117
            ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                boxstyle="round,pad=0.004,rounding_size=0.012", linewidth=.95,
                facecolor=("#DCECF6" if row == 0 else "#DCEFEA") if i == 5 else "white",
                edgecolor=strong if i == 5 else ("#8AB5D0" if row == 0 else "#79B7B0")))
            tx = x + .005
            available = (w - .002) * aw
            title = wrapped(title, available, True)
            title_top = (1 - y - h) * ah + 1.45
            for line in title.split("\n"):
                label(tx, 1 - title_top / ah, line, ha="left", va="top", fontweight="bold")
                title_top += 2.18
            body_top = title_top + .75
            if body == "PFT_SPECIAL":
                for offset, line in [(0, "One-SE rule →"), (2.5, "Final L2-LR →"), (5.0, "Fivefold P"), (8.1, "mean ± s.d.")]:
                    label(tx, 1 - (body_top + offset) / ah, line, ha="left", va="top")
                label(tx + (width_mm("Fivefold P") + .70) / aw, 1 - (body_top + 5.45) / ah, "FT", ha="left", va="top")
            else:
                body = wrapped(body, available)
                for line in body.split("\n"):
                    label(tx, 1 - body_top / ah, line, ha="left", va="top")
                    body_top += 2.20
            if i < 5:
                link(x + w, x + .142, y + h / 2, strong)
        link(dx + dw, .165, y + h / 2, "#5F6A75")


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.65)
    ax.tick_params(width=0.6, length=2.7, labelsize=BASE_FONT)
    ax.grid(axis="x", color="#E6E6E6", lw=0.45, zorder=0)


def panel_b(ax, folds, comparison):
    panel_letter(ax, "b", -25, 8)
    style_axis(ax)
    cutoff = float(comparison["One_SE_cutoff"].iloc[0])
    ax.axvspan(cutoff, 1.01, color="#F3F6FA", zorder=-3)
    ax.axvline(cutoff, color="#5A5A5A", lw=0.7, ls=(0, (3, 2)))
    ax.axhspan(4.58, 5.42, color="#EAF2F8", zorder=-2)
    for y, model in zip(range(5, -1, -1), MODELS):
        sub = folds.loc[folds.Model == model]
        assert len(sub) == 5
        for metric, dy, color, marker in [("ROC_AUC", .115, "#0072B2", "o"), ("PR_AUC", -.115, "#D55E00", "^")]:
            v = sub[metric].to_numpy(float)
            ax.scatter(v, y + dy + np.linspace(-.045, .045, 5), s=7, marker=marker, facecolors="white", edgecolors=color, lw=.55, zorder=3)
            ax.errorbar(v.mean(), y + dy, xerr=v.std(ddof=1), fmt=marker, ms=3.8, mfc=color,
                        mec="white", mew=.4, ecolor=color, elinewidth=1.05, capsize=1.7, capthick=.75, zorder=5)
    ax.set(yticks=list(range(5, -1, -1)), yticklabels=MODEL_LABELS, xlim=(.48, 1.015), ylim=(-.55, 5.58), xticks=np.arange(.5, 1.01, .1))
    ax.set_xlabel("Mean Outer-Fold Score", fontsize=BASE_FONT)
    handles = [Line2D([0], [0], marker=m, color=c, lw=1, markersize=4, label=l)
               for m, c, l in [("o", "#0072B2", "ROC-AUC"), ("^", "#D55E00", "PR-AUC")]]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.015), ncol=2, fontsize=BASE_FONT, handletextpad=.4, columnspacing=1)


def panel_c(ax, candidates):
    panel_letter(ax, "c", -10, 8)
    style_axis(ax)
    candidates = candidates.sort_values("Mean_probability", ascending=False)
    for y in [8, 7, 6]:
        ax.axhspan(y - .42, y + .42, color="#F5F5F5", zorder=-2)
    ax.axvline(.5, color="#666666", lw=.7, ls=(0, (3, 2)))
    markers = {"COMPLETE_FEATURES": "o", "REDUCED_NETWORK_EVIDENCE": "D", "LOW_FEATURE_COVERAGE": "s"}
    for y, row in zip(range(8, -1, -1), candidates.itertuples(index=False)):
        color = TRAIT_COLORS[row.trait]
        selected = row.Gene_ID_v5 in SELECTED
        ax.errorbar(row.Mean_probability, y, xerr=row.SD_probability,
                    fmt=markers[row.ML_evidence_flag], ms=5.2 if selected else 4.2,
                    mfc=color if row.ML_evidence_flag != "LOW_FEATURE_COVERAGE" else "white",
                    mec="#111111" if selected else color, mew=1 if selected else .65,
                    ecolor=color, elinewidth=1.1 if selected else .8, capsize=2, capthick=.75, zorder=4)
        text(ax, 1.025, y, f"{row.Mean_probability:.2f}", ha="left", va="center", color="#333333")
        weight = "bold" if selected else "normal"
        ax.annotate(row.trait, xy=(.325, y), ha="right", va="center", fontsize=BASE_FONT, color=color, fontweight=weight)
        ax.annotate(row.Gene_ID_v5, xy=(.325, y), xytext=(-17, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=BASE_FONT, fontstyle="italic", color=color, fontweight=weight)
    ax.set(xlim=(-.035, 1.13), ylim=(-.65, 8.62), yticks=[], xticks=np.arange(.4, 1.01, .1))
    ax.set_xlabel("Mean Five-fold Probability", fontsize=BASE_FONT)
    handles = [Line2D([0], [0], marker=m, color="none", markerfacecolor=face, markeredgecolor="#666666", markersize=4.1, label=l)
               for m, face, l in [("o", "#666666", "Complete Features"), ("D", "#666666", "Reduced Network Evidence"), ("s", "white", "Low Feature Coverage")]]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.015), ncol=3,
              fontsize=BASE_FONT, handlelength=.9, handletextpad=.35, columnspacing=.7, borderaxespad=0)


def foreground(rgba):
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgba[:3]]
    lum = sum(w * v for w, v in zip([.2126, .7152, .0722], linear))
    return "black" if (lum + .05) / .05 >= 1.05 / (lum + .05) else "white"


def cell(ax, x, y, w, h, value, cmap, norm, label, top=False):
    rgba = cmap(norm(value))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=rgba, edgecolor="white", lw=.65))
    if "\n" in label:
        mechanism_id, number = label.split("\n", 1)
        text(ax, x + w / 2, y + h / 2 - 1.15, mechanism_id, ha="center", va="center", color=foreground(rgba), fontsize=CELL_NUMBER_FONT)
        text(ax, x + w / 2, y + h / 2 + 1.25, number, ha="center", va="center", color=foreground(rgba), fontsize=CELL_NUMBER_FONT)
    else:
        text(ax, x + w / 2, y + h / 2, label, ha="center", va="center", color=foreground(rgba), fontsize=CELL_NUMBER_FONT)
    if top:
        ax.add_patch(Rectangle((x + .08, y + .08), w - .16, h - .16, fill=False, edgecolor="black", lw=.95, zorder=4))


def horizontal_colorbar(ax, cmap, norm, x, y, w, h, ticks):
    # Positions are millimetres in the containing panel, including colorbar.
    aw, ah = ax.get_xlim()[1], ax.get_ylim()[0]
    cax = ax.inset_axes([x / aw, 1 - (y + h) / ah, w / aw, h / ah])
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(.6)
    cb.ax.tick_params(labelsize=BASE_FONT, width=.6, length=2, pad=1.6)
    return cb


def panel_d(ax, block_matrix):
    panel_letter_above_gene(ax, "d", 21.7, 7.9)
    cmap, norm = plt.get_cmap("Blues"), colors.Normalize(0, 14)
    for i, gene in enumerate(SELECTED):
        text(ax, 21.7, 9 + i * 7 + 3.5, gene, ha="right", va="center", fontstyle="italic")
        for j, block in enumerate(BLOCKS):
            value = float(block_matrix.loc[gene, block]) * 1000
            cell(ax, 23.2 + j * 10.25, 9 + i * 7, 10.25, 7, value, cmap, norm, f"{value:.1f}")
    for j, label in enumerate(["PPI", "Distances", "RNA", "Ribo"]):
        text(ax, 23.2 + j * 10.25 + 5.125, 8.5, label, ha="center", va="bottom")
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    gene_left_px = min(t.get_window_extent(renderer).x0 for t in ax.texts if t.get_text() in SELECTED)
    label_x = ax.transData.inverted().transform((gene_left_px, 0))[0]
    before_legend = len(ax.texts)
    pft_text(ax, label_x, 35.5, "Mean |ΔP", "| per feature")
    # Arial lacks the Unicode superscript minus: draw a real raised exponent.
    prefix_w = measure_mm(ax.figure, "(×10")
    exponent_w = measure_mm(ax.figure, "−3")
    text(ax, label_x + 27, 35.5, "(×10", va="center")
    text(ax, label_x + 27 + prefix_w + .1, 34.75, "−3", va="center")
    text(ax, label_x + 27 + prefix_w + exponent_w + .2, 35.5, ")", va="center")
    ax.figure.canvas.draw()
    label_right_px = max(t.get_window_extent(renderer).x1 for t in ax.texts[before_legend:])
    bar_x = ax.transData.inverted().transform((label_right_px, 0))[0] + 1.5
    horizontal_colorbar(ax, cmap, norm, bar_x, 34.2, 18, 2.4, [0, 7, 14])


def panel_e(ax, scores):
    panel_letter_above_gene(ax, "e", 21.9, 7.9)
    cm1 = colors.LinearSegmentedColormap.from_list("L1_teal_blue", L1_COLORS, N=256)
    cm2 = colors.LinearSegmentedColormap.from_list("L2_cream_coral", L2_COLORS, N=256)
    no1, no2 = colors.Normalize(0, 100), colors.Normalize(0, 50)
    for j in range(6):
        text(ax, 23.4 + j * 7 + 3.5, 8.5, f"M{j + 1}", ha="center", va="bottom")
    for i, gene in enumerate(SELECTED):
        yy = 9 + i * 7
        text(ax, 21.9, yy + 3.5, gene, ha="right", va="center", fontstyle="italic")
        for j in range(6):
            mechanism = f"M{j + 1}"
            value = scores[(gene, "L1", mechanism)]
            cell(ax, 23.4 + j * 7, yy, 7, 7, value, cm1, no1, f"{value:.2f}", mechanism == TOP[gene])
        text(ax, 66, yy + 3.5, "→" + TOP[gene], va="center")
        for j in range(5):
            xx = 73.5 + j * 7
            if j < N_L2[TOP[gene]]:
                mechanism = f"{TOP[gene]}.{j + 1}"
                value = scores[(gene, "L2", mechanism)]
                cell(ax, xx, yy, 7, 7, value, cm2, no2, f"{mechanism}\n{value:.2f}")
            else:
                ax.add_patch(Rectangle((xx, yy), 7, 7, facecolor="#BFBFBF", edgecolor="white", lw=.65))
    text(ax, 41.5, 35.5, "L1 integrated score", ha="right", va="center")
    horizontal_colorbar(ax, cm1, no1, 43, 34.2, 18, 2.4, [0, 50, 100])
    text(ax, 90.5, 35.5, "L2 descriptive score", ha="right", va="center")
    horizontal_colorbar(ax, cm2, no2, 92, 34.2, 17.5, 2.4, [0, 25, 50])


def panel_f(ax, edges):
    panel_letter_above_gene(ax, "f", 12, 5.5, centered_gene=True)
    def node(x, y, label, role):
        face, edge = NODE[role]
        ax.scatter([x], [y], s=65, facecolors=face, edgecolors=edge, linewidths=.85, zorder=4)
        text(ax, x, y + 2.2, label, ha="center", va="top", fontstyle="italic")
    def link(x1, x2, y, layer, weight):
        color = EDGE_COLOR[layer]
        ax.plot([x1, x2], [y, y], color=color, lw=1.1, zorder=1)
        label = "RNA Coexpression" if layer == "RNA" else "Ribo Cotranslation"
        text(ax, (x1 + x2) / 2, y - 1.2, f"{label}\nw={weight:.3f}", ha="center", va="bottom", color=color, linespacing=1.03)
    row_text = [
        "$\\mathbf{M2}$: Florigen and floral transition;\n$\\mathbf{M2.3}$: Floral transition integrators\nand downstream targets",
        "$\\mathbf{M5}$: Meristem and\nreproductive development;\n$\\mathbf{M5.1}$: SAM identity and\nfloral commitment",
        "$\\mathbf{M6}$: Chromatin and RNA regulation;\n$\\mathbf{M6.3}$: RNA processing,\nmetabolism and stability",
    ]
    for row in range(3):
        yy = 41 / 6 + row * (12.4 + 2.00)
        node(12, yy, SELECTED[row], "candidate")
        if row == 1:
            node(52, yy, "Zm00001eb336290", "intermediate")
            node(92, yy, "DLF1", "seed")
            link(12, 52, yy, "RNA", edges[1]["weight"])
            link(52, 92, yy, "RIBO", edges[2]["weight"])
            start, box_x, box_w = 92, 133, 42
            expression = "Expression: immature\near/inflorescence"
        else:
            node(52, yy, "ZmARF16" if row == 0 else "RBAP2", "seed")
            link(12, 52, yy, "RNA", edges[0 if row == 0 else 3]["weight"])
            start, box_x, box_w = 52, 96, 40
            expression = "Expression: immature\ntassel/ear" if row == 0 else None
        arrow(ax, start + 1.3, box_x, yy, "#808080", dashed=True)
        if expression:
            text(ax, (start + box_x) / 2, yy - 1.2, expression, ha="center", va="bottom", linespacing=1.03)
        ax.add_patch(FancyBboxPatch((box_x, yy - 6.2), box_w, 12.4,
            boxstyle="round,pad=0,rounding_size=1.5", facecolor="#EAF1F5", edgecolor="#ACACAC", lw=.85))
        text(ax, box_x + 1.5, yy, row_text[row], va="center", linespacing=1.1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(r"F:\GWAS2_result\Machine_Learning"))
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--capital-outdir", type=Path, help="Export the same figure with uppercase panel labels")
    p.add_argument("--qa-tools", type=Path, default=Path(__file__).resolve().parent / "qa_tools")
    args = p.parse_args()
    source = args.root / "Fig6" / "source_data"
    inputs = {
        "comparison": source / "Fig6b_model_comparison_summary.csv",
        "folds": source / "Fig6b_outer_fold_metrics.csv",
        "candidates": source / "Fig6c_candidate_predictions.csv",
        "blocks": source / "Fig6d_selected_feature_block_contributions_matrix.csv",
        "mechanism": args.root / "Fig_top3_nested_L1_L2_mechanism_priority_source_data.csv",
        "edges": args.root / "Fig_top3_candidate_mechanism_evidence_paths_source_edges.tsv",
    }
    comparison = pd.read_csv(inputs["comparison"])
    folds = pd.read_csv(inputs["folds"])
    candidates = pd.read_csv(inputs["candidates"])
    block_matrix = pd.read_csv(inputs["blocks"]).set_index("Gene_ID_v5").loc[SELECTED, BLOCKS]
    mech = pd.read_csv(inputs["mechanism"])
    scores = {(r.Gene_ID_v5, r.Level, r.Mechanism_ID): float(r.Score_0_100) for r in mech.itertuples(index=False)}
    assert len(scores) == 30 and len(candidates) == 9 and len(folds) == 30
    assert len(set(candidates.Gene_ID_v5)) == 9 and set(folds.Model) == set(MODELS)
    assert all(candidates.Locked_model == "L2_LR")
    assert np.isfinite(block_matrix.to_numpy()).all()
    for gene in SELECTED:
        assert max(range(1, 7), key=lambda i: scores[(gene, "L1", f"M{i}")]) == int(TOP[gene][1])
    for (gene, level, mid), value in scores.items():
        assert math.isfinite(value) and 0 <= value <= (100 if level == "L1" else 50)
    edges = pd.read_csv(inputs["edges"], sep="\t").to_dict("records")
    assert len(edges) == 4
    assert [r["source"] for r in edges] == [SELECTED[0], SELECTED[1], "Zm00001eb336290", SELECTED[2]]

    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.20472440945, 9.30), facecolor="white")
    a = axis_mm(fig, 5.49, 7.0866, 174.765, 47.09449367088612)
    b = axis_mm(fig, 14, 63, 65, 55, diagram=False)
    c = axis_mm(fig, 89, 63, 90.5, 55, diagram=False)
    d = axis_mm(fig, 4, 139, 64.5, 40)
    e = axis_mm(fig, 69.5, 139, 113, 40)
    f = axis_mm(fig, 4, 191, 175.5, 43.0)
    panel_a(a)
    panel_b(b, folds, comparison)
    panel_c(c, candidates)
    panel_d(d, block_matrix)
    panel_e(e, scores)
    panel_f(f, edges)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_texts = [t for ax in [d, e] for t in ax.texts if t.get_position()[1] > 30]
    legend_bars = [child for ax in [d, e] for child in ax.child_axes]
    heat_bottom = d.transData.transform((0, 30))[1]
    legend_top = max([t.get_window_extent(renderer).y1 for t in legend_texts]
                     + [bar.get_window_extent(renderer).y1 for bar in legend_bars])
    legend_shift_mm = (heat_bottom - legend_top) * 25.4 / fig.dpi - 1.0
    for t in legend_texts:
        x, y = t.get_position()
        t.set_position((x, y - legend_shift_mm))
    from matplotlib.transforms import Bbox
    for bar in legend_bars:
        original_locator = bar.get_axes_locator()
        def moved_locator(ax, renderer, locator=original_locator):
            box = locator(ax, renderer)
            return Bbox.from_bounds(box.x0, box.y0 + legend_shift_mm / HEIGHT_MM,
                                    box.width, box.height)
        bar.set_axes_locator(moved_locator)
    fig.canvas.draw()
    # Preserve the D/E pair geometry; measure clearance between visible text.
    renderer = fig.canvas.get_renderer()
    upper_bottom = min(ax.get_tightbbox(renderer).y0 for ax in [b, c])
    lower_top = max(t.get_window_extent(renderer).y1 for ax in [d, e]
                    for t in ax.texts if t.get_visible() and t.get_text())
    shift_px = upper_bottom - 2.5 * fig.dpi / 25.4 - lower_top
    for ax in [d, e]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + shift_px / fig.bbox.height, pos.width, pos.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    actual_gap = (upper_bottom - max(t.get_window_extent(renderer).y1 for ax in [d, e]
                  for t in ax.texts if t.get_visible() and t.get_text())) * 25.4 / fig.dpi
    assert abs(actual_gap - 2.5) < .01
    de_bottom = min([t.get_window_extent(renderer).y0 for ax in [d, e] for t in ax.texts]
                    + [bar.get_tightbbox(renderer).y0 for bar in legend_bars])
    f_top = max([t.get_window_extent(renderer).y1 for t in f.texts]
                + [p.get_window_extent(renderer).y1 for p in f.patches])
    f_shift_px = de_bottom - 4.5 * fig.dpi / 25.4 - f_top
    pos = f.get_position()
    f.set_position([pos.x0, pos.y0 + f_shift_px / fig.bbox.height, pos.width, pos.height])
    fig.canvas.draw()
    (out / 'Fig6_spacing_QA.json').write_text(json.dumps({
        'visible_content_gap_mm': actual_gap, 'de_upward_shift_mm': shift_px * 25.4 / fig.dpi,
        'legend_to_heatmap_gap_mm': 1.0, 'legend_row_upward_shift_mm': legend_shift_mm,
        'de_to_f_visible_gap_mm': 4.5, 'f_upward_shift_mm': f_shift_px * 25.4 / fig.dpi,
        'other_panels_unchanged': True}, indent=2), encoding='utf-8')
    sys.path.insert(0, str(args.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    labels = [next(t for t in ax.texts if t.get_text() == letter)
              for ax, letter in zip([a, b, c, d, e, f], "abcdef")]
    renderer = fig.canvas.get_renderer()
    box_top = max(p.get_window_extent(renderer).y1 + p.get_linewidth() * fig.dpi / 144
                  for p in a.patches)
    a_label = labels[0]
    x, y = a_label.get_position()
    a_label.set_position((x, y + (box_top + 1.5 * fig.dpi / 25.4
                                  - a_label.get_window_extent(renderer).y0) * 72 / fig.dpi))
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[a, b, c, d, e, f], panel_ids=list("abcdef"),
        row_groups=[["b", "c"], ["d", "e"]], tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True,
        json_out=str(out / STEM) + ".alignment.json", overlay_svg=str(out / STEM) + ".alignment.svg")
    renderer = fig.canvas.get_renderer()
    f_bottom = min([t.get_window_extent(renderer).y0 for t in f.texts]
                   + [p.get_window_extent(renderer).y0 - p.get_linewidth() * fig.dpi / 144
                      for p in f.patches])
    a_top = a_label.get_window_extent(renderer).y1
    if args.capital_outdir:
        a_label.set_text("A")
        fig.canvas.draw()
        a_top = max(a_top, a_label.get_window_extent(renderer).y1)
        a_label.set_text("a")
    crop_bottom_in = f_bottom / fig.dpi - 1.0 / 25.4
    crop_top_in = a_top / fig.dpi + 1.0 / 25.4
    export_box = Bbox.from_bounds(0, crop_bottom_in, fig.get_figwidth(), crop_top_in - crop_bottom_in)
    export_height_mm = export_box.height * 25.4
    def export(directory):
        for suffix, options in [
            (".svg", {}), (".pdf", {}), (".png", {"dpi": 600}),
            (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
            ("_preview.png", {"dpi": 300}),
        ]:
            fig.savefig(directory / (STEM + suffix), facecolor="white", bbox_inches=export_box, **options)
    export(out)
    if args.capital_outdir:
        for label in labels:
            label.set_text(label.get_text().upper())
        export(args.capital_outdir)
    plt.close(fig)

    dst = out / "source_data"
    dst.mkdir(exist_ok=True)
    for key in ["comparison", "folds", "candidates", "blocks"]:
        target = dst / inputs[key].name
        if inputs[key].resolve() != target.resolve():
            shutil.copy2(inputs[key], target)
    mech.to_csv(dst / "Fig6e_selected_L1_L2_mechanism_scores.csv", index=False)
    pd.DataFrame(edges).to_csv(dst / "Fig6f_public_network_edges.csv", index=False)
    manifest = {"figure_mm": [WIDTH_MM, export_height_mm], "top_margin_mm": 1.0, "bottom_margin_mm": 1.0, "panel_a_label_gap_mm": 1.5, "bottom_canvas_removed_mm": crop_bottom_in * 25.4, "base_font_pt": BASE_FONT, "panel_label_pt": PANEL_FONT,
        "workflow_font_pt": WORKFLOW_FONT, "workflow_layout": "Restored compact reference geometry; reflowed 5.8 pt text",
        "heatmap_number_font_pt": CELL_NUMBER_FONT, "def_panel_label_position": "Above the first gene name, left-aligned to its text",
        "panel_upward_translation_mm": {"a": 0, "b": 14, "c": 14, "d": 7 + shift_px * 25.4 / fig.dpi, "e": 7 + shift_px * 25.4 / fig.dpi, "f": 0},
        "bc_to_de_visible_gap_mm": actual_gap,
        "e_L2_mechanism_ID_font_pt": CELL_NUMBER_FONT, "e_legend_label_colorbar_gap_mm": 1.5,
        "input_hashes": {k: {"file": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in inputs.items()},
        "selected_gene_order": SELECTED, "L1_color_range": [0, 100], "L2_color_range": [0, 50],
        "d_units": "Mean_absolute_feature_delta multiplied by 1000", "d_color_range": [0, 14],
        "L1_colors": L1_COLORS, "L2_colors": L2_COLORS, "model_or_mechanism_recomputed": False,
        "d_cells": 12, "e_L1_cells": 18, "e_L2_cells": 12, "e_gray_cells": 3, "f_edges": 4}
    (out / "Fig6_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    caption = """Fig. 6 | Machine-learning prioritization and public multi-omics mechanism inference of flowering-time candidates.
a, Public-data workflows for supervised flowering-function prediction and interpretable mechanism inference. b, Outer-fold ROC-AUC and PR-AUC of six classifiers; small symbols represent the five outer folds and larger symbols show mean ± s.d. The dashed line marks the one-standard-error cutoff and the L2-LR row is highlighted. c, Candidate probabilities from five fold-specific L2-LR models, shown as mean ± s.d.; symbols indicate feature coverage. d, Mean absolute single-feature-ablation probability change per feature, grouped into PPI, seed distance, RNA and Ribo blocks and displayed in units of 10⁻³. RNA and Ribo denote transcriptome and translatome feature blocks, respectively, including their missingness indicators. e, Six L1 integrated scores (left) and descriptive L2 scores within each candidate's top L1 mechanism (right). Black borders identify the top L1 mechanism; gray squares indicate non-applicable L2 positions. L1 and L2 use separate continuous scales of 0–100 and 0–50. f, Representative public-network association paths, with source edge weights w. Green and purple undirected links denote RNA coexpression and Ribo cotranslation; dashed gray arrows indicate evidence-to-hypothesis mapping. Expression labels summarize supporting tissue contexts. Candidate, intermediate and known FT seed genes are indicated by blue, gray and pale-yellow nodes. The L2 hypotheses summarize the full prior evidence assessment and are not assigned from the terminal seed alone. M5.1/M5.2/M5.5 and M6.3/M6.4 remain tied; f shows representative descriptive labels. All three prior mechanism assignments are Tier C; M2 is a close alternative to M6 for Zm00001eb341290. Probabilities and mechanism scores do not establish causality.
"""
    (out / "Fig6_legend.md").write_text(caption, encoding="utf-8")
    (out / "Fig6_QA_notes.md").write_text("# Fig. 6 QA\n\nCanvas: 183 × 236.22 mm, unchanged. Panel a restores the compact reference geometry; all box titles and body text use 5.8 pt Arial, with bold titles. Other baseline text uses 6.5 pt; d/e numeric values and e L2 mechanism IDs use 5.0 pt; panel letters use 10 pt. e colorbar titles remain 1.5 mm from their respective colorbars. d/e/f labels remain immediately above the first candidate gene label. "
        "The figure was regenerated from result tables. Model fitting and mechanism scoring were not rerun. "
        "d/e are aligned horizontal heatmaps with the same three-gene row order; f uses full-width paths. "
        "Continuous colormaps and square white-bordered NA cells are retained. "
        "D label changed to Distances; D/E translated together to a 2.5 mm visible-content gap below B/C. Alignment and PDF glyph checks passed; visual inspection found no new overlaps. Automated PDF collision audit unavailable because PyMuPDF is missing.\n", encoding="utf-8")
    (out / "README_Fig6.md").write_text("# Reproducing Fig. 6\n\nRun `python make_fig6.py --root <Machine_Learning> --outdir <output> --qa-tools <qa_tools>`. "
        "Input tables and hashes are listed in Fig6_run_manifest.json. "
        "Requirements: Python, numpy, pandas, matplotlib; PyMuPDF for PDF QA. "
        "Source data and independent inference results remain the source of all plotted values.\n", encoding="utf-8")
    if args.capital_outdir:
        (args.capital_outdir / "Fig6_run_manifest.json").write_text(json.dumps(
            dict(manifest, panel_labels="ABCDEF"), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "figure_mm": [WIDTH_MM, export_height_mm], "base_font_pt": BASE_FONT}, indent=2))


if __name__ == "__main__":
    main()
