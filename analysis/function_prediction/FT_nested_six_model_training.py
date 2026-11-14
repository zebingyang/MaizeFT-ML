"""Reproduce the maize flowering-time six-model nested-CV analysis.

This is the publication-ready analysis script used for the following tasks:

1. train six classifiers with fixed five-fold outer validation and four-group
   inner hyperparameter tuning;
2. generate out-of-fold predictions for 130 independent training genes;
3. compare ROC-AUC, PR-AUC, F1, MCC, balanced accuracy and calibration;
4. lock one final model with a prespecified one-standard-error rule;
5. predict nine candidate genes with the five outer-fold models;
6. quantify global permutation importance and local feature-ablation effects;
7. save fitted models, machine-readable tables, figures and run manifests.

Required input files
--------------------
FT_training_feature_matrix.csv
    650 rows: 130 genes x five precomputed outer-fold feature views.
candidate9_feature_matrix.csv
    45 rows: nine genes x five precomputed outer-fold feature views.

The first 14 columns are metadata and the remaining 83 columns are numerical
machine-learning features. The script validates the complete study design before
fitting any model.

Example
-------
python FT_nested_six_model_training.py \
  --input-dir /path/to/input \
  --output-dir /path/to/results

Interpretation
--------------
The resulting probabilities quantify multi-omics similarity to the curated maize
flowering-time training genes. They are not causal probabilities and do not by
themselves establish a molecular mechanism.

Analysis version: 1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

__version__ = "1.0.0"
RANDOM_SEED = 20260823
OUTER_FOLDS = ["F1", "F2", "F3", "F4", "F5"]
META_COLUMNS = 14
THRESHOLD = 0.5
BOOTSTRAP_REPEATS = 1000
PERMUTATION_REPEATS = 50
GRID_N_JOBS = 4

MODEL_ORDER = [
    "L2_LR",
    "ElasticNet_LR",
    "Linear_SVM",
    "Bagging",
    "Shallow_XGBoost",
    "Shallow_NeuralNet",
]

MODEL_DISPLAY = {
    "L2_LR": "L2 Logistic Regression",
    "ElasticNet_LR": "Elastic Net Logistic Regression",
    "Linear_SVM": "Linear SVM",
    "Bagging": "Bagging",
    "Shallow_XGBoost": "Shallow XGBoost",
    "Shallow_NeuralNet": "Shallow NeuralNet",
}

PALETTE = {
    "L2_LR": "#0072B2",
    "ElasticNet_LR": "#56B4E9",
    "Linear_SVM": "#009E73",
    "Bagging": "#E69F00",
    "Shallow_XGBoost": "#D55E00",
    "Shallow_NeuralNet": "#CC79A7",
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def feature_block(feature: str) -> str:
    if feature.startswith("PPI_"):
        return "PPI"
    if feature.startswith("SD_"):
        return "Shortest_distance"
    if feature.startswith("RNA_") or feature == "MISS_Transcriptome":
        return "Transcriptome"
    if feature.startswith("Ribo_") or feature == "MISS_Translatome":
        return "Translatome"
    return "Other"


def expected_calibration_error(y_true, probability, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probability, boundaries[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return ece


def metric_dict(y_true, probability, threshold=THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": average_precision_score(y_true, probability),
        "F1_at_0.5": f1_score(y_true, predicted, zero_division=0),
        "MCC_at_0.5": matthews_corrcoef(y_true, predicted),
        "Balanced_accuracy_at_0.5": balanced_accuracy_score(y_true, predicted),
        "Brier_score": brier_score_loss(y_true, probability),
        "Log_loss": log_loss(y_true, np.clip(probability, 1e-7, 1 - 1e-7)),
        "ECE_10_bins": expected_calibration_error(y_true, probability, n_bins=10),
    }


def bootstrap_intervals(y_true, probability, repeats=BOOTSTRAP_REPEATS, seed=RANDOM_SEED):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    roc_values = np.empty(repeats, dtype=float)
    pr_values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        rng.shuffle(sample)
        roc_values[index] = roc_auc_score(y_true[sample], probability[sample])
        pr_values[index] = average_precision_score(y_true[sample], probability[sample])
    return {
        "ROC_AUC_CI_low": float(np.quantile(roc_values, 0.025)),
        "ROC_AUC_CI_high": float(np.quantile(roc_values, 0.975)),
        "PR_AUC_CI_low": float(np.quantile(pr_values, 0.025)),
        "PR_AUC_CI_high": float(np.quantile(pr_values, 0.975)),
    }


def make_scaled_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def make_tree_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", model),
        ]
    )


def model_specs(seed=RANDOM_SEED):
    specs = OrderedDict()

    specs["L2_LR"] = (
        make_scaled_pipeline(
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=10000,
                random_state=seed,
            )
        ),
        {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    )

    specs["ElasticNet_LR"] = (
        make_scaled_pipeline(
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=15000,
                random_state=seed,
            )
        ),
        {
            "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
    )

    specs["Linear_SVM"] = (
        make_scaled_pipeline(
            SVC(
                kernel="linear",
                probability=True,
                random_state=seed,
            )
        ),
        {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    )

    specs["Bagging"] = (
        make_tree_pipeline(
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=seed),
                bootstrap=True,
                max_samples=0.8,
                random_state=seed,
                n_jobs=1,
            )
        ),
        {
            "model__n_estimators": [100, 300],
            "model__max_features": [0.7, 1.0],
            "model__estimator__max_depth": [2, 4],
        },
    )

    specs["Shallow_XGBoost"] = (
        make_tree_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                reg_alpha=0.0,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
        ),
        {
            "model__n_estimators": [50, 150],
            "model__max_depth": [1, 2],
            "model__learning_rate": [0.03, 0.1],
        },
    )

    specs["Shallow_NeuralNet"] = (
        make_scaled_pipeline(
            MLPClassifier(
                solver="adam",
                learning_rate_init=0.001,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=30,
                random_state=seed,
            )
        ),
        {
            "model__hidden_layer_sizes": [(8,), (16,), (32,)],
            "model__alpha": [0.001, 0.01, 0.1],
        },
    )
    return specs


def validate_inputs(train, candidate, feature_columns):
    errors = []
    if train.shape != (650, 97):
        errors.append(f"Training shape is {train.shape}, expected (650, 97)")
    if candidate.shape != (45, 97):
        errors.append(f"Candidate shape is {candidate.shape}, expected (45, 97)")
    if list(train.columns) != list(candidate.columns):
        errors.append("Training and candidate headers/order differ")
    if len(feature_columns) != 83:
        errors.append(f"Expected 83 features, found {len(feature_columns)}")
    if train["Gene_ID_v5"].nunique() != 130:
        errors.append("Training matrix does not contain 130 unique genes")
    if candidate["Gene_ID_v5"].nunique() != 9:
        errors.append("Candidate matrix does not contain 9 unique genes")
    unique_labels = train.drop_duplicates("Gene_ID_v5")["Label"].value_counts().to_dict()
    if unique_labels != {0: 65, 1: 65}:
        errors.append(f"Unique label counts are {unique_labels}, expected {{0: 65, 1: 65}}")
    for fold in OUTER_FOLDS:
        part = train.loc[train["Validation_fold"] == fold]
        train_part = part.loc[part["Use_as_training_sample_in_fold"] == 1]
        valid_part = part.loc[part["Use_as_validation_sample_in_fold"] == 1]
        if len(part) != 130 or len(train_part) != 104 or len(valid_part) != 26:
            errors.append(f"Unexpected row counts in {fold}")
        if train_part["Assigned_CV_fold"].nunique() != 4:
            errors.append(f"{fold} does not contain four inner groups")
        if len(candidate.loc[candidate["Validation_fold"] == fold]) != 9:
            errors.append(f"Candidate matrix does not contain 9 rows for {fold}")
    if errors:
        raise ValueError("\n".join(errors))


def write_figures(oof, fold_metrics, comparison, candidate_summary, global_importance, output_dir):
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="Random")
    for model_name in MODEL_ORDER:
        part = oof.loc[oof["Model"] == model_name]
        fpr, tpr, _ = roc_curve(part["Label"], part["Predicted_probability"])
        auc = roc_auc_score(part["Label"], part["Predicted_probability"])
        ax.plot(fpr, tpr, color=PALETTE[model_name], linewidth=2, label=f"{MODEL_DISPLAY[model_name]} ({auc:.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Out-of-fold ROC curves")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_ROC_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    prevalence = float(oof.drop_duplicates(["Gene_ID_v5", "Label"])["Label"].mean())
    ax.axhline(prevalence, linestyle="--", color="#777777", linewidth=1, label=f"Prevalence ({prevalence:.2f})")
    for model_name in MODEL_ORDER:
        part = oof.loc[oof["Model"] == model_name]
        precision, recall, _ = precision_recall_curve(part["Label"], part["Predicted_probability"])
        ap = average_precision_score(part["Label"], part["Predicted_probability"])
        ax.plot(recall, precision, color=PALETTE[model_name], linewidth=2, label=f"{MODEL_DISPLAY[model_name]} ({ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Out-of-fold precision-recall curves", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_PR_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    long_metrics = fold_metrics.melt(
        id_vars=["Model", "Outer_fold"],
        value_vars=["ROC_AUC", "PR_AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, metric in zip(axes, ["ROC_AUC", "PR_AUC"]):
        part = long_metrics.loc[long_metrics["Metric"] == metric].copy()
        order = comparison.sort_values(f"Mean_fold_{metric}", ascending=False)["Model"].tolist()
        sns.boxplot(data=part, x="Model", y="Score", order=order, ax=ax, color="#D9EAF7", showfliers=False)
        sns.stripplot(data=part, x="Model", y="Score", order=order, ax=ax, hue="Model", palette=PALETTE, size=5, jitter=0.12, legend=False)
        ax.set_title(metric.replace("_", "-"))
        ax.set_xlabel("")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=40)
    axes[0].set_ylabel("Outer-fold score")
    axes[1].set_ylabel("")
    fig.suptitle("Nested outer-fold model comparison", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="Perfect calibration")
    for model_name in MODEL_ORDER:
        part = oof.loc[oof["Model"] == model_name]
        observed, predicted = calibration_curve(
            part["Label"], part["Predicted_probability"], n_bins=5, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", color=PALETTE[model_name], linewidth=1.8, label=MODEL_DISPLAY[model_name])
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Out-of-fold calibration (5 quantile bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_candidate = candidate_summary.sort_values("Mean_probability", ascending=True).copy()
    color_map = {
        "COMPLETE_FEATURES": "#009E73",
        "REDUCED_NETWORK_EVIDENCE": "#E69F00",
        "LOW_FEATURE_COVERAGE": "#D55E00",
    }
    colors = [color_map.get(value, "#777777") for value in plot_candidate["ML_evidence_flag"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y_pos = np.arange(len(plot_candidate))
    ax.errorbar(
        plot_candidate["Mean_probability"],
        y_pos,
        xerr=plot_candidate["SD_probability"].fillna(0),
        fmt="none",
        ecolor="#666666",
        elinewidth=1.2,
        capsize=3,
    )
    ax.scatter(plot_candidate["Mean_probability"], y_pos, c=colors, s=55, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(THRESHOLD, linestyle="--", color="#555555", linewidth=1)
    ax.set_yticks(y_pos, plot_candidate["Gene_ID_v5"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean five-fold probability")
    ax.set_ylabel("")
    ax.set_title(f"Candidate-gene predictions: {plot_candidate['Locked_model'].iloc[0]}")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label.replace("_", " ").title())
        for label, color in color_map.items()
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_candidate9_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top = global_importance.sort_values("Mean_permutation_importance", ascending=False).head(25).sort_values("Mean_permutation_importance")
    block_colors = {
        "PPI": "#0072B2",
        "Shortest_distance": "#CC79A7",
        "Transcriptome": "#009E73",
        "Translatome": "#E69F00",
        "Other": "#777777",
    }
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    ax.barh(
        top["Feature"],
        top["Mean_permutation_importance"],
        xerr=top["SD_across_outer_folds"].fillna(0),
        color=[block_colors.get(block, "#777777") for block in top["Feature_block"]],
        alpha=0.9,
    )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Decrease in validation ROC-AUC after permutation")
    ax.set_ylabel("")
    ax.set_title("Locked-model global feature importance")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=block.replace("_", " ")) for block, color in block_colors.items() if block in set(top["Feature_block"])]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_feature_importance_top25.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir
    default_input_names = ("FT_training_feature_matrix.csv", "candidate9_feature_matrix.csv")
    if not all((script_dir / name).is_file() for name in default_input_names) and all(
        (script_dir.parent / name).is_file() for name in default_input_names
    ):
        default_input_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Reproduce the nested five-fold, six-model maize flowering-time analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help=(
            "Directory containing FT_training_feature_matrix.csv and candidate9_feature_matrix.csv; "
            "if both files are absent beside the script but present one level above, the parent is used"
        ),
    )
    parser.add_argument(
        "--training-file",
        default="FT_training_feature_matrix.csv",
        help="Training feature-matrix filename, resolved relative to --input-dir",
    )
    parser.add_argument(
        "--candidate-file",
        default="candidate9_feature_matrix.csv",
        help="Candidate feature-matrix filename, resolved relative to --input-dir",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; default is <input-dir>/FT_six_model_results_reproduced",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=GRID_N_JOBS,
        help="Parallel workers for grid search and permutation importance",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=BOOTSTRAP_REPEATS,
        help="Stratified bootstrap replicates for ROC-AUC and PR-AUC confidence intervals",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=PERMUTATION_REPEATS,
        help="Feature permutations per outer validation fold",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the input matrices and exit without model training",
    )
    parser.add_argument(
        "--allow-nonempty-output",
        action="store_true",
        help="Allow known outputs to be overwritten in a nonempty directory; stale files are not deleted",
    )
    args = parser.parse_args()

    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1")
    if args.bootstrap_repeats < 100:
        parser.error("--bootstrap-repeats must be at least 100")
    if args.permutation_repeats < 1:
        parser.error("--permutation-repeats must be at least 1")

    started = time.time()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "FT_six_model_results_reproduced"
    )
    train_path = input_dir / args.training_file
    candidate_path = input_dir / args.candidate_file
    if not train_path.is_file():
        parser.error(f"Training matrix not found: {train_path}")
    if not candidate_path.is_file():
        parser.error(f"Candidate matrix not found: {candidate_path}")
    log(f"Reading {train_path}")
    train = pd.read_csv(train_path, na_values=["NA"])
    candidate = pd.read_csv(candidate_path, na_values=["NA"])
    feature_columns = list(train.columns[META_COLUMNS:])
    validate_inputs(train, candidate, feature_columns)
    log("Input validation PASS: 130 unique genes, 83 features, 5 balanced outer folds")

    for column in feature_columns:
        train[column] = pd.to_numeric(train[column], errors="raise")
        candidate[column] = pd.to_numeric(candidate[column], errors="raise")

    if args.validate_only:
        log("Validation-only run complete; no models were fitted")
        return

    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_nonempty_output:
        parser.error(
            f"Output directory is not empty: {output_dir}. Use a new directory or pass "
            "--allow-nonempty-output to overwrite known files without deleting stale files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models_all_outer_folds"
    models_dir.mkdir(parents=True, exist_ok=True)
    locked_models_dir = output_dir / "models_locked"
    locked_models_dir.mkdir(parents=True, exist_ok=True)

    specs = model_specs(seed=RANDOM_SEED)
    fold_metric_rows = []
    oof_rows = []
    candidate_all_rows = []
    inner_result_rows = []
    best_param_rows = []
    fitted_models = {}

    for outer_index, outer_fold in enumerate(OUTER_FOLDS, start=1):
        fold_data = train.loc[train["Validation_fold"] == outer_fold].copy()
        outer_train = fold_data.loc[fold_data["Use_as_training_sample_in_fold"] == 1].copy()
        outer_valid = fold_data.loc[fold_data["Use_as_validation_sample_in_fold"] == 1].copy()
        candidate_fold = candidate.loc[candidate["Validation_fold"] == outer_fold].copy()

        X_train = outer_train[feature_columns]
        y_train = outer_train["Label"].astype(int).to_numpy()
        X_valid = outer_valid[feature_columns]
        y_valid = outer_valid["Label"].astype(int).to_numpy()
        X_candidate = candidate_fold[feature_columns]

        inner_labels = sorted(outer_train["Assigned_CV_fold"].astype(str).unique())
        inner_map = {label: index for index, label in enumerate(inner_labels)}
        inner_test_fold = outer_train["Assigned_CV_fold"].astype(str).map(inner_map).to_numpy()
        inner_cv = PredefinedSplit(inner_test_fold)
        log(f"Outer {outer_fold} ({outer_index}/5): train={len(outer_train)}, validation={len(outer_valid)}, inner_groups={inner_labels}")

        for model_index, (model_name, (pipeline, parameter_grid)) in enumerate(specs.items(), start=1):
            model_started = time.time()
            log(f"  Training {model_name} ({model_index}/6) with nested inner CV")
            search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid=parameter_grid,
                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                refit="roc_auc",
                cv=inner_cv,
                n_jobs=args.n_jobs,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            probability = best_estimator.predict_proba(X_valid)[:, 1]
            predicted = (probability >= THRESHOLD).astype(int)
            candidate_probability = best_estimator.predict_proba(X_candidate)[:, 1]
            metrics = metric_dict(y_valid, probability)
            fold_metric_rows.append(
                {
                    "Model": model_name,
                    "Model_display": MODEL_DISPLAY[model_name],
                    "Outer_fold": outer_fold,
                    "N_train": len(outer_train),
                    "N_validation": len(outer_valid),
                    "Inner_best_ROC_AUC": float(search.best_score_),
                    **metrics,
                }
            )

            for row_position, (_, row) in enumerate(outer_valid.reset_index(drop=True).iterrows()):
                oof_rows.append(
                    {
                        "Gene_ID_v5": row["Gene_ID_v5"],
                        "Gene_ID_v4": row["Gene_ID_v4"],
                        "Gene_symbol": row["Gene_symbol"],
                        "Label": int(row["Label"]),
                        "Sample_role": row["Sample_role"],
                        "Evidence_level": row["Evidence_level"],
                        "CV_group": row["CV_group"],
                        "Assigned_CV_fold": row["Assigned_CV_fold"],
                        "Validation_fold": outer_fold,
                        "Feature_QC": row["Feature_QC"],
                        "Model": model_name,
                        "Model_display": MODEL_DISPLAY[model_name],
                        "Predicted_probability": float(probability[row_position]),
                        "Predicted_label_at_0.5": int(predicted[row_position]),
                    }
                )

            for row_position, (_, row) in enumerate(candidate_fold.reset_index(drop=True).iterrows()):
                candidate_all_rows.append(
                    {
                        "Gene_ID_v5": row["Gene_ID_v5"],
                        "Gene_ID_v4": row["Gene_ID_v4"],
                        "Gene_symbol": row["Gene_symbol"],
                        "Validation_fold": outer_fold,
                        "Feature_QC": row["Feature_QC"],
                        "Model": model_name,
                        "Model_display": MODEL_DISPLAY[model_name],
                        "Predicted_probability": float(candidate_probability[row_position]),
                        "Predicted_label_at_0.5": int(candidate_probability[row_position] >= THRESHOLD),
                    }
                )

            params_json = json.dumps(search.best_params_, default=json_safe, ensure_ascii=False, sort_keys=True)
            best_param_rows.append(
                {
                    "Model": model_name,
                    "Outer_fold": outer_fold,
                    "Best_parameters": params_json,
                    "Best_inner_ROC_AUC": float(search.best_score_),
                    "Best_inner_PR_AUC": float(search.cv_results_["mean_test_pr_auc"][search.best_index_]),
                }
            )
            cv_results = search.cv_results_
            for param_index, params in enumerate(cv_results["params"]):
                inner_result_rows.append(
                    {
                        "Model": model_name,
                        "Outer_fold": outer_fold,
                        "Parameter_index": param_index,
                        "Parameters": json.dumps(params, default=json_safe, ensure_ascii=False, sort_keys=True),
                        "Mean_inner_ROC_AUC": float(cv_results["mean_test_roc_auc"][param_index]),
                        "SD_inner_ROC_AUC": float(cv_results["std_test_roc_auc"][param_index]),
                        "Rank_inner_ROC_AUC": int(cv_results["rank_test_roc_auc"][param_index]),
                        "Mean_inner_PR_AUC": float(cv_results["mean_test_pr_auc"][param_index]),
                        "SD_inner_PR_AUC": float(cv_results["std_test_pr_auc"][param_index]),
                    }
                )

            fitted_models[(model_name, outer_fold)] = best_estimator
            model_output_dir = models_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_estimator, model_output_dir / f"{model_name}_{outer_fold}.joblib", compress=3)
            elapsed = time.time() - model_started
            log(
                f"  Finished {model_name} in {elapsed:.1f}s; inner AUC={search.best_score_:.3f}, "
                f"outer AUC={metrics['ROC_AUC']:.3f}, outer PR-AUC={metrics['PR_AUC']:.3f}"
            )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    oof = pd.DataFrame(oof_rows)
    candidate_all = pd.DataFrame(candidate_all_rows)
    inner_results = pd.DataFrame(inner_result_rows)
    best_params = pd.DataFrame(best_param_rows)

    errors = []
    for model_name in MODEL_ORDER:
        part = oof.loc[oof["Model"] == model_name]
        if len(part) != 130 or part["Gene_ID_v5"].nunique() != 130:
            errors.append(f"{model_name} OOF rows/genes invalid: {len(part)}/{part['Gene_ID_v5'].nunique()}")
        cand_part = candidate_all.loc[candidate_all["Model"] == model_name]
        if len(cand_part) != 45 or cand_part["Gene_ID_v5"].nunique() != 9:
            errors.append(f"{model_name} candidate rows/genes invalid: {len(cand_part)}/{cand_part['Gene_ID_v5'].nunique()}")
    if errors:
        raise RuntimeError("\n".join(errors))

    comparison_rows = []
    for model_index, model_name in enumerate(MODEL_ORDER):
        part = oof.loc[oof["Model"] == model_name].copy()
        pooled = metric_dict(part["Label"], part["Predicted_probability"])
        intervals = bootstrap_intervals(
            part["Label"],
            part["Predicted_probability"],
            repeats=args.bootstrap_repeats,
            seed=RANDOM_SEED + model_index,
        )
        fold_part = fold_metrics.loc[fold_metrics["Model"] == model_name]
        comparison_rows.append(
            {
                "Model": model_name,
                "Model_display": MODEL_DISPLAY[model_name],
                "Mean_fold_ROC_AUC": fold_part["ROC_AUC"].mean(),
                "SD_fold_ROC_AUC": fold_part["ROC_AUC"].std(ddof=1),
                "SE_fold_ROC_AUC": fold_part["ROC_AUC"].std(ddof=1) / math.sqrt(len(fold_part)),
                "Mean_fold_PR_AUC": fold_part["PR_AUC"].mean(),
                "SD_fold_PR_AUC": fold_part["PR_AUC"].std(ddof=1),
                "SE_fold_PR_AUC": fold_part["PR_AUC"].std(ddof=1) / math.sqrt(len(fold_part)),
                "Pooled_OOF_ROC_AUC": pooled["ROC_AUC"],
                "Pooled_OOF_PR_AUC": pooled["PR_AUC"],
                "Pooled_OOF_F1_at_0.5": pooled["F1_at_0.5"],
                "Pooled_OOF_MCC_at_0.5": pooled["MCC_at_0.5"],
                "Pooled_OOF_Balanced_accuracy_at_0.5": pooled["Balanced_accuracy_at_0.5"],
                "Pooled_OOF_Brier_score": pooled["Brier_score"],
                "Pooled_OOF_Log_loss": pooled["Log_loss"],
                "Pooled_OOF_ECE_10_bins": pooled["ECE_10_bins"],
                **intervals,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    raw_best_row = comparison.sort_values(
        ["Mean_fold_ROC_AUC", "Mean_fold_PR_AUC", "Pooled_OOF_Brier_score"],
        ascending=[False, False, True],
    ).iloc[0]
    raw_best_model = raw_best_row["Model"]
    one_se_cutoff = float(raw_best_row["Mean_fold_ROC_AUC"] - raw_best_row["SE_fold_ROC_AUC"])
    comparison["Within_one_SE_of_best"] = comparison["Mean_fold_ROC_AUC"] >= one_se_cutoff
    eligible = set(comparison.loc[comparison["Within_one_SE_of_best"], "Model"])
    locked_model = next(model for model in MODEL_ORDER if model in eligible)
    comparison["Raw_best_model"] = comparison["Model"] == raw_best_model
    comparison["Locked_by_one_SE_rule"] = comparison["Model"] == locked_model
    comparison["One_SE_cutoff"] = one_se_cutoff
    comparison["Rank_mean_fold_ROC_AUC"] = comparison["Mean_fold_ROC_AUC"].rank(method="min", ascending=False).astype(int)
    comparison = comparison.sort_values("Rank_mean_fold_ROC_AUC")
    log(f"Raw best model: {raw_best_model}; locked model by one-SE rule: {locked_model}; cutoff={one_se_cutoff:.4f}")

    for fold in OUTER_FOLDS:
        source = models_dir / locked_model / f"{locked_model}_{fold}.joblib"
        shutil.copy2(source, locked_models_dir / source.name)

    candidate_locked_fold = candidate_all.loc[candidate_all["Model"] == locked_model].copy()
    candidate_locked_fold["Locked_model"] = locked_model
    candidate_locked_fold = candidate_locked_fold.sort_values(["Gene_ID_v5", "Validation_fold"])

    candidate_summary_rows = []
    for gene, part in candidate_locked_fold.groupby("Gene_ID_v5", sort=False):
        probabilities = part["Predicted_probability"].to_numpy(dtype=float)
        qc = str(part["Feature_QC"].iloc[0])
        if "MISSING_TRANSCRIPTOME" in qc and "MISSING_TRANSLATOME" in qc:
            evidence_flag = "LOW_FEATURE_COVERAGE"
        elif "OUTSIDE_PPI_NETWORK" in qc or "NO_REACHABLE_FT_SEED" in qc:
            evidence_flag = "REDUCED_NETWORK_EVIDENCE"
        else:
            evidence_flag = "COMPLETE_FEATURES"
        row = {
            "Gene_ID_v5": gene,
            "Gene_ID_v4": part["Gene_ID_v4"].iloc[0],
            "Gene_symbol": part["Gene_symbol"].iloc[0],
            "Locked_model": locked_model,
            "Mean_probability": float(np.mean(probabilities)),
            "SD_probability": float(np.std(probabilities, ddof=1)),
            "Median_probability": float(np.median(probabilities)),
            "Min_probability": float(np.min(probabilities)),
            "Max_probability": float(np.max(probabilities)),
            "N_folds_probability_ge_0.5": int(np.sum(probabilities >= THRESHOLD)),
            "Feature_QC": qc,
            "ML_evidence_flag": evidence_flag,
        }
        for fold in OUTER_FOLDS:
            row[f"Probability_{fold}"] = float(part.loc[part["Validation_fold"] == fold, "Predicted_probability"].iloc[0])
        candidate_summary_rows.append(row)
    candidate_summary = pd.DataFrame(candidate_summary_rows).sort_values("Mean_probability", ascending=False)
    candidate_summary.insert(0, "Prediction_rank", np.arange(1, len(candidate_summary) + 1))

    all_model_summary = (
        candidate_all.groupby(["Model", "Model_display", "Gene_ID_v5", "Gene_ID_v4", "Gene_symbol", "Feature_QC"], dropna=False)["Predicted_probability"]
        .agg(Mean_probability="mean", SD_probability="std", Min_probability="min", Max_probability="max")
        .reset_index()
        .sort_values(["Model", "Mean_probability"], ascending=[True, False])
    )

    calibration_rows = []
    for model_name in MODEL_ORDER:
        part = oof.loc[oof["Model"] == model_name]
        observed, predicted = calibration_curve(
            part["Label"], part["Predicted_probability"], n_bins=5, strategy="quantile"
        )
        for bin_index, (mean_prediction, observed_fraction) in enumerate(zip(predicted, observed), start=1):
            calibration_rows.append(
                {
                    "Model": model_name,
                    "Calibration_bin": bin_index,
                    "Mean_predicted_probability": float(mean_prediction),
                    "Observed_positive_fraction": float(observed_fraction),
                }
            )
    calibration = pd.DataFrame(calibration_rows)

    log("Computing locked-model permutation importance on outer validation folds")
    permutation_rows = []
    local_contribution_rows = []
    for fold_index, outer_fold in enumerate(OUTER_FOLDS, start=1):
        estimator = fitted_models[(locked_model, outer_fold)]
        fold_data = train.loc[train["Validation_fold"] == outer_fold].copy()
        outer_train = fold_data.loc[fold_data["Use_as_training_sample_in_fold"] == 1].copy()
        outer_valid = fold_data.loc[fold_data["Use_as_validation_sample_in_fold"] == 1].copy()
        candidate_fold = candidate.loc[candidate["Validation_fold"] == outer_fold].copy()
        X_valid = outer_valid[feature_columns]
        y_valid = outer_valid["Label"].astype(int).to_numpy()
        importance = permutation_importance(
            estimator,
            X_valid,
            y_valid,
            scoring="roc_auc",
            n_repeats=args.permutation_repeats,
            random_state=RANDOM_SEED + fold_index,
            n_jobs=args.n_jobs,
        )
        for feature_index, feature in enumerate(feature_columns):
            permutation_rows.append(
                {
                    "Feature": feature,
                    "Feature_block": feature_block(feature),
                    "Outer_fold": outer_fold,
                    "Mean_permutation_importance": float(importance.importances_mean[feature_index]),
                    "SD_permutation_repeats": float(importance.importances_std[feature_index]),
                }
            )

        training_medians = outer_train[feature_columns].median(axis=0, skipna=True).fillna(0.0)
        for _, candidate_row in candidate_fold.iterrows():
            original = candidate_row[feature_columns].to_frame().T.astype(float)
            base_probability = float(estimator.predict_proba(original)[:, 1][0])
            for feature in feature_columns:
                perturbed = original.copy()
                perturbed.loc[:, feature] = float(training_medians[feature])
                perturbed_probability = float(estimator.predict_proba(perturbed)[:, 1][0])
                local_contribution_rows.append(
                    {
                        "Gene_ID_v5": candidate_row["Gene_ID_v5"],
                        "Validation_fold": outer_fold,
                        "Locked_model": locked_model,
                        "Feature": feature,
                        "Feature_block": feature_block(feature),
                        "Original_probability": base_probability,
                        "Probability_with_feature_replaced_by_training_median": perturbed_probability,
                        "Local_ablation_delta": base_probability - perturbed_probability,
                        "Absolute_local_ablation_delta": abs(base_probability - perturbed_probability),
                    }
                )
        log(f"  Importance and candidate contributions complete for {outer_fold}")

    permutation_fold = pd.DataFrame(permutation_rows)
    global_importance = (
        permutation_fold.groupby(["Feature", "Feature_block"])["Mean_permutation_importance"]
        .agg(Mean_permutation_importance="mean", SD_across_outer_folds="std", Min_outer_fold="min", Max_outer_fold="max")
        .reset_index()
        .sort_values("Mean_permutation_importance", ascending=False)
    )
    global_importance.insert(0, "Importance_rank", np.arange(1, len(global_importance) + 1))

    local_fold = pd.DataFrame(local_contribution_rows)
    local_aggregate = (
        local_fold.groupby(["Gene_ID_v5", "Locked_model", "Feature", "Feature_block"])
        .agg(
            Mean_local_ablation_delta=("Local_ablation_delta", "mean"),
            SD_local_ablation_delta=("Local_ablation_delta", "std"),
            Mean_absolute_local_ablation_delta=("Absolute_local_ablation_delta", "mean"),
            Max_absolute_local_ablation_delta=("Absolute_local_ablation_delta", "max"),
        )
        .reset_index()
    )
    local_aggregate["Within_gene_importance_rank"] = local_aggregate.groupby("Gene_ID_v5")["Mean_absolute_local_ablation_delta"].rank(method="min", ascending=False).astype(int)
    local_aggregate = local_aggregate.sort_values(["Gene_ID_v5", "Within_gene_importance_rank"])
    block_importance = (
        local_aggregate.groupby(["Gene_ID_v5", "Locked_model", "Feature_block"])["Mean_absolute_local_ablation_delta"]
        .agg(Mean_absolute_feature_delta="mean", Sum_absolute_feature_delta="sum", N_features="count")
        .reset_index()
        .sort_values(["Gene_ID_v5", "Sum_absolute_feature_delta"], ascending=[True, False])
    )

    fold_metrics.to_csv(output_dir / "FT_outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "FT_model_comparison.csv", index=False, encoding="utf-8-sig")
    oof.sort_values(["Model", "Validation_fold", "Gene_ID_v5"]).to_csv(output_dir / "FT_oof_predictions.csv", index=False, encoding="utf-8-sig")
    inner_results.to_csv(output_dir / "FT_inner_cv_results.csv", index=False, encoding="utf-8-sig")
    best_params.to_csv(output_dir / "FT_best_hyperparameters.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(output_dir / "FT_calibration_curve.csv", index=False, encoding="utf-8-sig")
    candidate_all.sort_values(["Model", "Gene_ID_v5", "Validation_fold"]).to_csv(output_dir / "candidate9_all_models_fold_predictions.csv", index=False, encoding="utf-8-sig")
    all_model_summary.to_csv(output_dir / "candidate9_all_model_summary.csv", index=False, encoding="utf-8-sig")
    candidate_locked_fold.to_csv(output_dir / "candidate9_locked_model_fold_predictions.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(output_dir / "candidate9_predictions.csv", index=False, encoding="utf-8-sig")
    permutation_fold.to_csv(output_dir / "FT_locked_model_permutation_importance_by_fold.csv", index=False, encoding="utf-8-sig")
    global_importance.to_csv(output_dir / "FT_locked_model_global_feature_importance.csv", index=False, encoding="utf-8-sig")
    local_fold.to_csv(output_dir / "candidate9_feature_contributions_by_fold.csv", index=False, encoding="utf-8-sig")
    local_aggregate.to_csv(output_dir / "candidate9_feature_contributions.csv", index=False, encoding="utf-8-sig")
    block_importance.to_csv(output_dir / "candidate9_feature_block_contributions.csv", index=False, encoding="utf-8-sig")

    write_figures(oof, fold_metrics, comparison, candidate_summary, global_importance, output_dir)

    selected_row = comparison.loc[comparison["Model"] == locked_model].iloc[0]
    report_lines = [
        "# Maize flowering-time six-model nested cross-validation report",
        "",
        f"- Analysis script version: {__version__}",
        f"- Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Input training matrix: {train_path}",
        f"- Candidate matrix: {candidate_path}",
        "- Independent training genes: 130 (65 positive, 65 negative)",
        "- Machine-learning features: 83",
        "- Outer validation: fixed F1-F5; 104 training and 26 validation genes per fold",
        "- Inner tuning: four preassigned gene groups within each outer training set",
        "- Selection metric: mean outer-fold ROC-AUC",
        "- Locking rule: one-standard-error rule, choosing the simplest eligible model",
        f"- Raw best model: {raw_best_model}",
        f"- Locked model: {locked_model}",
        f"- Locked model pooled OOF ROC-AUC: {selected_row['Pooled_OOF_ROC_AUC']:.4f}",
        f"- Locked model pooled OOF PR-AUC: {selected_row['Pooled_OOF_PR_AUC']:.4f}",
        f"- Locked model pooled OOF F1 at 0.5: {selected_row['Pooled_OOF_F1_at_0.5']:.4f}",
        f"- Locked model pooled OOF MCC at 0.5: {selected_row['Pooled_OOF_MCC_at_0.5']:.4f}",
        f"- Locked model Brier score: {selected_row['Pooled_OOF_Brier_score']:.4f}",
        "",
        "## Important interpretation",
        "",
        "The model predicts similarity to known maize flowering-time genes, not causal probability or a proven molecular mechanism.",
        "Candidate local contributions are model-agnostic ablation deltas: each feature was replaced by its outer-training-fold median. They are not additive SHAP values.",
        "Six candidate genes lack current PPI/shortest-distance evidence, and one also lacks transcriptome/translatome measurements; Feature_QC and ML_evidence_flag must accompany probabilities.",
        "",
        "## Leakage control and limitation",
        "",
        "Outer validation genes are evaluated with their precomputed fold-specific feature rows and never enter fitting, imputation, scaling, or hyperparameter tuning.",
        "The current matrix contains outer-fold-specific SD summaries. Inner CV tunes hyperparameters using those outer-fold features; fully nested SD recomputation for every inner split was not available in the supplied matrix. Outer-fold performance remains the primary unbiased evaluation.",
    ]
    (output_dir / "FT_model_training_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_manifest = {
        "status": "PASS",
        "analysis_script_version": __version__,
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": time.time() - started,
        "random_seed": RANDOM_SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_cv_groups_per_outer_fold": 4,
        "threshold": THRESHOLD,
        "bootstrap_repeats": args.bootstrap_repeats,
        "permutation_repeats": args.permutation_repeats,
        "n_jobs": args.n_jobs,
        "command_line": sys.argv,
        "model_order_simplest_to_most_complex": MODEL_ORDER,
        "raw_best_model": raw_best_model,
        "one_se_cutoff": one_se_cutoff,
        "locked_model": locked_model,
        "training_rows": len(train),
        "training_unique_genes": int(train["Gene_ID_v5"].nunique()),
        "candidate_rows": len(candidate),
        "candidate_unique_genes": int(candidate["Gene_ID_v5"].nunique()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "input_sha256": {
            str(train_path): sha256_file(train_path),
            str(candidate_path): sha256_file(candidate_path),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    (output_dir / "FT_training_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, default=json_safe), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output_dir / "FT_nested_six_model_training.py")

    manifest_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "file_manifest_sha256.csv":
            manifest_rows.append(
                {
                    "Relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                    "Size_bytes": path.stat().st_size,
                    "SHA256": sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "file_manifest_sha256.csv", index=False, encoding="utf-8-sig")

    log(f"Training and outputs complete in {time.time() - started:.1f}s")
    log(f"Locked model: {locked_model}")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
