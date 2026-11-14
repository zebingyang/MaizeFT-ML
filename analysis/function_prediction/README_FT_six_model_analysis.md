# Maize flowering-time six-model nested cross-validation analysis

## Purpose

`FT_nested_six_model_training.py` reproduces the machine-learning analysis used
to prioritize nine maize flowering-time candidate genes from multi-omics
features. It trains and evaluates six classifiers, creates out-of-fold (OOF)
predictions for all independent training genes, locks one model using a
prespecified one-standard-error rule, predicts the nine candidate genes, and
calculates global and candidate-level feature contributions.

The output probability represents similarity to curated maize flowering-time
genes. It is not a causal probability and does not establish a molecular
mechanism.

## Required input files

Place these files in the same directory as the script or its parent directory,
or provide their directory explicitly with `--input-dir`. The supplied project
combined-package layout keeps the code in `ML_prediction_script/function_prediction/`
and the two input matrices two levels above.

1. `FT_training_feature_matrix.csv`
   - 650 rows = 130 independent genes x five outer-fold feature views.
   - 65 positive and 65 negative genes.
   - 14 metadata columns followed by 83 numeric features.
2. `candidate9_feature_matrix.csv`
   - 45 rows = nine candidate genes x five outer-fold feature views.
   - It must have exactly the same columns and column order as the training
     matrix.

Missing numeric values may be empty or encoded as `NA`. Imputation is performed
within each outer training fold only.

## Models

Six models are compared:

1. L2-regularized logistic regression
2. Elastic Net logistic regression
3. Linear support vector machine
4. Bagged shallow decision trees
5. Shallow XGBoost
6. Shallow single-hidden-layer neural network

All preprocessing steps are contained inside scikit-learn pipelines. Median
imputation and, where applicable, standardization are fitted only on the outer
training data.

## Validation and model selection

- The outer validation folds are the fixed `F1`-`F5` assignments supplied in
  the feature matrix.
- Each outer fold contains 104 training genes and 26 validation genes, balanced
  as 52/52 and 13/13 positives/negatives, respectively.
- Hyperparameters are tuned in the outer training set using its four
  preassigned inner groups and ROC-AUC as the refit metric.
- Each independent training gene receives exactly one OOF prediction per
  model.
- Models are compared using mean outer-fold ROC-AUC. The final model is the
  simplest model whose mean ROC-AUC lies within one standard error of the raw
  best model. The prespecified simplicity order is L2 logistic regression,
  Elastic Net logistic regression, linear SVM, bagging, shallow XGBoost and
  shallow neural network.
- Pooled OOF ROC-AUC, PR-AUC, F1, MCC, balanced accuracy, Brier score, log loss
  and expected calibration error are reported. Stratified bootstrap confidence
  intervals are calculated for ROC-AUC and PR-AUC.

## Candidate prediction and feature contribution

Each candidate is predicted with each of the five fitted outer-fold models. The
primary candidate result is the mean and standard deviation of these five
probabilities.

Global feature importance is the decrease in outer-validation ROC-AUC after
feature permutation. Candidate-level contributions are model-agnostic ablation
deltas: one feature at a time is replaced by its corresponding outer-training
median, and the resulting probability change is recorded. These local values
are not SHAP values and are not additive.

## Software environment

The exact environment used for the reported analysis is listed in
`requirements_FT_six_model.txt`. Python 3.12 is recommended.

Create an isolated environment and install the dependencies:

```bash
python -m venv ft_ml_env
ft_ml_env/Scripts/python -m pip install -r requirements_FT_six_model.txt
```

On Linux or macOS, use `ft_ml_env/bin/python` instead.

## Usage

Validate the input matrices without fitting models:

```bash
python FT_nested_six_model_training.py --input-dir /path/to/data --validate-only
```

Run the complete analysis:

```bash
python FT_nested_six_model_training.py \
  --input-dir /path/to/data \
  --output-dir /path/to/results
```

The default input directory is the directory containing the script. If both
required matrices are absent there but present one level above, the parent
directory is selected automatically. If `--output-dir` is omitted, outputs are written to
`FT_six_model_results_reproduced` under the input directory. The script refuses
to write into a nonempty output directory unless `--allow-nonempty-output` is
specified; it never deletes stale files.

Useful optional arguments:

```text
--training-file FILE
--candidate-file FILE
--n-jobs INTEGER
--bootstrap-repeats INTEGER
--permutation-repeats INTEGER
```

Use the default 1,000 bootstrap and 50 permutation repeats for the reported
analysis.

For the supplied Windows project layout, the following command can be run from
`F:\GWAS2_result\Machine_Learning\ML_prediction_script\function_prediction`:

```powershell
python .\FT_nested_six_model_training.py `
  --input-dir ..\.. `
  --output-dir ..\..\ML_prediction_results_reproduced\function_prediction
```

## Principal output files

- `FT_model_comparison.csv`: pooled and fold-level performance summaries and
  model-lock indicators.
- `FT_outer_fold_metrics.csv`: metrics for every model and outer fold.
- `FT_oof_predictions.csv`: 130 OOF predictions for each of the six models.
- `FT_best_hyperparameters.csv`: selected parameters for each model/fold.
- `FT_inner_cv_results.csv`: complete inner tuning results.
- `FT_calibration_curve.csv`: calibration-curve coordinates.
- `candidate9_predictions.csv`: primary five-fold candidate probabilities and
  feature-quality flags.
- `candidate9_all_model_summary.csv`: candidate predictions from all six
  models.
- `FT_locked_model_global_feature_importance.csv`: global permutation
  importance.
- `candidate9_feature_contributions.csv`: aggregated candidate-level feature
  ablation results.
- `candidate9_feature_block_contributions.csv`: local contributions summarized
  by PPI, shortest-distance, transcriptome and translatome blocks.
- `models_all_outer_folds/`: all 30 fitted outer-fold models.
- `models_locked/`: the five fitted models used for the primary candidate
  prediction.
- `Fig_*.png`: ROC, precision-recall, calibration, model comparison, candidate
  probability and feature-importance figures.
- `FT_training_run_manifest.json`: input hashes, environment versions, random
  seed and run metadata.
- `file_manifest_sha256.csv`: output file sizes and SHA-256 hashes.

## Reproduction target for the supplied matrices

With the exact software versions and default arguments, the reported run gives:

- raw best mean outer-fold ROC-AUC model: `ElasticNet_LR`;
- final one-SE locked model: `L2_LR`;
- locked-model pooled OOF ROC-AUC: approximately 0.8432;
- locked-model pooled OOF PR-AUC: approximately 0.8310;
- locked-model F1 at 0.5: approximately 0.7846;
- locked-model MCC at 0.5: approximately 0.5692.

Small floating-point differences may occur across operating systems or numerical
libraries, but the locked model and candidate ordering should be checked against
the supplied reference result tables.

## Important methodological limitation

The shortest-distance features in the supplied matrices were precomputed for
each outer fold. They exclude the outer validation labels, which preserves the
primary outer-fold evaluation. They were not recomputed for every inner tuning
split. Consequently, the outer-fold results are the primary performance
estimate, and this limitation should be reported in the manuscript.

## Reference

The network-feature strategy was informed by:

Han L, Zhong W, Qian J, et al. A multi-omics integrative network map of maize.
Nature Genetics. 2023;55:144-153. doi:10.1038/s41588-022-01262-1.
