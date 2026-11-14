# Maize flowering-time machine-learning code package

This directory separates the analysis into two reproducible layers.

## 1. Upstream feature construction

See `upstream/README_upstream_feature_construction.md` and run:

```bash
cd upstream
cp upstream_config.example.json upstream_config.json
python run_upstream_pipeline.py --config upstream_config.json
```

This layer constructs PPI/network features, leakage-controlled fold-specific
shortest-distance features, 31-tissue transcriptome features, 21-tissue
translatome features, the 130-gene training matrix and the nine-gene prediction
matrix. Its direct model-ready outputs are:

- `FT_training_feature_matrix.csv`
- `candidate9_feature_matrix.csv`

## 2. Nested six-model fitting and prediction

See `README_FT_six_model_analysis.md` and run
`FT_nested_six_model_training.py` after the two matrices above have been placed
in the configured input directory.

This layer performs nested five-fold evaluation of L2 logistic regression,
Elastic Net logistic regression, linear SVM, bagging, shallow XGBoost and a
shallow neural network; generates out-of-fold metrics and calibration results;
locks the final model; predicts the nine candidates; and produces contribution
analysis and publication figures.

## Reproducibility rule

The 650 rows of the training matrix are five fold-specific feature views of 130
unique genes. They must never be randomly split as 650 independent samples.
Preprocessing and hyperparameter tuning must remain inside the training portion
of each outer fold. Positive seed distances must be recomputed with the current
validation-fold positive seeds inactive.

The curated positive catalogue and operational negative-control definitions are
versioned research data. Deposit them with row-wise provenance; do not describe
them as an automatically generated truth set.
