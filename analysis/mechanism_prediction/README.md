# Maize candidate-gene mechanism-prediction workflow

This package reproduces the **public-data mechanism-inference module** for nine maize flowering-time candidate genes. It takes 11 prepared tabular inputs, validates them, performs mechanism-specific multilayer-network propagation, integrates supporting evidence, and exports analysis-ready JSON/Excel tables plus publication figures.

The workflow intentionally does **not** read NAM GWAS, QTL, LD, haplotype, founder-allele, additive-effect, or phenotype files. Its output is an explainable public-data hypothesis that supplements, but does not replace, genetic or experimental validation.

## Reproducible analysis boundary

The primary reproducible workflow starts from the 11 files in `Mechanism_input`. This is the recommended archival boundary because the files preserve the exact database snapshots and ID mappings used in the article. Exact scripts used to assemble those inputs are retained under `provenance/upstream_builders`; some of those provenance scripts require external databases, downloaded source files, and the original artifact-rendering environment and are not called by the portable pipeline.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# Linux/macOS
.venv/bin/python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate maizegenexai-mechanism
```

## One-command reproduction

With this directory at `Machine_Learning/ML_prediction_script/mechanism_prediction` and the prepared input folder at `Machine_Learning/Mechanism_input`:

```bash
python run_pipeline.py --config config.example.json
```

For an archival run that streams and validates every multilayer-network edge:

```bash
python run_pipeline.py --config config.example.json --full-network-check
```

To use explicit locations:

```bash
python run_pipeline.py --input-dir /path/to/Mechanism_input --output-dir /path/to/new_results
```

The launcher refuses to use a non-empty output directory unless `--overwrite` is supplied. Existing unrelated files are never deleted.

## Pipeline stages

1. `src/00_validate_inputs.py` — validates file presence, schemas, identifiers, candidates, ontology classes, seed exclusions, node coverage, expression coverage, and hashes.
2. `src/01_run_mechanism_inference.py` — performs L1 mechanism-specific random walk with restart (RWR), degree-bin adjustment, evidence integration, sensitivity analysis, and explanatory shortest-path extraction.
3. `src/02_standardize_results.py` — converts the raw analysis object into stable long-format and summary tables.
4. `src/03_plot_results.py` — creates mechanism landscapes, selected association paths, and per-candidate profiles in raster and vector formats.
5. `src/04_export_results_excel.py` — exports the standardized tables and interpretation guidance to a formatted workbook.
6. `src/05_validate_outputs.py` — checks dimensions, ranks, score reconstruction, workbook sheets, figures, and creates SHA-256 manifests.

## Main outputs

- `candidate9_mechanism_results.json`: machine-readable standardized result.
- `candidate9_mechanism_results.xlsx`: 13-sheet analysis workbook.
- `Fig_mechanism_landscape.*`: main candidate-by-mechanism and evidence-component figure.
- `Fig_selected_candidate_evidence_paths.*`: selected public-network association paths.
- `Fig_candidate_mechanism_profiles_overview.*`: nine-candidate overview.
- `candidate_profiles/*`: one profile per candidate.
- `raw/mechanism_analysis_results.json`: full unrounded inference output.
- `QC/input_validation_report.json`: input schema and checksum report.
- `output_validation_report.json`: output QC report.
- `output_manifest_sha256.tsv`: output checksums.
- `pipeline_run.log`: complete command and runtime log.

## Interpretation rules

- Compare the six L1 mechanisms **within the same candidate**.
- Do not interpret the integrated support score as a probability of causality.
- Do not use these scores to rank causal genes across candidates because evidence coverage differs.
- Keep `NETWORK_RANKED`, `DESCRIPTIVE_ONLY`, and `UNRESOLVED` results separate.
- L2 labels are descriptive refinements, not separately trained or propagated predictions.
- Multilayer shortest paths are undirected functional-association explanations, not regulatory cascades.
- Only directed TF-target evidence supports a regulatory direction.
- Missing public-data coverage is not evidence that a gene lacks flowering-time function.

See `docs/METHODS.md`, `docs/INPUTS.md`, `docs/OUTPUTS.md`, and `docs/TESTING.md` for article-ready details and verification records.

## Before public deposition

Replace all placeholders in `CITATION_TEMPLATE.cff`, rename it to `CITATION.cff`, select a software license approved by the authors/institution, and archive the exact 11 input files or provide persistent accession/version metadata. Do not publish credentials, cookies, private server paths, or unpublished restricted data.
