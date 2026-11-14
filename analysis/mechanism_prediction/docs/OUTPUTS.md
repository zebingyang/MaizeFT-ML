# Output specification

## Primary result tables

`candidate9_mechanism_results.json` and `candidate9_mechanism_results.xlsx` contain:

- candidate summary and prediction status;
- all six L1 within-gene mechanism scores;
- top-hypothesis component scores and weighted contributions;
- explanatory top paths and terminal seed genes;
- direct public evidence edges and provenance;
- top tissue/developmental contexts;
- coverage and sensitivity QC;
- mechanism ontology, input manifest, data dictionary, and method parameters.

The raw JSON under `raw/` preserves unrounded values and all L1/L2/path records.

## Figures

- `Fig_mechanism_landscape`: network-ranked and descriptive-only heatmaps are separated, with unavailable components shown explicitly.
- `Fig_selected_candidate_evidence_paths`: association paths for configured candidates; edge colors identify public network layers.
- `Fig_candidate_mechanism_profiles_overview`: compact within-candidate profiles for all queried genes.
- `candidate_profiles`: one reusable profile per candidate.

PNG/TIFF files support raster submission; PDF/SVG files support vector editing. Journal production requirements take precedence over the default 400 dpi.

## Quality-control artifacts

- `QC/input_validation_report.json`: input schemas and checksums.
- `output_validation_report.json`: expected row counts, score reconstruction, workbook and figure checks.
- `output_manifest_sha256.tsv`: relative file paths, byte sizes, and SHA-256 hashes.
- `pipeline_run.log`: exact commands and their output.

