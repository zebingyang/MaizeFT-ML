# Verification record

The publication package was verified on 2026-08-27 with Python 3.12.13, NumPy 2.3.5, pandas 3.0.1, Matplotlib 3.11.1, and openpyxl 3.1.5.

## Checks completed

- All portable Python scripts compiled and returned a valid `--help` interface.
- All 11 input files passed schema and SHA-256 inventory checks.
- Full streaming validation of the 820,836,285-byte multilayer edge table passed with no invalid v5 IDs, negative weights, or all-zero edge rows.
- The complete pipeline ran from the 11 inputs through RWR inference, standardization, figures, Excel export, and output validation.
- Generated dimensions: 9 candidate summaries, 54 L1 score rows, 207 L2 descriptive rows, 45 component rows, 9 top-path rows, and 17 direct-evidence rows.
- Prediction status counts: 6 `NETWORK_RANKED`, 2 `DESCRIPTIVE_ONLY`, and 1 `UNRESOLVED`.
- The generated workbook reopened successfully and contained all 13 expected sheets.
- All expected PNG, PDF, SVG, and TIFF main figures plus nine individual candidate profiles were non-empty.
- Reproduced and previous results agreed exactly for candidate status, top L1 mechanism, primary displayed score, top-versus-runner-up margin, and layer-weight sensitivity stability.

Machine-readable equivalence results are stored in `REPRODUCIBILITY_TEST_REPORT.json`. Exact numerical equality depends on using the archived input files whose checksums are recorded by the pipeline.

