# MaizeFT-ML

*Machine learning for maize flowering-time candidate gene prioritization, with network-based mechanism inference.*

Reproduction materials for maize flowering-time candidate-gene prioritization
and public-network mechanism inference (Fig. 6 and four supplementary figures).

## Setup

Use Python 3.12. Install the pinned dependencies with
`python -m pip install -r requirements.txt`. Install Arial to match the
publication typography. Run the commands below from the repository root.

```sh
python reproduce.py verify
python reproduce.py redraw
python reproduce.py analyze
python reproduce.py check
```

- `verify` checks the distributed files against SHA-256 hashes.
- `redraw` regenerates the five figures from the exact archived plotting
  tables in `figure_data/`. This route does not refit models or unpack networks.
- `analyze` restores the eleven mechanism input TSVs byte for byte and reruns
  the two analysis modules. Allow several GB of free disk space and sufficient
  memory for the network analysis. Use a clean checkout for a second full run.
- `check` compares selected recalculated result tables with the archived
  tables; see the implementation for the fields and tolerances.

The two routes serve distinct purposes. The archived figure data preserve
manuscript-specific selections and formatting inputs, including the Fig. 6e/f
tables. Redrawing uses these snapshots, not newly generated analysis files.
No NAM, GWAS, QTL, haplotype or qRT-PCR data are required.

## Contents

- `analysis/`: six-model evaluation, candidate prediction and public-network
  mechanism inference, with a repository-relative configuration.
- `inputs/`: exact training and candidate matrices, label and candidate
  tables, and the eleven frozen mechanism inputs stored as gzip chunks.
  Chunks are ordinary Git files below 25 MB; do not edit or concatenate them
  manually. Analysis expects the uncompressed original TSV contents.
- `plotting/`: final manuscript plotting scripts and the alignment helper.
- `figure_data/`: values used for Fig. 6a–f and the four supplementary figures.
- `outputs/`: locally generated results; excluded from Git.

The training matrix contains 650 rows representing 130 independent genes
viewed across five outer folds; the prediction matrix has 45 rows representing
nine candidate genes. Scores support prioritization or within-gene mechanism
hypotheses and do not establish causation.

Public maize multi-omics inputs build on Han et al., *Nature Genetics*,
doi:10.1038/s41588-022-01262-1. See the mechanism documentation and input
tables for provenance. Before public release, confirm redistribution terms
for third-party input snapshots and add the authors' citation and license
information; no redistribution license is asserted by this archive.
