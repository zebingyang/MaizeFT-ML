# Input specification

All inputs are UTF-8 tab-separated files. The portable workflow reads exactly the following 11 files from one directory.

| File | Role in inference |
|---|---|
| `candidate9_mechanism_query.tsv` | Candidate IDs, trait labels, inclusion flag, and query metadata. |
| `FT_mechanism_ontology.tsv` | Six L1 flowering-time mechanism classes and 23 descriptive L2 subclasses, including annotation terms and tissue-context rules. |
| `FT_mechanism_seed_annotations_v5.tsv` | Curated known flowering-time seed genes assigned to mechanisms; candidate genes are excluded from eligible seeds. |
| `maize_gene_ID_mapping_v5.tsv` | Stable mapping among maize v5/v4 IDs, symbols, and external identifiers. |
| `maize_FT_multilayer_network_edges_v5.tsv` | Unified undirected ChIA-PET, RNA-coexpression, Ribo-cotranslation, and PPI edges and weights. |
| `maize_TF_target_regulatory_edges_v5.tsv` | Public directed TF-to-target evidence and confidence scores. |
| `maize_chromatin_regulatory_edges_v5.tsv` | Direct chromatin-contact support; reported but not added a second time when ChIA-PET is already a propagation layer. |
| `maize_expression_sample_metadata.tsv` | Maps expression columns to tissue, developmental stage, omics type, and mechanism relevance. |
| `maize_multilayer_network_nodes_v5.tsv` | Node universe, layer degrees, and public-data coverage flags. |
| `maize_gene_function_annotations_v5.tsv` | Public functional descriptions, GO/domain/pathway/orthology annotations, and source provenance. |
| `candidate9_tissue_context_evidence.tsv` | Candidate expression percentiles and QC in mechanism-relevant tissues/stages. |

## Required archival practice

For a paper release, deposit the exact input snapshots alongside the scripts when licensing permits. Keep source accession, release/date, URL/DOI, ID-mapping policy, filters, and SHA-256 checksums. Database pages change; a URL without a frozen table is not sufficient for exact reproduction.

Run:

```bash
python src/00_validate_inputs.py --input-dir /path/to/Mechanism_input --report input_validation_report.json --full-network-check
```

The analysis must stop on missing columns, duplicated/invalid candidate IDs, missing L1 seed classes, candidate leakage into seed labels, or absent candidate-node/expression coverage.

