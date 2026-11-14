# Mechanism-inference method

## Analysis question

For each genetically nominated candidate gene, which broad flowering-time mechanism is most strongly supported by independent public molecular evidence? The analysis is a hypothesis-prioritization procedure, not a classifier trained on NAM evidence and not an experimental causal test.

## Mechanism ontology and seeds

Known maize flowering-time genes are assigned to six L1 mechanisms: environmental/light signaling (M1), florigen/floral transition (M2), age/autonomous pathways (M3), hormone/stress response (M4), meristem/reproductive development (M5), and chromatin/RNA regulation (M6). The 23 L2 classes are used only as descriptive refinements. Seed eligibility is explicitly stored in `FT_mechanism_seed_annotations_v5.tsv`; queried candidates are excluded from eligible seeds to prevent target leakage.

## Layer-specific propagation

Each public network layer is analyzed separately. Edge weights are non-negative and the adjacency matrix is row-normalized. For mechanism-specific seed vector `s`, random walk with restart iterates

`p(t+1) = r*s + (1-r)*P^T*p(t)`

with restart probability `r=0.5`, L1 convergence tolerance `1e-11`, and maximum 200 iterations. No edge from one evidence layer is allowed to numerically dominate another before per-layer scoring.

## Degree adjustment and layer integration

Raw RWR scores are adjusted for network connectivity using empirical midrank percentiles within `floor(log2(layer_degree))` bins. A node with no seed-reachable RWR support receives percentile zero. The primary network score is the equally weighted mean of assessable ChIA-PET, RNA, Ribo, and PPI layer percentiles. Two predeclared sensitivity scenarios up-weight interaction evidence or expression-linked evidence; stability is the proportion of scenarios retaining the primary top mechanism.

## Supporting evidence and integrated score

For network-assessable candidates, the L1 integrated support score uses declared weights:

- multilayer network propagation: 0.60;
- direct TF-target evidence: 0.15;
- mechanism-relevant tissue expression: 0.15;
- exact functional-annotation support: 0.05;
- trait compatibility: 0.05.

Weights are renormalized across assessable components. ChIA-PET direct evidence is reported but is not added again because chromatin contacts are already a propagation layer. Candidates without multilayer-network propagation are explicitly labeled `DESCRIPTIVE_ONLY`; their descriptive score is not comparable with network-augmented scores. Candidates without sufficient assessable public evidence are labeled `UNRESOLVED`.

## Explanations

For the top L1 mechanism, the workflow reports evidence components, mechanism-relevant expression contexts, direct TF/functional-association edges, and the shortest unweighted network path to an eligible mechanism seed (maximum four edges). Coexpression, cotranslation, PPI, and chromatin-contact paths are undirected associations. Only a TF-target edge supplies directionality.

## Reporting language

Use “public-data evidence supports” or “was prioritized as.” Avoid “proved,” “causal probability,” or “regulatory cascade” unless independently established experimentally. Scores compare mechanisms within a candidate, not causal genes across candidates. This module contains no GWAS, QTL, LD, haplotype, founder, allelic-effect, or NAM phenotype evidence.

