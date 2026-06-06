# Foundation Model Comparison Plan

Roadmap item: "Try to explore other foundation models (other protein models,
chemistry, physics, etc)."

## Contribution

This repository currently tests one narrow setting: ESM-2-8M plus a pretrained
InterPLM SAE. The negative causal-ablation result is valuable, but by itself it
does not distinguish between four possibilities:

1. The discovery hypothesis is weak.
2. The ESM-2-8M model is too small.
3. The ReLU InterPLM SAE is not the right sparse dictionary.
4. The residue-level ablation protocol is too local.

The immediate next experiment should change only one variable first: model
scale. ESM-2-650M is the best first comparison because it keeps the protein
modality, Swiss-Prot labels, ProteinGym validation, and InterPLM SAE family
mostly fixed while increasing representation capacity.

## Candidate Matrix

| key | modality | model | unit | public SAE | readiness | immediate question |
| --- | --- | --- | --- | --- | --- | --- |
| `esm2_650m` | protein | `facebook/esm2_t33_650M_UR50D` | amino-acid residue | yes | 5 | Does the dark-feature negative ablation result persist when model scale changes but the protein labels, SAE family, and ablation protocol stay comparable? |
| `prot_t5_xl_uniref50` | protein | `Rostlab/prot_t5_xl_uniref50` | amino-acid residue | no | 3 | Do dark residue-level motifs recur in a non-ESM architecture, or are they ESM/InterPLM-specific? |
| `chemberta_77m_mlm` | chemistry | `DeepChem/ChemBERTa-77M-MLM` | SMILES token or molecular substructure | no | 2 | Can an analogous dark-feature workflow recover under-annotated chemical substructures instead of protein motifs? |
| `mace_mp` | physics/materials | MACE | atom or local atomic environment | no | 1 | Can sparse features in atomistic foundation models identify local environments tied to stability, forces, or defects? |

## Runnable Smoke Path

List registered models:

```bash
python -m src.foundation_model_registry --include-baseline
```

Dry-run the highest-priority comparison without downloading weights:

```bash
python -m src.extract_foundation_embeddings \
  --model-key esm2_650m \
  --layer 33 \
  --n-proteins 25 \
  --dry-run
```

Run a small ESM-2-650M extraction with its InterPLM SAE:

```bash
python -m src.extract_foundation_embeddings \
  --model-key esm2_650m \
  --layer 33 \
  --n-proteins 25 \
  --batch-size 1 \
  --with-interplm-sae \
  --output-dir results/foundation_models/esm2_650m_layer33_smoke
```

For a full controlled comparison, use `--n-proteins 1500` and run downstream
scripts against a separate result directory:

```bash
python -m src.extract_foundation_embeddings \
  --model-key esm2_650m \
  --layer 33 \
  --n-proteins 1500 \
  --batch-size 1 \
  --with-interplm-sae \
  --output-dir results/foundation_models/esm2_650m_layer33_full

RESULTS_DIR=results/foundation_models/esm2_650m_layer33_full python -m src.fetch_annotations
RESULTS_DIR=results/foundation_models/esm2_650m_layer33_full python -m src.compute_f1
RESULTS_DIR=results/foundation_models/esm2_650m_layer33_full python -m src.dark_features
RESULTS_DIR=results/foundation_models/esm2_650m_layer33_full python -m src.ablation
RESULTS_DIR=results/foundation_models/esm2_650m_layer33_full python -m src.proteingym_correlation
```

## Evaluation Criteria

Compare the ESM-2-650M run against the original ESM-2-8M report on:

- number of concept-aligned SAE features at F1 >= 0.3, 0.4, and 0.5;
- number and entropy distribution of structured dark features;
- whether top dark features are still mostly annotation-granularity refinements;
- whether single-latent ablation becomes more disruptive at activating residues;
- ProteinGym correlation for dark, bright, and random feature pools.

## Why Chemistry And Physics Come Later

Chemistry and physics models are worth exploring, but they require a different
schema before their results are comparable. Protein models have residue spans
and UniProt concepts. Chemistry models need atom/substructure mappings and
property labels. Atomistic physics models need structural neighborhoods,
energies, forces, and perturbations that preserve valid geometries. Jumping
modalities before the protein-model comparison would mix model-family effects
with annotation-schema effects.
