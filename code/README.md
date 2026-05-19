# Code Repositories

External code cloned for this research project. Each subdirectory is a separate upstream
repository; see each repo's own LICENSE.

## Repositories

### `interPLM/` — Simon & Zou 2024 (Stanford) — **PRIMARY toolkit**

- **Upstream**: <https://github.com/ElanaPearl/interPLM>
- **Paper**: papers/InterPLM_Discovering_Interpretable_Features_in_PLMs_via_SAE.pdf
- **Why it matters**: This is *the* reference implementation of the research hypothesis. It
  contains end-to-end pipelines for: (a) extracting ESM-2 embeddings, (b) training SAEs, (c)
  mapping SAE features to Swiss-Prot concepts via F1-score evaluation, (d) automated LLM-based
  feature description, and (e) steering experiments. **Pretrained SAEs for every ESM-2-8M layer
  (1–6) and ESM-2-650M layers (1, 9, 18, 24, 30, 33) are released on HuggingFace** at
  `Elana/InterPLM-esm2-8m` and `Elana/InterPLM-esm2-650m`. Loading a pretrained SAE bypasses
  weeks of training:

  ```python
  from interplm.sae.inference import load_sae_from_hf
  sae = load_sae_from_hf(plm_model="esm2-8m", plm_layer=4)
  ```

- **Key entry points**:
  - `scripts/extract_embeddings.py` — ESM-2 layer-wise embedding extraction
  - `scripts/evaluate_sae.py` — SAE feature ↔ Swiss-Prot concept F1 evaluation
  - `scripts/embed_annotations.py` — Convert UniProt annotations to per-residue binary labels
  - `interplm/sae/` — SAE architecture and training code
  - `interplm/analysis/` — Feature interpretation, UMAP, clustering, structural-proximity analysis
- **Dependencies**: PyTorch, ESM (Facebook), nnsight, HuggingFace `datasets`, AlphaFold structures
  for structural analysis. Conda env in `env.yml`.

### `LowNSAE/` — Tsui, Talreja & Aghazadeh 2025 (Georgia Tech)

- **Upstream**: <https://github.com/amirgroup-codes/LowNSAE>
- **Paper**: papers/Sparse_Autoencoders_LowN_Protein_Function_Prediction.pdf
- **Why it matters**: Shows SAE latents are not just interpretable but also **better predictors
  of fitness in low-N regimes** (24–384 labeled variants). Provides the link between SAE features
  and ProteinGym downstream tasks — a natural baseline for asking "do *novel*/unannotated SAE
  features predict fitness on held-out variants?"
- **Key entry points**:
  - `sae_training/main.py` — Train SAE on fine-tuned ESM-2 embeddings
  - `linear_probe/` — Fitness prediction from SAE latents vs raw embeddings
  - `protein_engineering/main.py` — Steering predictive latents to design new variants
- **Includes**: A vendored `interprot/` subdirectory (their fork of InterPLM-style SAEs)

### `sparsify/` — EleutherAI

- **Upstream**: <https://github.com/EleutherAI/sparsify>
- **Why it matters**: General-purpose SAE training library. **The Antibody SAE paper builds on
  this directly**, so it has TopK SAE implementations + the recent Ordered/Matryoshka SAE variants.
  Useful if we want to compare SAE architectures (TopK vs Ordered vs vanilla L1) on protein
  embeddings.
- **Key entry points**:
  - `sparsify/sparse_coder.py` — TopK SAE implementation
  - `sparsify/trainer.py` — Training loop with auxiliary loss for dead-latent mitigation

### `esm/` — Facebook / Meta AI Research

- **Upstream**: <https://github.com/facebookresearch/esm>
- **Why it matters**: Reference implementation of the **ESM-2 family** (8M, 35M, 150M, 650M,
  3B, 15B params) and ESMFold. This is the model we are interpreting. Includes pretrained
  weights downloaded automatically by HuggingFace `transformers` when using `facebook/esm2_*`.
- **Key entry points**:
  - `esm/pretrained.py` — Load any ESM-2 variant
  - `examples/` — Variant effect prediction, contact prediction, inverse folding
- **Note**: For most uses, HuggingFace `transformers.EsmModel` is more convenient than this raw
  repo. But the original PyTorch implementations are useful for hooking activations at custom
  layer/position pairs.

### `ProtTrans/` — Elnaggar et al. 2020

- **Upstream**: <https://github.com/agemagician/ProtTrans>
- **Why it matters**: Alternative PLM family (ProtT5, ProtBERT, ProtAlbert, etc.). Useful for
  cross-model **universality** experiments — does the same SAE feature appear in ESM-2 and
  ProtT5? If yes, the feature is more likely to reflect biological structure than model-specific
  artifacts.

### `SAELens/` — Bloom et al.

- **Upstream**: <https://github.com/jbloomAus/SAELens>
- **Why it matters**: Industry-standard SAE training/eval framework from the NLP mech-interp
  community. Mostly NLP-focused but the architectures and training tricks are directly
  applicable to PLMs. Useful for the **Matryoshka / Hierarchical SAE** experiments (paper
  arXiv:2506.01197 we downloaded).

## Other relevant tools (not yet cloned)

- **InterPro / Pfam scanning**: `hmmer` is the canonical tool — install via apt/conda.
- **Biopython**: for PDB parsing — `pip install biopython`.
- **NNsight**: For PLM activation hooking — `pip install nnsight` (InterPLM uses this for steering).

## Sanity check — Are pretrained SAEs accessible?

```python
from huggingface_hub import list_repo_files
files = list_repo_files('Elana/InterPLM-esm2-8m', repo_type='model')
# Should list checkpoints for layers 1–6
```

## Recommended pipeline for this research project

1. **Skip SAE training** — load pretrained `Elana/InterPLM-esm2-8m` layer-4 SAE.
2. Run feature-vs-concept F1 evaluation on Swiss-Prot (using `code/interPLM/scripts/evaluate_sae.py`).
3. **Filter to features with F1 < 0.2 across all 433 Swiss-Prot concepts.**
4. For each "unannotated" feature, compute:
   - Maximally activating proteins
   - Spatial vs sequential clustering (need AlphaFold structures)
   - UMAP neighborhood to known features (gives hints about category)
5. **LLM-assisted annotation**: feed top-activating sequence windows to Claude/GPT, ask for a
   biological hypothesis.
6. **Validation**: pick top-3 candidates; check whether their activating residues correlate with
   variant effects in ProteinGym (using `code/LowNSAE/linear_probe/`).
