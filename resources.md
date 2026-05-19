# Resources Catalog

**Project**: Discovering Novel Biological Mechanisms from Protein Language Models

## Summary

| Resource | Count | Total size |
|---|---|---|
| Papers (PDFs) | 20 | 154 MB |
| Datasets | 5 directories, 3 datasets partly downloaded | 23 MB committed (≈14 MB Swiss-Prot human, ≈19 MB ProteinGym) |
| Code repositories | 6 | 130 MB |

All papers fetched via arXiv. Pretrained InterPLM SAEs (the most valuable single artifact) are not
in this repo but are confirmed accessible at `Elana/InterPLM-esm2-8m` on HuggingFace.

## Papers

20 PDFs in `papers/`. See `papers/README.md` for detailed descriptions.

| File | First author / year | arXiv ID | Cit | Role |
|---|---|---|---|---|
| InterPLM_…SAE.pdf | Simon & Zou, 2024 | 2412.12101 | 92 | **Core** — methodology + pretrained SAEs |
| Sparse_Autoencoders_LowN_…Prediction.pdf | Tsui, 2025 | 2508.18567 | — | Downstream-task protocol |
| Interpreting_Steering_…SAEs.pdf | Villegas Garcia, 2025 | 2502.09135 | 16 | Layer selection |
| ProtSAE_…Semantically.pdf | 2025 | 2509.05309 | 2 | Semantically-guided SAE |
| Mechanistic_Interpretability_Antibody_LMs_…SAEs.pdf | Haque, 2025/26 | 2512.05794 | 1 | TopK vs Ordered SAE |
| Automated_Protein_Motif_Localization_CAVs.pdf | Shamail & McWhite, 2025 | 2511.21614 | — | CAV alternative |
| Hierarchical_Semantics…SAE.pdf | 2025 | 2506.01197 | — | Matryoshka SAEs |
| Protein_Circuit_Tracing_…Transcoders.pdf | 2026 | 2602.12026 | 0 | Cross-layer circuits |
| Induction_Meets_Biology_…PLMs.pdf | 2026 | 2602.23179 | 0 | Circuit discovery example |
| Automated_Neuron_Labelling_…PLMs.pdf | Parsan, 2025 | 2507.06458 | 0 | LLM-based labelling |
| Towards_Interpretable_Protein_Structure_Prediction_…SAEs.pdf | 2025 | 2503.08764 | 11 | SAE on ESMFold |
| BERTology_Meets_Biology_…PLMs.pdf | Vig, 2020 | 2006.15222 | 359 | Attention-based seminal |
| Insights_Inner_Workings_…Protein_Function.pdf | Stark, 2023 | 2309.03631 | 16 | Integrated gradients |
| ProtTrans_Cracking_Language_Life_…Learning.pdf | Elnaggar, 2020 | 2007.06225 | 1276 | Alt PLM family |
| Tranception_…Prediction.pdf | Notin, 2022 | 2205.13760 | — | Fitness benchmark |
| Fine_Tuning_ESM2_Missense_…Impact.pdf | 2024 | 2410.10919 | — | Fine-tuning recipe |
| Protein_Generation_…Motif_Diversification.pdf | 2025 | 2510.18790 | — | Generative motif design |
| Central_Dogma_Transformer_III_…Protein.pdf | 2026 | 2603.23361 | — | DNA/RNA/protein multimodal |
| Interpretability_without_Actionability_…pdf | 2026 | 2603.18353 | — | Critical perspective |
| Continuous_Sparse_Activations_PLMs.pdf | 2025 | 2502.07154 | — | Alt sparse activation |

## Datasets

5 subdirectories in `datasets/`. Large data files excluded by `datasets/.gitignore`; README files
in each subdirectory have full download instructions.

| Name | Source | Size locally | Format | Status | Notes |
|---|---|---|---|---|---|
| swissprot/ | UniProtKB-reviewed | 13 MB | FASTA + 1 JSON example | **Downloaded** (full human proteome 20,442 seqs) | Primary evaluation set |
| proteingym/ | OATML-Markslab on HuggingFace | 19 MB | CSV | **Partial** (3/217 assays + reference CSV) | Downstream fitness validation |
| pfam_motifs/ | EBI Pfam | 0 (docs only) | TSV / HMM | Download instructions | For motif/domain ground truth |
| uniref50/ | UniProt | 0 (docs only) | FASTA | Download instructions | SAE training corpus (only if retraining) |
| alphafold/ | EBI AlphaFold DB | 0 (docs only) | PDB / mmCIF | Download instructions | 3D structures for spatial-clustering analysis |

See `datasets/README.md` and each subdirectory's README for download commands.

## Code Repositories

6 repositories in `code/`. See `code/README.md` for descriptions.

| Name | URL | Purpose | Size | Notes |
|---|---|---|---|---|
| interPLM/ | github.com/ElanaPearl/interPLM | **Primary** SAE pipeline + pretrained SAEs | 844 KB | Pretrained weights downloaded on demand from HF |
| LowNSAE/ | github.com/amirgroup-codes/LowNSAE | SAE → fitness prediction | 59 MB | Includes vendored `interprot/` |
| sparsify/ | github.com/EleutherAI/sparsify | General SAE training | 312 KB | TopK + Ordered SAE impls |
| SAELens/ | github.com/jbloomAus/SAELens | NLP-mechinterp SAE framework | 4.6 MB | Mostly NLP, but architectures transfer |
| esm/ | github.com/facebookresearch/esm | ESM-2 reference impl | 33 MB | Models also available via HF |
| ProtTrans/ | github.com/agemagician/ProtTrans | Alt PLM family | 33 MB | Cross-model universality |

Crucially, the **InterPLM pretrained SAEs are available on HuggingFace** at
`Elana/InterPLM-esm2-8m` (layers 1–6) and `Elana/InterPLM-esm2-650m` (layers 1, 9, 18, 24, 30, 33).
These checkpoints save us the ~weeks of training that would otherwise be needed and are loaded via:
```python
from interplm.sae.inference import load_sae_from_hf
sae = load_sae_from_hf(plm_model="esm2-8m", plm_layer=4)
```

## Resource Gathering Notes

### Search strategy
1. **paper-finder service** (local API at port 8000) in diligent mode — 175 candidates returned,
   relevance-ranked, with abstracts. This drove the bulk of the literature selection.
2. **arXiv API** for additional foundational papers (ProtTrans, Tranception, BERTology Meets
   Biology) where paper-finder returned only the Semantic Scholar landing page.
3. **HuggingFace API** for verifying that pretrained SAEs exist before relying on them.

### Selection criteria
- All relevance-3 papers from paper-finder that fit the topic (SAE / mech-interp / PLMs)
  were prioritized for download.
- Foundational PLM papers (ProtTrans, ESM, BERTology) were included for context.
- "Caveat" papers (Interpretability_without_Actionability, ProtSAE) were included to stress-test
  the research hypothesis.
- 2026 papers were prioritized for state-of-the-art (Protein Circuit Tracing, Induction Meets
  Biology).
- Topic-tangential papers (image enhancement, classification with no protein content) were
  excluded.

### Challenges encountered
- arXiv API rate-limited after ~25 queries; recovered by spacing requests and using the cached
  paper-finder results.
- **Sparse_autoencoders_uncover_biologically_interpretable_features (Adams et al. 2025, PNAS)**:
  paywalled and PMC PDF blocked. Abstract captured from paper-finder output. The InterPLM paper
  covers the same ground.
- **From_Mechanistic_Interpretability_to_Mechanistic_Biology** (Bricken et al.): bioRxiv-only and
  PDF retrieval blocked. Not critical — InterPLM and Antibody-SAE cover the relevant material.
- **marks.hms.harvard.edu SSL cert error** when fetching ProteinGym; routed through HuggingFace
  instead.

### Gaps and workarounds
- **No AlphaFold structures downloaded** — they are needed for spatial-clustering analysis but
  are large (10 GB for human proteome). Workaround: download instructions in
  `datasets/alphafold/README.md`; the experiment runner can pull per-protein on demand using only
  the ~100 maximally-activating proteins per SAE feature.
- **No SAELens-Antibody integration tested** — would require GPU and antibody-specific dataset.
  Deferred to experiment runner.
- **MotifAE code repository not located** — the MotifAE paper (cited in CAV paper) is mentioned
  but not yet on GitHub. Workaround: the InterPLM/LowNSAE code achieves the same goals.

## Recommendations for Experiment Design

Based on the gathered resources, the recommended experimental flow is:

### Phase A — Sanity reproduction (1–2 days of compute)
1. **Primary dataset**: Swiss-Prot human-reviewed proteome (`datasets/swissprot/full_human_proteome.fasta`),
   already downloaded.
2. Load **InterPLM's pretrained ESM-2-8M layer-4 SAE** via `interplm.sae.inference.load_sae_from_hf`.
3. Reproduce the feature-vs-concept F1 evaluation (InterPLM Figure 3, methodology in `papers/pages/`
   chunk 5). Expect ~2,000 features at F1 > 0.5 against the 14 quantitatively-evaluated Swiss-Prot
   feature types.

### Phase B — Dark feature characterization (3–7 days)
4. **Filter to dark features**: F1 < 0.2 across all Swiss-Prot concepts.
5. For each dark feature, compute:
   - Maximally activating proteins (top 100).
   - Spatial vs sequential clustering Cohen's d (needs AlphaFold structures; download on demand).
   - UMAP cluster of decoder weights.
6. **Discard noise**: drop features whose top-100 activating proteins share <2 common functional
   annotations and whose spatial Cohen's d < 0.3.

### Phase C — Causal validation (1–2 weeks)
7. Pick top 3–5 surviving dark features. For each:
   - **In-silico ablation**: zero the feature on its top-activating proteins; compute ESM-2 logits
     and structural changes via ESMFold. Significant change = causal role.
   - **Fitness correlation**: for proteins overlapping ProteinGym DMS assays, check
     whether residues with high feature activation have above-average |DMS_score|.
8. **Cross-model universality**: replicate the analysis on ESM-2-650M (different layers) using
   `Elana/InterPLM-esm2-650m`. Universal features are stronger candidates.

### Phase D — Annotation and writeup (3–5 days)
9. **LLM-assisted annotation**: feed top-activating sequence windows + AlphaFold structural
   context to Claude-Opus; ask for biological hypothesis. Validate with literature search.
10. Identify 1–2 features that pass causal validation **and** generate a coherent biological
    hypothesis **and** are universal across model sizes. Write up.

### Baselines to compare against
- **Raw ESM-2 neurons** (no SAE).
- **Random-weight ESM-2 + SAE** (null model).
- **ProtT5 SAE** (if time): cross-model validation.

### Evaluation metrics
- **Primary**: number of dark features that pass all three validation criteria above (causal,
  functional, universal).
- **Secondary**: % of dark features classifiable into a candidate biological category by Claude
  + literature search.
- **Tertiary**: ROC-AUC for variant-effect prediction using dark feature activations as the only
  predictor.

### Code to adapt/reuse
- `code/interPLM/scripts/evaluate_sae.py` — direct reuse for Phase A.
- `code/interPLM/scripts/extract_embeddings.py` — direct reuse.
- `code/interPLM/interplm/analysis/` — feature-clustering, structural-proximity primitives.
- `code/LowNSAE/linear_probe/` — for Phase C fitness-correlation experiments.
