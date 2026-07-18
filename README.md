# Discovering Novel Biological Mechanisms from Protein Language Models

A test of the hypothesis that protein language models (PLMs) contain
sparse-autoencoder-decomposable features that encode biological mechanisms not
yet catalogued by experimentalists. Using **ESM-2-8M + InterPLM's pretrained
SAE** (10,240 latents, layer 4), we run a three-axis evaluation —
statistical structure, causal ablation, and frontier-LLM-grounded annotation —
on the ~80% of SAE features that show no strong Swiss-Prot concept alignment.

## Key Findings

- **Reproduction of InterPLM**: SAE features dominate raw ESM-2-8M neurons on
  Swiss-Prot concept alignment (165 vs 29 units at F1 ≥ 0.5, **5.7×** more).
- **1,207 "structured dark" features** identified (active, max-F1<0.2, with
  activation entropy significantly below a random-residue null at p=0.05).
- **LLM-grounded annotation works strikingly well**: shown 15 sequence windows
  per feature, GPT-5 produced a confident, named, testable biological hypothesis
  for **10/10 of the top dark features**. Independent verification against an
  expanded concept list confirmed the LLM's identification of TGEKP zinc-finger
  linkers (8/10) and GPCR DRY/NPxxY motifs (2/10).
- **Causal ablation rejects "dark feature = causally important"**: across all
  30 features tested, single-feature forward-hook ablation produces *less*
  perturbation of ESM-2 MLM logits at activating residues than at matched
  controls. None pass BH-FDR < 0.05. This adds quantitative support to the
  Bricken et al. *Interpretability-without-Actionability* critique.
- **Operational takeaway**: SAEs on small PLMs are more useful as
  **annotation-refinement tools** (the model knows sub-domain motifs that
  Swiss-Prot doesn't separately mark) than as a brand-new-biology discovery
  pipeline — at least under the ReLU-SAE + ESM-2-8M + single-feature-ablation
  protocol used here.

See [REPORT.md](REPORT.md) for the full discussion, figures, and references.

## Expanded labels on Modal (no LLM)

After building local UniProt/motif labels and (optionally) activations:

```bash
source .venv/bin/activate
# Upload index, labels, activations to the Modal volume
modal run modal_app.py::sync_inputs
# GPU: F1 → dark_features → ablation → figures
modal run modal_app.py::run_pipeline
# Pull results back
modal run modal_app.py::download_results
```

`NOVEL_BIO_ROOT` on Modal points at the `/data` volume. LLM annotation is skipped.

## How to reproduce

```bash
# 1. Set up env (uses uv)
uv venv && source .venv/bin/activate
uv add torch transformers biopython scikit-learn matplotlib seaborn umap-learn \
       scipy h5py einops pyyaml safetensors anthropic openai wandb \
       hydra-core omegaconf requests \
       --index-strategy unsafe-best-match
uv pip install -e code/interPLM     # local interplm package

# 2. Run the full pipeline (in this order; each step writes to results/)
python -m src.extract_activations         # ESM-2-8M + InterPLM SAE on 1,500 Swiss-Prot proteins
python -m src.fetch_annotations           # UniProt REST per-residue concept labels
python -m src.compute_f1                  # feature × concept F1, with threshold sweep
python -m src.dark_features               # filter dark + structured, rank top
python -m src.ablation                    # causal ablation (forward hook)
python -m src.proteingym_correlation      # variant-effect fitness correlation
python -m src.recheck_dark_with_zf        # F1 against extra concepts (Zinc finger, etc.)
python -m src.llm_annotation              # GPT-5 hypotheses (uses $OPENAI_API_KEY)
python -m src.make_figures                # all figures into figures/
```

## File structure

```
.
├── README.md                  # this file
├── REPORT.md                  # full research report with results, analysis, references
├── planning.md                # Phase 0+1 motivation + experimental plan
├── pyproject.toml             # uv-managed dependencies
├── src/                       # all experiment scripts (no Jupyter notebooks)
│   ├── config.py              # global config (paths, seeds, model name)
│   ├── data.py                # FASTA loader and subset sampler
│   ├── extract_activations.py # ESM-2 forward + SAE encode → results/sae_activations.npz
│   ├── fetch_annotations.py   # UniProt REST → results/annotations.npz
│   ├── compute_f1.py          # feature × concept F1 with threshold sweep
│   ├── dark_features.py       # entropy null + structured-dark catalog
│   ├── ablation.py            # forward-hook causal ablation
│   ├── proteingym_correlation.py  # variant-effect Spearman
│   ├── recheck_dark_with_zf.py    # add Zinc finger / Topological domain etc.
│   ├── llm_annotation.py      # OpenAI/Anthropic biological-hypothesis prompt
│   └── make_figures.py        # all matplotlib figures into figures/
├── results/                   # all .npz / .json / .csv experiment outputs
├── figures/                   # all .png figures referenced in REPORT.md
├── logs/                      # tee'd stdout of each pipeline stage
├── papers/                    # 20 pre-fetched PDFs (literature review)
├── code/                      # vendored InterPLM / esm / LowNSAE / sparsify / ProtTrans
└── datasets/
    ├── swissprot/             # 20,442 human proteins (FASTA)
    └── proteingym/            # 3 DMS substitution assays
```

## Compute requirements

- **Model**: ESM-2-8M (35 MB) + InterPLM SAE (~25 MB), both via HuggingFace.
  HuggingFace cache uses ~250 MB.
- **Hardware**: 1 × NVIDIA RTX A6000 (48 GB VRAM) for the GPU steps.
  Pipeline easily fits in 8 GB VRAM with batch size 8.
- **Wall-clock**: ~25 min end-to-end (excluding LLM annotation): 1 min
  activation extraction, 8 min UniProt annotation fetch, 15 min F1 sweep,
  1 min causal ablation, <1 min ProteinGym & figures. LLM annotation adds
  ~9 min (10 features × GPT-5 chat completion).
- **API cost**: ~$0.50 for 10 GPT-5 annotations (each request: 15 windows
  × ~20 tokens of context + 1.5K-token output).

## Citation

Built on:
- **InterPLM**: Simon & Zou (2024), *Nature Methods* 2025. arXiv:2412.12101.
- **Antibody-SAE** (TopK vs Ordered): Haque et al. (2025/2026). arXiv:2512.05794.
- **ESM-2**: Lin et al. (2023), *Science*.
- **ProteinGym**: Notin et al. (2023).

Pretrained models / SAEs from:
- `facebook/esm2_t6_8M_UR50D` (HuggingFace).
- `Elana/InterPLM-esm2-8m`, layer 4 (HuggingFace).
