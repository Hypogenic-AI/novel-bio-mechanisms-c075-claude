# Datasets

This directory holds datasets used in the research project **Discovering Novel Biological
Mechanisms from Protein Language Models**. Data files themselves are excluded from git via
`.gitignore` because most are large (MB–GB). Small samples and full download instructions are kept
in git so the experiment runner (or any collaborator) can reproduce the data state locally.

## Subdirectories

| Directory | Purpose | Status |
|---|---|---|
| `swissprot/` | UniProtKB-reviewed sequences + per-residue annotations (binding sites, active sites, disulfide bonds, secondary structure, etc.). **Primary evaluation set** for SAE-feature-to-biology mapping. | Full human proteome downloaded (~13 MB). 50-seq sample + cytochrome c JSON in git. |
| `proteingym/` | 217 deep mutational scanning (DMS) assays for fitness extrapolation. **Downstream validation set** — do SAE features predict variant effects better than raw embeddings? | Reference CSV + 3 DMS assays downloaded (~19 MB total). |
| `pfam_motifs/` | Pfam family/motif definitions for ground-truth domain calls. | Download instructions only. UniProt's `xref_pfam` field gives per-protein domains for free. |
| `uniref50/` | ESM-2 pretraining corpus. Only needed if retraining SAEs. | Download instructions only. |
| `alphafold/` | AlphaFold-DB 3D structures, for the 6 Å structural-proximity analysis used by InterPLM. | Download instructions only — pull per-protein on demand. |

## Quick start for the experiment runner

For a first pass, the minimal viable dataset set is just **Swiss-Prot**:

```bash
# Already downloaded; just verify
ls datasets/swissprot/full_human_proteome.fasta  # 20442 sequences, ~13 MB
```

That alone is enough to:
1. Generate ESM-2 embeddings for ~20K sequences.
2. Train a small SAE.
3. Map SAE features to Swiss-Prot annotations (binding sites, helices, etc.) using the InterPLM
   F1 methodology.
4. Identify SAE features with *no* significant alignment to any Swiss-Prot concept — these are
   our candidates for novel mechanisms.

For deeper analyses (steering experiments, fitness prediction, structural proximity), follow the
README in each subdirectory.

## Storage cost summary

| Dataset | Subset used here | Storage |
|---|---|---|
| Swiss-Prot human (FASTA, no annotations) | All 20,442 reviewed human proteins | 13 MB |
| Swiss-Prot global (FASTA, full) | All ~570K reviewed | 280 MB |
| ProteinGym substitutions | 3 of 217 assays | 19 MB |
| ProteinGym substitutions | All 217 | ~1 GB |
| AlphaFold human proteome (PDB) | All ~20K human structures | ~10 GB |

Everything in this directory after `git status` should be either documentation or the small
samples files — the large data files are gitignored.
