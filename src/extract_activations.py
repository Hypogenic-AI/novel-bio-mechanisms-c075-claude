"""Extract ESM-2 layer-4 hidden states and InterPLM SAE activations.

For each protein in our Swiss-Prot subset, we run ESM-2-8M, take the layer-4
hidden state, pass it through the pretrained InterPLM SAE encoder, and store:
  - hidden states (fp16)  -> results/hidden_states.npz (indexed by accession)
  - sparse SAE activations (CSR) -> results/sae_activations.npz

We also store an `index.json` mapping accession -> (offset, length) so we can
slice each protein later without keeping everything in RAM.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from src import config
from src.data import load_swissprot_subset


def get_hidden_state(
    sequences: List[str],
    model: EsmModel,
    tokenizer,
    layer: int,
    device: str,
    max_len: int = 512,
) -> List[np.ndarray]:
    """Run a batch through ESM and return the per-residue hidden state at `layer`.

    Returns one array per input sequence with shape (L_i, hidden_dim).
    Special tokens (CLS, EOS) and padding are stripped.
    """
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[layer]  # (B, L, D)
    attn_mask = enc["attention_mask"].bool()
    results = []
    for i, seq in enumerate(sequences):
        mask_i = attn_mask[i]  # includes CLS and EOS
        h = hidden[i][mask_i]  # (L_real, D) with CLS at idx 0 and EOS at -1
        # Strip CLS and EOS
        h = h[1:-1]
        # ESM tokenizer adds neither for empty seqs; safety:
        assert h.shape[0] == len(seq) or h.shape[0] == max_len - 2, (
            f"len mismatch: hidden {h.shape[0]} vs seq {len(seq)}"
        )
        results.append(h.cpu().float().numpy())
    return results


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading Swiss-Prot subset...")
    proteins = load_swissprot_subset()
    print(f"  Loaded {len(proteins)} proteins, lengths "
          f"{min(len(s) for _, s in proteins)}–{max(len(s) for _, s in proteins)} aa")

    print(f"[{time.strftime('%H:%M:%S')}] Loading ESM-2-8M ({config.ESM_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL)
    model = EsmModel.from_pretrained(config.ESM_MODEL, add_pooling_layer=False)
    model.eval()
    model = model.to(config.DEVICE)

    print(f"[{time.strftime('%H:%M:%S')}] Loading InterPLM SAE...")
    from interplm.sae.inference import load_sae_from_hf
    sae = load_sae_from_hf(plm_model="esm2-8m", plm_layer=config.ESM_LAYER)
    sae = sae.to(config.DEVICE)
    sae.eval()

    # Output storage
    out_dir = config.RESULTS
    index: Dict[str, Tuple[int, int]] = {}  # accession -> (start, length)
    total_residues = 0

    # We collect everything in memory (1500*200 ~= 300K residues * 320 dim hidden = ~190MB fp16,
    # plus sparse SAE which is small)
    all_hidden: List[np.ndarray] = []
    all_sparse_features: List[sparse.csr_matrix] = []

    # Batched inference
    BATCH = 8
    for start in tqdm(range(0, len(proteins), BATCH), desc="ESM+SAE"):
        batch = proteins[start:start + BATCH]
        accs, seqs = zip(*batch)
        hidden_list = get_hidden_state(
            list(seqs), model, tokenizer, layer=config.ESM_LAYER, device=config.DEVICE
        )
        for acc, seq, h in zip(accs, seqs, hidden_list):
            L = h.shape[0]
            # SAE encode  -> dense (L, dict_size) then convert to sparse
            with torch.no_grad():
                h_t = torch.from_numpy(h).to(config.DEVICE)
                feats = sae.encode(h_t)  # (L, dict_size) ReLU SAE -> sparse
                feats_np = feats.cpu().float().numpy()
            # Sparsify (most entries are zero)
            sp = sparse.csr_matrix(feats_np)
            index[acc] = (total_residues, L)
            total_residues += L
            all_hidden.append(h.astype(np.float16))
            all_sparse_features.append(sp)

    print(f"[{time.strftime('%H:%M:%S')}] Collected {total_residues:,} residues; "
          f"saving to disk...")

    # Concatenate
    hidden_all = np.concatenate(all_hidden, axis=0)
    feats_all = sparse.vstack(all_sparse_features, format="csr")
    print(f"  hidden_all: {hidden_all.shape} ({hidden_all.dtype})")
    print(f"  feats_all : {feats_all.shape} nnz={feats_all.nnz:,} "
          f"(density={feats_all.nnz/(feats_all.shape[0]*feats_all.shape[1]):.5f})")

    np.savez_compressed(out_dir / "hidden_states.npz", hidden=hidden_all)
    sparse.save_npz(out_dir / "sae_activations.npz", feats_all)
    with open(out_dir / "index.json", "w") as f:
        json.dump(
            {
                "proteins": [{"accession": acc, "sequence": seq} for acc, seq in proteins],
                "offsets": {acc: list(v) for acc, v in index.items()},
                "total_residues": total_residues,
                "hidden_dim": config.HIDDEN_DIM,
                "sae_dict_size": config.SAE_DICT_SIZE,
                "esm_model": config.ESM_MODEL,
                "esm_layer": config.ESM_LAYER,
            },
            f,
            indent=2,
        )
    print(f"[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    main()
