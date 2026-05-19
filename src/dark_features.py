"""Identify and characterize 'dark' SAE features.

A dark feature is one whose max-F1 against any Swiss-Prot concept is below
DARK_F1_THRESHOLD. We further filter for *structured* dark features: those
that have a non-trivial number of nonzero activations and concentrate their
activations across a small number of proteins (low protein-entropy).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src import config


def feature_protein_entropy(
    feats: sparse.csr_matrix,
    protein_lengths: List[int],
    feature_idx: np.ndarray,
) -> np.ndarray:
    """For each feature, compute the entropy of its activation distribution across proteins.

    Lower entropy means the feature concentrates on fewer proteins (more interpretable).
    """
    # Build a per-residue protein assignment
    prot_ids = np.empty(sum(protein_lengths), dtype=np.int32)
    pos = 0
    for pi, L in enumerate(protein_lengths):
        prot_ids[pos:pos + L] = pi
        pos += L
    n_prot = len(protein_lengths)
    entropies = np.zeros(len(feature_idx), dtype=np.float32)
    feats_csc = feats.tocsc()
    for j, fi in enumerate(feature_idx):
        col = feats_csc.getcol(fi).toarray().ravel()
        mask = col > 0
        if mask.sum() < 5:
            entropies[j] = np.log(n_prot)  # max entropy when no signal
            continue
        # Sum activation per protein
        acts_by_prot = np.bincount(prot_ids[mask], weights=col[mask], minlength=n_prot)
        p = acts_by_prot / acts_by_prot.sum()
        p_pos = p[p > 0]
        entropies[j] = float(-(p_pos * np.log(p_pos)).sum())
    return entropies


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading F1 and activations...")
    f1_data = np.load(config.RESULTS / "f1_sae.npz")
    F1 = f1_data["F1"]  # (n_features, n_concepts)
    P = f1_data["precision"]
    R = f1_data["recall"]
    concepts = f1_data["concepts"].tolist()
    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")

    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    proteins = index["proteins"]  # list of {accession, sequence}
    offsets = index["offsets"]
    protein_lengths = [len(p["sequence"]) for p in proteins]

    n_features = F1.shape[0]
    max_f1 = F1.max(axis=1)
    nonzero_per_feat = np.asarray((feats != 0).sum(axis=0)).ravel()
    mean_act_per_feat = np.asarray(feats.sum(axis=0)).ravel() / max(feats.shape[0], 1)
    print(f"  n_features={n_features}")
    print(f"  dead (no activations): {(nonzero_per_feat == 0).sum()}")
    print(f"  active features:       {(nonzero_per_feat > 0).sum()}")

    # Identify dark features
    is_dark = (max_f1 < config.DARK_F1_THRESHOLD) & (nonzero_per_feat >= config.DARK_MIN_ACTIVE_RESIDUES)
    dark_idx = np.where(is_dark)[0]
    print(f"  dark (max_f1<{config.DARK_F1_THRESHOLD}, >=100 acts): {len(dark_idx)}")

    # Compute protein-entropy for dark features
    print(f"[{time.strftime('%H:%M:%S')}] Computing protein-entropy for {len(dark_idx)} dark features...")
    entropies = feature_protein_entropy(feats, protein_lengths, dark_idx)
    n_prot = len(proteins)
    max_entropy = float(np.log(n_prot))
    print(f"  Max possible entropy (uniform over {n_prot} proteins) = {max_entropy:.2f}")
    print(f"  Dark feature entropy quantiles: "
          f"10%={np.percentile(entropies, 10):.2f}  "
          f"50%={np.percentile(entropies, 50):.2f}  "
          f"90%={np.percentile(entropies, 90):.2f}")

    # Also compute entropy null: for each random subset of residues of comparable size,
    # what would the entropy be if they were uniform on proteins?
    rng = np.random.default_rng(config.SEED)
    n_null = 500
    null_entropies = []
    sizes = nonzero_per_feat[dark_idx]
    # Per-residue protein assignment
    prot_ids = np.empty(sum(protein_lengths), dtype=np.int32)
    pos = 0
    for pi, L in enumerate(protein_lengths):
        prot_ids[pos:pos + L] = pi
        pos += L
    for sz in rng.choice(sizes, size=min(n_null, len(sizes)), replace=False):
        if sz < 5:
            continue
        idx = rng.choice(len(prot_ids), size=int(sz), replace=False)
        counts = np.bincount(prot_ids[idx], minlength=n_prot)
        p = counts / counts.sum()
        p = p[p > 0]
        null_entropies.append(float(-(p * np.log(p)).sum()))
    print(f"  Null entropy (random residues, n={len(null_entropies)}) "
          f"median={np.median(null_entropies):.2f}, p5={np.percentile(null_entropies, 5):.2f}")

    # "Structured" dark features: entropy well below the null
    null_thresh = float(np.percentile(null_entropies, 5))
    structured = entropies < null_thresh
    structured_idx = dark_idx[structured]
    print(f"  Structured dark features (entropy < null p5={null_thresh:.2f}): "
          f"{len(structured_idx)}")

    # Rank by mean activation among the structured
    feats_csc = feats.tocsc()
    means_structured = np.zeros(len(structured_idx))
    nz_structured = np.zeros(len(structured_idx), dtype=np.int64)
    for j, fi in enumerate(structured_idx):
        col = feats_csc.getcol(fi).toarray().ravel()
        nz = col > 0
        if nz.sum() > 0:
            means_structured[j] = col[nz].mean()
            nz_structured[j] = int(nz.sum())

    # Save the top-N for downstream causal analysis
    rank_score = means_structured * np.log1p(nz_structured)
    top_order = np.argsort(-rank_score)
    n_top = min(30, len(structured_idx))
    top_idx = structured_idx[top_order[:n_top]]

    # Build per-feature top-activating protein info for ALL structured features and especially top ones
    feature_info: Dict[int, Dict] = {}
    for j, fi in enumerate(structured_idx):
        col = feats_csc.getcol(int(fi)).toarray().ravel()
        # Top-K activating residues
        topk = np.argsort(-col)[:50]
        topk = topk[col[topk] > 0]
        proteins_hit: Dict[int, List[int]] = {}
        for r_idx in topk:
            # Determine which protein
            # Use offsets dict — invert mapping
            pass
        feature_info[int(fi)] = {
            "max_f1": float(max_f1[fi]),
            "best_concept": str(concepts[int(np.argmax(F1[fi]))]),
            "n_active": int(nonzero_per_feat[fi]),
            "entropy": float(entropies[j]),
            "mean_activation_nonzero": float(means_structured[j]),
        }

    # Save
    np.savez(
        config.RESULTS / "dark_features.npz",
        all_max_f1=max_f1.astype(np.float32),
        nonzero_per_feat=nonzero_per_feat.astype(np.int64),
        dark_idx=dark_idx.astype(np.int32),
        dark_entropy=entropies,
        structured_dark_idx=structured_idx.astype(np.int32),
        top_dark_idx=top_idx.astype(np.int32),
        null_entropies=np.asarray(null_entropies, dtype=np.float32),
    )
    with open(config.RESULTS / "dark_features.json", "w") as f:
        json.dump(
            {
                "n_features": int(n_features),
                "n_dead": int((nonzero_per_feat == 0).sum()),
                "n_active": int((nonzero_per_feat > 0).sum()),
                "n_f1_ge_05": int((max_f1 >= 0.5).sum()),
                "n_f1_ge_03": int((max_f1 >= 0.3).sum()),
                "n_f1_lt_02": int((max_f1 < 0.2).sum()),
                "n_dark_candidate": int(len(dark_idx)),
                "n_structured_dark": int(len(structured_idx)),
                "null_entropy_p5": null_thresh,
                "null_entropy_median": float(np.median(null_entropies)),
                "dark_entropy_median": float(np.median(entropies)),
                "top_dark_indices": top_idx.tolist(),
                "top_dark_info": {int(i): feature_info[int(i)] for i in top_idx.tolist()},
            },
            f,
            indent=2,
        )
    print(f"[{time.strftime('%H:%M:%S')}] Saved dark features metadata.")
    print(f"  Top 10 dark feature indices: {top_idx[:10].tolist()}")


if __name__ == "__main__":
    main()
