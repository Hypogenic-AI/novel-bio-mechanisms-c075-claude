"""Identify and characterize 'dark' SAE features under the expanded label ladder.

A residual dark feature has max F1 below DARK_F1_THRESHOLD against *both*
UniProt concepts (any/subtype) and motif concepts. We further filter for
structured dark features via protein-activation entropy.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
from scipy import sparse

from src import config


def feature_protein_entropy(
    feats: sparse.csr_matrix,
    protein_lengths: List[int],
    feature_idx: np.ndarray,
) -> np.ndarray:
    """For each feature, compute the entropy of its activation distribution across proteins."""
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
            entropies[j] = np.log(n_prot)
            continue
        acts_by_prot = np.bincount(prot_ids[mask], weights=col[mask], minlength=n_prot)
        p = acts_by_prot / acts_by_prot.sum()
        p_pos = p[p > 0]
        entropies[j] = float(-(p_pos * np.log(p_pos)).sum())
    return entropies


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading F1 and activations...")
    f1_data = np.load(config.RESULTS / "f1_sae.npz", allow_pickle=True)
    F1 = f1_data["F1"]  # (n_features, n_concepts)
    concepts = f1_data["concepts"].tolist()
    if "grains" in f1_data.files:
        grains = f1_data["grains"].tolist()
    else:
        grains = ["subtype"] * len(concepts)
    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")

    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    proteins = index["proteins"]
    protein_lengths = [len(p["sequence"]) for p in proteins]

    n_features = F1.shape[0]
    grain_arr = np.asarray(grains)
    is_uniprot = grain_arr != "motif"
    is_motif = grain_arr == "motif"

    max_f1 = F1.max(axis=1)
    max_uniprot = F1[:, is_uniprot].max(axis=1) if is_uniprot.any() else np.zeros(n_features)
    max_motif = F1[:, is_motif].max(axis=1) if is_motif.any() else np.zeros(n_features)
    best_concept_idx = np.argmax(F1, axis=1)
    best_uniprot_idx = np.argmax(F1[:, is_uniprot], axis=1) if is_uniprot.any() else np.zeros(n_features, dtype=int)
    best_motif_idx = np.argmax(F1[:, is_motif], axis=1) if is_motif.any() else np.zeros(n_features, dtype=int)
    uniprot_concepts = [c for c, g in zip(concepts, grains) if g != "motif"]
    motif_concepts = [c for c, g in zip(concepts, grains) if g == "motif"]

    nonzero_per_feat = np.asarray((feats != 0).sum(axis=0)).ravel()
    print(f"  n_features={n_features}")
    print(f"  dead (no activations): {(nonzero_per_feat == 0).sum()}")
    print(f"  active features:       {(nonzero_per_feat > 0).sum()}")

    thr = config.DARK_F1_THRESHOLD
    active_enough = nonzero_per_feat >= config.DARK_MIN_ACTIVE_RESIDUES
    # Residual dark: low on UniProt ladder AND low on motifs
    is_dark = (max_uniprot < thr) & (max_motif < thr) & active_enough
    # Granularity gap: low UniProt, high motif
    is_gap = (max_uniprot < thr) & (max_motif >= thr) & active_enough
    # Bright on UniProt ladder
    is_bright = (max_uniprot >= 0.5) & active_enough

    dark_idx = np.where(is_dark)[0]
    gap_idx = np.where(is_gap)[0]
    print(f"  residual dark (uni< {thr} & motif<{thr}, >=100 acts): {len(dark_idx)}")
    print(f"  granularity gap (uni<{thr} & motif>={thr}): {len(gap_idx)}")
    print(f"  bright UniProt (max uni F1>=0.5, >=100 acts): {is_bright.sum()}")

    print(f"[{time.strftime('%H:%M:%S')}] Computing protein-entropy for {len(dark_idx)} dark features...")
    entropies = feature_protein_entropy(feats, protein_lengths, dark_idx) if len(dark_idx) else np.array([])
    n_prot = len(proteins)
    if len(dark_idx) == 0:
        print("  No residual dark features; skipping entropy / top ranking.")
        structured_idx = np.array([], dtype=np.int32)
        top_idx = np.array([], dtype=np.int32)
        null_thresh = float("nan")
        null_entropies: List[float] = []
        feature_info: Dict[int, Dict] = {}
    else:
        print(f"  Dark feature entropy quantiles: "
              f"10%={np.percentile(entropies, 10):.2f}  "
              f"50%={np.percentile(entropies, 50):.2f}  "
              f"90%={np.percentile(entropies, 90):.2f}")

        rng = np.random.default_rng(config.SEED)
        n_null = 500
        null_entropies = []
        sizes = nonzero_per_feat[dark_idx]
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
        null_thresh = float(np.percentile(null_entropies, 5))
        print(f"  Null entropy median={np.median(null_entropies):.2f}, p5={null_thresh:.2f}")

        structured = entropies < null_thresh
        structured_idx = dark_idx[structured]
        print(f"  Structured residual dark (entropy < null p5): {len(structured_idx)}")

        feats_csc = feats.tocsc()
        means_structured = np.zeros(len(structured_idx))
        nz_structured = np.zeros(len(structured_idx), dtype=np.int64)
        for j, fi in enumerate(structured_idx):
            col = feats_csc.getcol(fi).toarray().ravel()
            nz = col > 0
            if nz.sum() > 0:
                means_structured[j] = col[nz].mean()
                nz_structured[j] = int(nz.sum())

        rank_score = means_structured * np.log1p(nz_structured)
        top_order = np.argsort(-rank_score)
        n_top = min(30, len(structured_idx))
        top_idx = structured_idx[top_order[:n_top]]

        feature_info = {}
        for j, fi in enumerate(structured_idx):
            feature_info[int(fi)] = {
                "max_f1": float(max_f1[fi]),
                "max_uniprot_f1": float(max_uniprot[fi]),
                "max_motif_f1": float(max_motif[fi]),
                "best_concept": str(concepts[int(best_concept_idx[fi])]),
                "best_grain": str(grains[int(best_concept_idx[fi])]),
                "best_uniprot_concept": (
                    str(uniprot_concepts[int(best_uniprot_idx[fi])]) if uniprot_concepts else None
                ),
                "best_motif_concept": (
                    str(motif_concepts[int(best_motif_idx[fi])]) if motif_concepts else None
                ),
                "n_active": int(nonzero_per_feat[fi]),
                "entropy": float(entropies[j]),
                "mean_activation_nonzero": float(means_structured[j]),
            }

    # Gap feature examples for reporting
    gap_info = {}
    for fi in gap_idx[:50]:
        gap_info[int(fi)] = {
            "max_uniprot_f1": float(max_uniprot[fi]),
            "max_motif_f1": float(max_motif[fi]),
            "best_motif_concept": (
                str(motif_concepts[int(best_motif_idx[fi])]) if motif_concepts else None
            ),
            "n_active": int(nonzero_per_feat[fi]),
        }

    np.savez(
        config.RESULTS / "dark_features.npz",
        all_max_f1=max_f1.astype(np.float32),
        max_uniprot_f1=max_uniprot.astype(np.float32),
        max_motif_f1=max_motif.astype(np.float32),
        nonzero_per_feat=nonzero_per_feat.astype(np.int64),
        dark_idx=dark_idx.astype(np.int32),
        gap_idx=gap_idx.astype(np.int32),
        dark_entropy=entropies.astype(np.float32) if len(entropies) else np.array([], dtype=np.float32),
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
                "n_uniprot_f1_ge_05": int((max_uniprot >= 0.5).sum()),
                "n_motif_f1_ge_05": int((max_motif >= 0.5).sum()),
                "n_bright_uniprot": int(is_bright.sum()),
                "n_granularity_gap": int(len(gap_idx)),
                "n_dark_candidate": int(len(dark_idx)),
                "n_structured_dark": int(len(structured_idx)),
                "null_entropy_p5": null_thresh if null_entropies else None,
                "null_entropy_median": float(np.median(null_entropies)) if null_entropies else None,
                "dark_entropy_median": float(np.median(entropies)) if len(entropies) else None,
                "top_dark_indices": top_idx.tolist(),
                "top_dark_info": {int(i): feature_info[int(i)] for i in top_idx.tolist()},
                "gap_examples": gap_info,
                "label_ladder": {
                    "n_any": int((grain_arr == "any").sum()),
                    "n_subtype": int((grain_arr == "subtype").sum()),
                    "n_motif": int((grain_arr == "motif").sum()),
                },
            },
            f,
            indent=2,
        )
    print(f"[{time.strftime('%H:%M:%S')}] Saved dark features metadata.")
    if len(top_idx):
        print(f"  Top 10 residual dark indices: {top_idx[:10].tolist()}")


if __name__ == "__main__":
    main()
