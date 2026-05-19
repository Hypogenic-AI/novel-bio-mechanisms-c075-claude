"""Compute SAE feature ↔ Swiss-Prot concept alignment using InterPLM-style
domain-adjusted F1, with a threshold sweep for fair comparison between sparse
SAE features and dense ESM-2 neurons.

For each feature × concept pair, we sweep activation thresholds and report the
*best* F1 across thresholds. Following InterPLM:
  - Precision: fraction of residues with `activation >= threshold` whose label is True.
  - Recall: fraction of domain instances (e.g., a specific binding site in one
    protein) that have at least one residue above threshold.

We exclude the "Chain" concept because it covers ~the entire protein (1486
chain-spans, 410K positive residues out of 422K) and is trivially matched by any
broadly-firing feature.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src import config


# Percentile sweep used for SAE features (over nonzero activations)
SAE_PERCENTILES = [50, 75, 90, 95, 99]
# Absolute percentile sweep for raw neurons (over positive part of activations)
NEU_PERCENTILES = [80, 90, 95, 99]


def _flatten_concept_index(
    per_protein_domains: Dict[str, Dict[str, List[Tuple[int, int]]]],
    offsets: Dict[str, Tuple[int, int]],
    concept: str,
) -> List[Tuple[int, int]]:
    spans = []
    for acc, by_type in per_protein_domains.items():
        if concept not in by_type or acc not in offsets:
            continue
        start, _ = offsets[acc]
        for s, e in by_type[concept]:
            spans.append((start + s, start + e))
    return spans


def best_f1_for_column(
    activations: np.ndarray,
    labels: np.ndarray,
    domain_spans: List[Tuple[int, int]],
    percentiles: List[float],
) -> Tuple[float, float, float]:
    """Return (best_precision, best_recall, best_f1) by sweeping thresholds at
    `percentiles` of nonzero activations.
    """
    if len(domain_spans) == 0:
        return 0.0, 0.0, 0.0
    nonzero = activations[activations > 0]
    if len(nonzero) < 5:
        return 0.0, 0.0, 0.0
    best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    for pct in percentiles:
        thr = np.percentile(nonzero, pct)
        pred = activations >= thr
        if pred.sum() < 3:
            continue
        tp = (pred & labels).sum()
        n_pred = pred.sum()
        precision = tp / n_pred
        # Domain-adjusted recall: at least one positive prediction in the span
        n_hit = sum(1 for s, e in domain_spans if pred[s:e].any())
        recall = n_hit / len(domain_spans)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best_f1:
            best_p, best_r, best_f1 = precision, recall, f1
    return float(best_p), float(best_r), float(best_f1)


def evaluate_features_with_sweep(
    feats_csc: sparse.csc_matrix,
    labels_per_concept: List[np.ndarray],
    domain_spans_per_concept: List[List[Tuple[int, int]]],
    n_features: int,
    n_concepts: int,
    percentiles: List[float],
    desc: str = "features",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    F1 = np.zeros((n_features, n_concepts), dtype=np.float32)
    P = np.zeros_like(F1)
    R = np.zeros_like(F1)
    for fi in tqdm(range(n_features), desc=desc):
        col = feats_csc.getcol(fi).toarray().ravel()
        if col.sum() == 0:
            continue
        for ci in range(n_concepts):
            p, r, f1 = best_f1_for_column(
                col, labels_per_concept[ci], domain_spans_per_concept[ci], percentiles
            )
            F1[fi, ci] = f1
            P[fi, ci] = p
            R[fi, ci] = r
    return F1, P, R


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")
    hidden = np.load(config.RESULTS / "hidden_states.npz")["hidden"]
    ann_data = np.load(config.RESULTS / "annotations.npz")
    labels = ann_data["labels"]
    with open(config.RESULTS / "annotations_meta.json") as f:
        meta = json.load(f)
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    all_concepts = meta["concepts"]
    per_protein_domains = meta["per_protein_domains"]
    offsets = {k: tuple(v) for k, v in index["offsets"].items()}

    # Filter out the universal Chain concept
    keep = [c != "Chain" for c in all_concepts]
    concepts = [c for c, k in zip(all_concepts, keep) if k]
    labels = labels[:, np.array(keep)]
    print(f"  feats: {feats.shape} nnz={feats.nnz:,}")
    print(f"  hidden: {hidden.shape}")
    print(f"  labels: {labels.shape}, concepts: {concepts}")

    domain_spans_per_concept = [
        _flatten_concept_index(per_protein_domains, offsets, c) for c in concepts
    ]
    labels_per_concept = [labels[:, i] for i in range(len(concepts))]
    print("  concept positives / domain counts:")
    for i, c in enumerate(concepts):
        print(f"    {c:30s} positives={int(labels[:, i].sum()):>10d}  "
              f"domains={len(domain_spans_per_concept[i]):>6d}")

    # ---- SAE features ----
    print(f"[{time.strftime('%H:%M:%S')}] Computing SAE feature F1 (threshold sweep)...")
    feats_csc = feats.tocsc()
    F1_sae, P_sae, R_sae = evaluate_features_with_sweep(
        feats_csc, labels_per_concept, domain_spans_per_concept,
        n_features=feats.shape[1], n_concepts=len(concepts),
        percentiles=SAE_PERCENTILES, desc="SAE feats",
    )
    np.savez_compressed(
        config.RESULTS / "f1_sae.npz",
        F1=F1_sae, precision=P_sae, recall=R_sae, concepts=np.array(concepts),
    )
    max_per_feat = F1_sae.max(axis=1)
    print(f"  SAE max-F1 quantiles (over 10240 features): "
          f"50%={np.percentile(max_per_feat, 50):.3f}  "
          f"90%={np.percentile(max_per_feat, 90):.3f}  "
          f"99%={np.percentile(max_per_feat, 99):.3f}  "
          f"max={max_per_feat.max():.3f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  N features F1>={thr}: {(max_per_feat >= thr).sum()}")

    # ---- Raw ESM-2 neurons baseline ----
    print(f"[{time.strftime('%H:%M:%S')}] Computing neuron F1...")
    # Make hidden into a sparse-equivalent: clip negatives to 0, then evaluate
    hidden_pos = np.clip(hidden.astype(np.float32), 0, None)
    n_neurons = hidden_pos.shape[1]
    F1_neu = np.zeros((n_neurons, len(concepts)), dtype=np.float32)
    P_neu = np.zeros_like(F1_neu)
    R_neu = np.zeros_like(F1_neu)
    for ni in tqdm(range(n_neurons), desc="neurons"):
        col = hidden_pos[:, ni]
        if col.sum() == 0:
            continue
        for ci in range(len(concepts)):
            p, r, f1 = best_f1_for_column(
                col, labels_per_concept[ci], domain_spans_per_concept[ci], NEU_PERCENTILES
            )
            F1_neu[ni, ci] = f1
            P_neu[ni, ci] = p
            R_neu[ni, ci] = r
    np.savez_compressed(
        config.RESULTS / "f1_neurons.npz",
        F1=F1_neu, precision=P_neu, recall=R_neu, concepts=np.array(concepts),
    )
    max_per_neu = F1_neu.max(axis=1)
    print(f"  Neuron max-F1 quantiles: "
          f"50%={np.percentile(max_per_neu, 50):.3f}  "
          f"90%={np.percentile(max_per_neu, 90):.3f}  "
          f"99%={np.percentile(max_per_neu, 99):.3f}  "
          f"max={max_per_neu.max():.3f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  N neurons F1>={thr}: {(max_per_neu >= thr).sum()}")

    summary = {
        "n_residues": int(feats.shape[0]),
        "n_features": int(feats.shape[1]),
        "n_neurons": int(n_neurons),
        "concepts": concepts,
        "sae_percentile_sweep": SAE_PERCENTILES,
        "neu_percentile_sweep": NEU_PERCENTILES,
        "sae": {
            "n_F1_ge_0_5": int((max_per_feat >= 0.5).sum()),
            "n_F1_ge_0_4": int((max_per_feat >= 0.4).sum()),
            "n_F1_ge_0_3": int((max_per_feat >= 0.3).sum()),
            "n_F1_lt_0_2": int((max_per_feat < 0.2).sum()),
            "max_F1_quantiles": {
                "p50": float(np.percentile(max_per_feat, 50)),
                "p90": float(np.percentile(max_per_feat, 90)),
                "p99": float(np.percentile(max_per_feat, 99)),
                "max": float(max_per_feat.max()),
            },
        },
        "neurons": {
            "n_F1_ge_0_5": int((max_per_neu >= 0.5).sum()),
            "n_F1_ge_0_4": int((max_per_neu >= 0.4).sum()),
            "n_F1_ge_0_3": int((max_per_neu >= 0.3).sum()),
            "max_F1_quantiles": {
                "p50": float(np.percentile(max_per_neu, 50)),
                "p90": float(np.percentile(max_per_neu, 90)),
                "p99": float(np.percentile(max_per_neu, 99)),
                "max": float(max_per_neu.max()),
            },
        },
        "n_concepts_covered_at_05": {
            "sae": int(((F1_sae >= 0.5).any(axis=0)).sum()),
            "neurons": int(((F1_neu >= 0.5).any(axis=0)).sum()),
        },
    }
    with open(config.RESULTS / "f1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Concepts covered (>=0.5 F1) by SAE: {summary['n_concepts_covered_at_05']['sae']}/{len(concepts)}")
    print(f"  Concepts covered (>=0.5 F1) by neurons: {summary['n_concepts_covered_at_05']['neurons']}/{len(concepts)}")
    print(f"[{time.strftime('%H:%M:%S')}] Saved summary to f1_summary.json")


if __name__ == "__main__":
    main()
