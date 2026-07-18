"""Compute SAE feature ↔ concept alignment using InterPLM-style domain-adjusted F1.

Uses the expanded multi-grain label ladder:
  - UniProt ``type::ANY`` / ``type::description`` from reference_features
  - sequence motifs (DRY / NPxxY / TGEKP) from motif_annotations
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src import config
from src.label_data import load_expanded_labels


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


def best_f1_for_feature_all_concepts(
    activations: np.ndarray,
    labels_matrix: np.ndarray,
    domain_spans_per_concept: List[List[Tuple[int, int]]],
    percentiles: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized F1 of one activation column against all concept label columns."""
    n_concepts = labels_matrix.shape[1]
    best_p = np.zeros(n_concepts, dtype=np.float32)
    best_r = np.zeros(n_concepts, dtype=np.float32)
    best_f1 = np.zeros(n_concepts, dtype=np.float32)

    nonzero = activations[activations > 0]
    if len(nonzero) < 5:
        return best_p, best_r, best_f1

    for pct in percentiles:
        thr = np.percentile(nonzero, pct)
        pred = activations >= thr
        n_pred = int(pred.sum())
        if n_pred < 3:
            continue
        tp = labels_matrix[pred].sum(axis=0).astype(np.float64)
        precision = tp / n_pred
        recall = np.zeros(n_concepts, dtype=np.float64)
        for ci, spans in enumerate(domain_spans_per_concept):
            if not spans:
                continue
            n_hit = sum(1 for s, e in spans if pred[s:e].any())
            recall[ci] = n_hit / len(spans)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-9)
        better = f1 > best_f1
        best_f1 = np.where(better, f1, best_f1).astype(np.float32)
        best_p = np.where(better, precision, best_p).astype(np.float32)
        best_r = np.where(better, recall, best_r).astype(np.float32)
    return best_p, best_r, best_f1


def evaluate_features_with_sweep(
    feats_csc: sparse.csc_matrix,
    labels_matrix: np.ndarray,
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
        p, r, f1 = best_f1_for_feature_all_concepts(
            col, labels_matrix, domain_spans_per_concept, percentiles
        )
        F1[fi] = f1
        P[fi] = p
        R[fi] = r
    return F1, P, R


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")
    hidden = np.load(config.RESULTS / "hidden_states.npz")["hidden"]
    labels, concepts, grains, _u_meta, _m_meta, per_protein_domains = load_expanded_labels()
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    offsets = {k: tuple(v) for k, v in index["offsets"].items()}

    if feats.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Activation residue count {feats.shape[0]} != label count {labels.shape[0]}. "
            "Re-run extract_activations using the same index.json as the labels."
        )

    print(f"  feats: {feats.shape} nnz={feats.nnz:,}")
    print(f"  hidden: {hidden.shape}")
    print(f"  labels: {labels.shape}, concepts: {len(concepts)} "
          f"(any={grains.count('any')}, subtype={grains.count('subtype')}, "
          f"motif={grains.count('motif')})")

    domain_spans_per_concept = [
        _flatten_concept_index(per_protein_domains, offsets, c) for c in concepts
    ]
    print("  concept positives / domain counts (top 15 by positives):")
    ranked = sorted(
        range(len(concepts)),
        key=lambda i: int(labels[:, i].sum()),
        reverse=True,
    )[:15]
    for i in ranked:
        print(f"    {concepts[i][:45]:45s} positives={int(labels[:, i].sum()):>10d}  "
              f"domains={len(domain_spans_per_concept[i]):>6d}  [{grains[i]}]")

    # ---- SAE features ----
    print(f"[{time.strftime('%H:%M:%S')}] Computing SAE feature F1 (threshold sweep)...")
    feats_csc = feats.tocsc()
    F1_sae, P_sae, R_sae = evaluate_features_with_sweep(
        feats_csc, labels, domain_spans_per_concept,
        n_features=feats.shape[1], n_concepts=len(concepts),
        percentiles=SAE_PERCENTILES, desc="SAE feats",
    )
    np.savez_compressed(
        config.RESULTS / "f1_sae.npz",
        F1=F1_sae,
        precision=P_sae,
        recall=R_sae,
        concepts=np.array(concepts),
        grains=np.array(grains),
    )
    max_per_feat = F1_sae.max(axis=1)
    print(f"  SAE max-F1 quantiles (over {feats.shape[1]} features): "
          f"50%={np.percentile(max_per_feat, 50):.3f}  "
          f"90%={np.percentile(max_per_feat, 90):.3f}  "
          f"99%={np.percentile(max_per_feat, 99):.3f}  "
          f"max={max_per_feat.max():.3f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  N features F1>={thr}: {(max_per_feat >= thr).sum()}")

    # Grain-split max F1
    grain_arr = np.array(grains)
    is_uniprot = grain_arr != "motif"
    is_motif = grain_arr == "motif"
    max_uniprot = F1_sae[:, is_uniprot].max(axis=1) if is_uniprot.any() else np.zeros(F1_sae.shape[0])
    max_motif = F1_sae[:, is_motif].max(axis=1) if is_motif.any() else np.zeros(F1_sae.shape[0])
    print(f"  SAE max UniProt-F1>=0.5: {(max_uniprot >= 0.5).sum()}")
    print(f"  SAE max motif-F1>=0.5:   {(max_motif >= 0.5).sum()}")

    # ---- Raw ESM-2 neurons baseline ----
    print(f"[{time.strftime('%H:%M:%S')}] Computing neuron F1...")
    hidden_pos = np.clip(hidden.astype(np.float32), 0, None)
    n_neurons = hidden_pos.shape[1]
    F1_neu = np.zeros((n_neurons, len(concepts)), dtype=np.float32)
    P_neu = np.zeros_like(F1_neu)
    R_neu = np.zeros_like(F1_neu)
    for ni in tqdm(range(n_neurons), desc="neurons"):
        col = hidden_pos[:, ni]
        if col.sum() == 0:
            continue
        p, r, f1 = best_f1_for_feature_all_concepts(
            col, labels, domain_spans_per_concept, NEU_PERCENTILES
        )
        F1_neu[ni] = f1
        P_neu[ni] = p
        R_neu[ni] = r
    np.savez_compressed(
        config.RESULTS / "f1_neurons.npz",
        F1=F1_neu,
        precision=P_neu,
        recall=R_neu,
        concepts=np.array(concepts),
        grains=np.array(grains),
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
        "grains": grains,
        "n_concepts_by_grain": {
            "any": grains.count("any"),
            "subtype": grains.count("subtype"),
            "motif": grains.count("motif"),
        },
        "sae_percentile_sweep": SAE_PERCENTILES,
        "neu_percentile_sweep": NEU_PERCENTILES,
        "sae": {
            "n_F1_ge_0_5": int((max_per_feat >= 0.5).sum()),
            "n_F1_ge_0_4": int((max_per_feat >= 0.4).sum()),
            "n_F1_ge_0_3": int((max_per_feat >= 0.3).sum()),
            "n_F1_lt_0_2": int((max_per_feat < 0.2).sum()),
            "n_uniprot_F1_ge_0_5": int((max_uniprot >= 0.5).sum()),
            "n_motif_F1_ge_0_5": int((max_motif >= 0.5).sum()),
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
