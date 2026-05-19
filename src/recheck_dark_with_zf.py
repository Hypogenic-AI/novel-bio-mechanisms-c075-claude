"""Recheck whether the top dark features overlap with Zinc finger / Topological
domain / Repeat annotations, which were missing from the original F1 evaluation.

Output: results/zf_recheck.json with per-feature F1 against these extra concepts.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src import config
from src.compute_f1 import best_f1_for_column, SAE_PERCENTILES


EXTRA_CONCEPTS = ["Zinc finger", "Topological domain", "Repeat", "Compositional bias",
                  "Cross-link", "Coiled coil", "Nucleotide binding"]


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")
    feats_csc = feats.tocsc()
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    proteins = index["proteins"]
    offsets = {k: tuple(v) for k, v in index["offsets"].items()}
    total_residues = index["total_residues"]

    # Reload raw UniProt annotations to build labels for extra concepts
    with open(config.RESULTS / "uniprot_raw.json") as f:
        annotations = json.load(f)

    # Build per-residue labels for each extra concept
    n_extra = len(EXTRA_CONCEPTS)
    labels = np.zeros((total_residues, n_extra), dtype=bool)
    per_protein_domains: Dict[int, List[Tuple[str, List[Tuple[int, int]]]]] = defaultdict(list)
    domain_spans = {c: [] for c in EXTRA_CONCEPTS}

    concept_index = {c: i for i, c in enumerate(EXTRA_CONCEPTS)}
    for acc, _ in [(p["accession"], p["sequence"]) for p in proteins]:
        if acc not in annotations or acc not in offsets:
            continue
        start, length = offsets[acc]
        rec = annotations[acc]
        for ft in rec.get("features", []):
            ftype = ft.get("type")
            if ftype not in concept_index:
                continue
            ci = concept_index[ftype]
            loc = ft.get("location", {})
            try:
                s = int(loc["start"]["value"])
                e = int(loc["end"]["value"])
            except (KeyError, TypeError):
                continue
            s0, e0 = s - 1, min(e, length)
            labels[start + s0:start + e0, ci] = True
            domain_spans[ftype].append((start + s0, start + e0))

    print(f"  Extra concepts and their counts:")
    for i, c in enumerate(EXTRA_CONCEPTS):
        print(f"    {c:25s} positives={labels[:, i].sum():>10d}  "
              f"domains={len(domain_spans[c]):>5d}")

    # Load top dark features
    with open(config.RESULTS / "dark_features.json") as f:
        dark_meta = json.load(f)
    top_dark_idx = dark_meta["top_dark_indices"]
    print(f"\n[{time.strftime('%H:%M:%S')}] Checking F1 of top {len(top_dark_idx)} dark features "
          f"against {len(EXTRA_CONCEPTS)} extra concepts...")

    results = {}
    for fi in top_dark_idx:
        col = feats_csc.getcol(int(fi)).toarray().ravel()
        per_concept = {}
        for ci, c in enumerate(EXTRA_CONCEPTS):
            p, r, f1 = best_f1_for_column(
                col, labels[:, ci], domain_spans[c], SAE_PERCENTILES
            )
            per_concept[c] = {"precision": p, "recall": r, "f1": f1}
        best = max(per_concept.items(), key=lambda kv: kv[1]["f1"])
        results[int(fi)] = {
            "per_concept": per_concept,
            "best_concept": best[0],
            "best_f1": best[1]["f1"],
            "best_precision": best[1]["precision"],
            "best_recall": best[1]["recall"],
        }

    # Also run on a BROADER set of dark features (1207 structured ones)
    print(f"[{time.strftime('%H:%M:%S')}] Running on broader structured-dark population...")
    structured_idx = np.load(config.RESULTS / "dark_features.npz")["structured_dark_idx"]
    broader_best_f1 = np.zeros(len(structured_idx), dtype=np.float32)
    broader_best_concept = []
    for j, fi in enumerate(tqdm(structured_idx, desc="broader")):
        col = feats_csc.getcol(int(fi)).toarray().ravel()
        best_f1 = 0.0
        best_c = ""
        for ci, c in enumerate(EXTRA_CONCEPTS):
            _, _, f1 = best_f1_for_column(
                col, labels[:, ci], domain_spans[c], SAE_PERCENTILES
            )
            if f1 > best_f1:
                best_f1 = f1
                best_c = c
        broader_best_f1[j] = best_f1
        broader_best_concept.append(best_c)

    # How many of the structured dark features are recovered at various F1 thresholds?
    recovery = {}
    for thr in [0.2, 0.3, 0.4, 0.5]:
        recovery[f"recovered_at_f1_{thr}"] = int((broader_best_f1 >= thr).sum())
        recovery[f"frac_at_f1_{thr}"] = float((broader_best_f1 >= thr).mean())
    print(f"  Structured dark features 'rescued' by extra concepts (out of {len(structured_idx)}):")
    for k, v in recovery.items():
        print(f"    {k:30s} {v}")

    # Save
    with open(config.RESULTS / "zf_recheck.json", "w") as f:
        json.dump({
            "extra_concepts": EXTRA_CONCEPTS,
            "extra_concept_positives": {c: int(labels[:, i].sum())
                                         for i, c in enumerate(EXTRA_CONCEPTS)},
            "extra_concept_domains": {c: len(domain_spans[c]) for c in EXTRA_CONCEPTS},
            "top_dark_recheck": results,
            "broader_recovery": recovery,
        }, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Saved zf_recheck.json")


if __name__ == "__main__":
    main()
