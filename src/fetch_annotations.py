"""Fetch Swiss-Prot per-residue annotations for our subset from the UniProt REST API.

Uses the per-accession endpoint `https://rest.uniprot.org/uniprotkb/{ACC}.json`
in parallel with a small thread pool (UniProt allows ~30 req/s).

Output:
  results/annotations.npz - bool array (N_residues, N_concepts)
  results/annotations_meta.json - {concepts, per-protein domain spans}
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from tqdm import tqdm

from src import config


TARGET_FEATURE_TYPES = [
    "Active site",
    "Binding site",
    "Disulfide bond",
    "Motif",
    "Domain",
    "Region",
    "Helix",
    "Beta strand",
    "Turn",
    "Transmembrane",
    "Signal",
    "Chain",
    "Modified residue",
    "Lipidation",
    "Glycosylation",
    "Site",
]


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "InterPLM-research/0.1"})


def fetch_one(acc: str, retries: int = 3) -> dict | None:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (429, 503):
                time.sleep(2 * (attempt + 1))
            else:
                return None
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def location_span(loc: dict) -> Tuple[int, int] | None:
    try:
        s = loc["start"]["value"]
        e = loc["end"]["value"]
        if s is None or e is None:
            return None
        return int(s), int(e)
    except (KeyError, TypeError):
        return None


def build_per_residue_labels(
    proteins: List[Tuple[str, str]],
    annotation_records: Dict[str, dict],
    concepts: List[str],
    offsets: Dict[str, Tuple[int, int]],
    total_residues: int,
) -> Tuple[np.ndarray, Dict]:
    labels = np.zeros((total_residues, len(concepts)), dtype=bool)
    concept_index = {c: i for i, c in enumerate(concepts)}
    per_protein_domains: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for acc, seq in proteins:
        if acc not in annotation_records or acc not in offsets:
            continue
        start_idx, length = offsets[acc]
        rec = annotation_records[acc]
        for ft in rec.get("features", []):
            ftype = ft.get("type")
            if ftype not in concept_index:
                continue
            ci = concept_index[ftype]
            span = location_span(ft.get("location", {}))
            if span is None:
                continue
            s1, e1 = span
            s0, e0 = s1 - 1, min(e1, length)
            global_s = start_idx + s0
            global_e = start_idx + e0
            labels[global_s:global_e, ci] = True
            per_protein_domains[acc][ftype].append((s0, e0))
    return labels, {acc: dict(d) for acc, d in per_protein_domains.items()}


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading index.json...")
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    proteins = [(p["accession"], p["sequence"]) for p in index["proteins"]]
    offsets = {k: tuple(v) for k, v in index["offsets"].items()}
    total_residues = index["total_residues"]
    print(f"  {len(proteins)} proteins, {total_residues:,} residues")

    cache_path = config.RESULTS / "uniprot_raw.json"
    annotations: Dict[str, dict] = {}
    if cache_path.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Loading cached annotations from {cache_path}...")
        with open(cache_path) as f:
            annotations = json.load(f)
        print(f"  Loaded {len(annotations)} cached records")

    accs_to_fetch = [acc for acc, _ in proteins if acc not in annotations]
    print(f"[{time.strftime('%H:%M:%S')}] Fetching {len(accs_to_fetch)} annotations in parallel...")

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_one, acc): acc for acc in accs_to_fetch}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="UniProt")):
            acc = futures[fut]
            try:
                rec = fut.result()
            except Exception:  # noqa: BLE001
                rec = None
            if rec is not None:
                annotations[acc] = rec
            # Save cache every 200 records
            if (i + 1) % 200 == 0:
                with open(cache_path, "w") as f:
                    json.dump(annotations, f)
    # Final cache save
    with open(cache_path, "w") as f:
        json.dump(annotations, f)
    print(f"  Retrieved annotations for {len(annotations)} / {len(proteins)} proteins")

    # Sanity: top feature types
    type_counts: Dict[str, int] = defaultdict(int)
    for rec in annotations.values():
        for ft in rec.get("features", []):
            type_counts[ft.get("type", "?")] += 1
    print("  Top feature types:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1])[:25]:
        print(f"    {k:30s} {v}")

    labels, per_protein_domains = build_per_residue_labels(
        proteins, annotations, TARGET_FEATURE_TYPES, offsets, total_residues
    )
    print(f"  labels shape={labels.shape}")
    for i, c in enumerate(TARGET_FEATURE_TYPES):
        print(f"    {c:30s} positives={labels[:, i].sum():>10d} "
              f"protein-domains={sum(1 for v in per_protein_domains.values() if c in v)}")

    np.savez_compressed(config.RESULTS / "annotations.npz", labels=labels)
    with open(config.RESULTS / "annotations_meta.json", "w") as f:
        json.dump(
            {
                "concepts": TARGET_FEATURE_TYPES,
                "per_protein_domains": per_protein_domains,
                "n_proteins_annotated": len(annotations),
            },
            f,
        )
    print(f"[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
