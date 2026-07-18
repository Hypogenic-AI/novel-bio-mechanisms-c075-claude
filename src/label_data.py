"""Helpers to load expanded UniProt + motif reference labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src import config


def load_expanded_labels(
    results_dir: Path | None = None,
) -> Tuple[np.ndarray, List[str], List[str], Dict, Dict, Dict]:
    """Load UniProt reference + motif labels and concatenate columns.

    Returns
    -------
    labels
        (n_residues, n_uniprot + n_motif) bool array
    concepts
        concept id strings in column order
    grains
        grain per concept: ``any`` | ``subtype`` | ``motif``
    uniprot_meta, motif_meta
        raw metadata dicts
    per_protein_domains
        merged local span dicts keyed by accession then concept
    """
    results_dir = results_dir or config.RESULTS
    uniprot = np.load(results_dir / "reference_features.npz")
    with open(results_dir / "reference_features_meta.json") as f:
        uniprot_meta = json.load(f)
    motif = np.load(results_dir / "motif_annotations.npz")
    with open(results_dir / "motif_annotations_meta.json") as f:
        motif_meta = json.load(f)

    u_labels = uniprot["labels"]
    m_labels = motif["labels"]
    if u_labels.shape[0] != m_labels.shape[0]:
        raise ValueError(
            f"residue count mismatch: uniprot {u_labels.shape[0]} vs motif {m_labels.shape[0]}"
        )

    labels = np.concatenate([u_labels, m_labels], axis=1)
    concepts = list(uniprot_meta["concepts"]) + list(motif_meta["concepts"])
    grains = [c["grain"] for c in uniprot_meta["concept_metadata"]] + [
        c["grain"] for c in motif_meta["concept_metadata"]
    ]
    # Merge per-protein domain span dicts
    per_protein_domains: Dict = {}
    for source in (uniprot_meta["per_protein_domains"], motif_meta["per_protein_domains"]):
        for acc, by_concept in source.items():
            per_protein_domains.setdefault(acc, {}).update(by_concept)

    return labels, concepts, grains, uniprot_meta, motif_meta, per_protein_domains
