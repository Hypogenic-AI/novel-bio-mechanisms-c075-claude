"""Build expanded UniProt reference-feature labels.

This module turns cached UniProt feature records (from uniprot_raw.json) into a multi-resolution
concept dictionary:

This allows us to sidestep issues caused by the inclusion of more granular features by
placing both granular and coarse feature-type labels into the same concept dictionary.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src import config


DEFAULT_EXCLUDED_FEATURE_TYPES = frozenset(
    {
        "Alternative sequence",
        "Chain",
        "Mutagenesis",
        "Natural variant",
        "Sequence conflict",
    }
)

DEFAULT_MIN_DOMAIN_INSTANCES = 10
DEFAULT_MIN_POSITIVE_RESIDUES = 1000
ANY_DESCRIPTION = "ANY"
MISSING_DESCRIPTION = "(none)"


@dataclass(frozen=True)
class SpanRecord:
    """A local and global span for one concept instance."""

    accession: str
    local_start: int
    local_end: int
    global_start: int
    global_end: int


def normalize_description(description: object) -> str:
    """Return a stable subtype label from a UniProt feature description."""

    if description is None:
        return MISSING_DESCRIPTION
    text = str(description).strip()
    if not text:
        return MISSING_DESCRIPTION
    return re.sub(r"\s+", " ", text)


def location_span(location: Mapping[str, object]) -> Tuple[int, int] | None:
    """Return UniProt's 1-indexed inclusive start/end span."""

    try:
        start = location["start"]  # type: ignore[index]
        end = location["end"]  # type: ignore[index]
        start_value = start["value"]  # type: ignore[index]
        end_value = end["value"]  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    if start_value is None or end_value is None:
        return None
    try:
        return int(start_value), int(end_value)
    except (TypeError, ValueError):
        return None


def concept_id(feature_type: str, description: str) -> str:
    return f"{feature_type}::{description}"


def _concept_metadata(concept: str, n_domains: int, n_positive_residues: int) -> dict:
    feature_type, description = concept.split("::", 1)
    grain = "any" if description == ANY_DESCRIPTION else "subtype"
    return {
        "concept": concept,
        "feature_type": feature_type,
        "description": description,
        "grain": grain,
        "n_domains": int(n_domains),
        "n_positive_residues": int(n_positive_residues),
    }


def _iter_feature_concepts(feature: Mapping[str, object]) -> Iterable[str]:
    feature_type = feature.get("type")
    if not feature_type:
        return
    feature_type_str = str(feature_type)
    yield concept_id(feature_type_str, ANY_DESCRIPTION)
    yield concept_id(feature_type_str, normalize_description(feature.get("description")))


def _passes_frequency_filter(
    spans: Sequence[SpanRecord],
    min_domain_instances: int,
    min_positive_residues: int,
) -> bool:
    n_positive_residues = sum(span.global_end - span.global_start for span in spans)
    return len(spans) >= min_domain_instances or n_positive_residues >= min_positive_residues


def build_reference_feature_labels(
    proteins: Sequence[Tuple[str, str]],
    annotation_records: Mapping[str, Mapping[str, object]],
    offsets: Mapping[str, Tuple[int, int]],
    total_residues: int,
    *,
    excluded_feature_types: Iterable[str] = DEFAULT_EXCLUDED_FEATURE_TYPES,
    min_domain_instances: int = DEFAULT_MIN_DOMAIN_INSTANCES,
    min_positive_residues: int = DEFAULT_MIN_POSITIVE_RESIDUES,
) -> Tuple[np.ndarray, dict]:
    """Build per-residue labels and metadata for expanded UniProt concepts.

    ``proteins`` should match the order in ``results/index.json``. Spans in
    ``per_protein_domains`` are local, 0-indexed, half-open intervals so they
    can be flattened by ``src.compute_f1._flatten_concept_index``.
    """

    excluded = set(excluded_feature_types)
    spans_by_concept: Dict[str, List[SpanRecord]] = defaultdict(list)

    for accession, sequence in proteins:
        if accession not in annotation_records or accession not in offsets:
            continue
        start_idx, length = offsets[accession]
        record = annotation_records[accession]
        for feature in record.get("features", []):  # type: ignore[union-attr]
            if not isinstance(feature, Mapping):
                continue
            feature_type = feature.get("type")
            if feature_type is None or str(feature_type) in excluded:
                continue
            span = location_span(feature.get("location", {}))  # type: ignore[arg-type]
            if span is None:
                continue
            start_1, end_1 = span
            local_start = max(start_1 - 1, 0)
            local_end = min(end_1, length, len(sequence))
            if local_end <= local_start:
                continue
            global_start = start_idx + local_start
            global_end = start_idx + local_end
            span_record = SpanRecord(
                accession=accession,
                local_start=local_start,
                local_end=local_end,
                global_start=global_start,
                global_end=global_end,
            )
            for cid in _iter_feature_concepts(feature):
                spans_by_concept[cid].append(span_record)

    kept_concepts = [
        cid
        for cid, spans in spans_by_concept.items()
        if _passes_frequency_filter(spans, min_domain_instances, min_positive_residues)
    ]
    kept_concepts.sort(key=lambda cid: (cid.split("::", 1)[0], cid.split("::", 1)[1] != ANY_DESCRIPTION, cid))

    labels = np.zeros((total_residues, len(kept_concepts)), dtype=bool)
    per_protein_domains: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    concept_metadata = []

    for concept_index, cid in enumerate(kept_concepts):
        spans = spans_by_concept[cid]
        for span in spans:
            labels[span.global_start:span.global_end, concept_index] = True
            per_protein_domains[span.accession][cid].append((span.local_start, span.local_end))
        concept_metadata.append(
            _concept_metadata(
                cid,
                n_domains=len(spans),
                n_positive_residues=int(labels[:, concept_index].sum()),
            )
        )

    metadata = {
        "concepts": kept_concepts,
        "concept_metadata": concept_metadata,
        "per_protein_domains": {
            acc: dict(by_concept) for acc, by_concept in per_protein_domains.items()
        },
        "excluded_feature_types": sorted(excluded),
        "min_domain_instances": int(min_domain_instances),
        "min_positive_residues": int(min_positive_residues),
        "n_proteins_annotated": sum(1 for accession, _ in proteins if accession in annotation_records),
    }
    return labels, metadata


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Loading index + cached UniProt records...")
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    with open(config.RESULTS / "uniprot_raw.json") as f:
        annotation_records = json.load(f)

    proteins = [(p["accession"], p["sequence"]) for p in index["proteins"]]
    offsets = {key: tuple(value) for key, value in index["offsets"].items()}
    labels, metadata = build_reference_feature_labels(
        proteins,
        annotation_records,
        offsets,
        int(index["total_residues"]),
    )

    np.savez_compressed(config.RESULTS / "reference_features.npz", labels=labels)
    with open(config.RESULTS / "reference_features_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(config.RESULTS / "concepts_uniprot.json", "w") as f:
        json.dump(metadata["concept_metadata"], f, indent=2)

    n_any = sum(1 for concept in metadata["concept_metadata"] if concept["grain"] == "any")
    n_subtype = sum(1 for concept in metadata["concept_metadata"] if concept["grain"] == "subtype")
    print(
        f"[{time.strftime('%H:%M:%S')}] Wrote {labels.shape[1]} reference concepts "
        f"({n_any} ANY, {n_subtype} subtype) for {labels.shape[0]:,} residues."
    )


if __name__ == "__main__":
    main()
