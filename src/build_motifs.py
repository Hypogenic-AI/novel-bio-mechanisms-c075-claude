"""Build sequence-motif reference labels (finest grain of the annotation ladder).

Scans protein sequences for known short motifs that UniProt often does not
annotate as their own feature types (DRY, NPxxY, TGEKP linkers). Output uses
the same label / span conventions as ``src.build_concepts`` so F1 scoring can
treat motif concepts as ``grain: motif``.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src import config
from src.build_concepts import SpanRecord, location_span


@dataclass(frozen=True)
class MotifSpec:
    """One motif concept and how to find it."""

    name: str
    pattern: str
    # If True and UniProt Zinc finger spans are available, keep only overlaps.
    require_zinc_finger_overlap: bool = False


DEFAULT_MOTIF_SPECS: Tuple[MotifSpec, ...] = (
    MotifSpec(name="DRY", pattern="DRY"),
    MotifSpec(name="NPxxY", pattern=r"NP..Y"),
    MotifSpec(name="TGEKP", pattern=r"H?TGEKP", require_zinc_finger_overlap=True),
)


def concept_id(name: str) -> str:
    return f"Motif::{name}"


def find_motif_spans(sequence: str, pattern: str) -> List[Tuple[int, int]]:
    """Return 0-indexed half-open spans for non-overlapping regex matches."""

    spans: List[Tuple[int, int]] = []
    for match in re.finditer(pattern, sequence, flags=re.IGNORECASE):
        spans.append((match.start(), match.end()))
    return spans


def _zinc_finger_local_spans(
    record: Mapping[str, object],
    sequence_length: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for feature in record.get("features", []):  # type: ignore[union-attr]
        if not isinstance(feature, Mapping):
            continue
        if feature.get("type") != "Zinc finger":
            continue
        loc = location_span(feature.get("location", {}))  # type: ignore[arg-type]
        if loc is None:
            continue
        start_1, end_1 = loc
        local_start = max(start_1 - 1, 0)
        local_end = min(end_1, sequence_length)
        if local_end > local_start:
            spans.append((local_start, local_end))
    return spans


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def build_motif_labels(
    proteins: Sequence[Tuple[str, str]],
    offsets: Mapping[str, Tuple[int, int]],
    total_residues: int,
    *,
    motif_specs: Sequence[MotifSpec] = DEFAULT_MOTIF_SPECS,
    annotation_records: Optional[Mapping[str, Mapping[str, object]]] = None,
    require_zinc_finger_overlap: Optional[bool] = None,
) -> Tuple[np.ndarray, dict]:
    """Build per-residue motif labels and metadata.

    ``require_zinc_finger_overlap`` overrides the per-spec default when not
    None. Spans in ``per_protein_domains`` are local, 0-indexed, half-open.
    """

    spans_by_concept: Dict[str, List[SpanRecord]] = defaultdict(list)

    for accession, sequence in proteins:
        if accession not in offsets:
            continue
        start_idx, length = offsets[accession]
        seq = sequence[:length]
        zf_spans: List[Tuple[int, int]] = []
        if annotation_records and accession in annotation_records:
            zf_spans = _zinc_finger_local_spans(annotation_records[accession], len(seq))

        for spec in motif_specs:
            cid = concept_id(spec.name)
            use_zf = (
                spec.require_zinc_finger_overlap
                if require_zinc_finger_overlap is None
                else require_zinc_finger_overlap and spec.require_zinc_finger_overlap
            )
            # When caller forces require_zinc_finger_overlap=False, disable for all.
            if require_zinc_finger_overlap is False:
                use_zf = False
            elif require_zinc_finger_overlap is True and spec.name == "TGEKP":
                use_zf = True

            for local_start, local_end in find_motif_spans(seq, spec.pattern):
                if use_zf:
                    if not zf_spans:
                        continue
                    if not any(
                        _spans_overlap((local_start, local_end), zf)
                        for zf in zf_spans
                    ):
                        continue
                spans_by_concept[cid].append(
                    SpanRecord(
                        accession=accession,
                        local_start=local_start,
                        local_end=local_end,
                        global_start=start_idx + local_start,
                        global_end=start_idx + local_end,
                    )
                )

    concepts = [concept_id(spec.name) for spec in motif_specs]
    # Keep only motifs that actually fired at least once (preserve spec order).
    concepts = [cid for cid in concepts if cid in spans_by_concept]

    labels = np.zeros((total_residues, len(concepts)), dtype=bool)
    per_protein_domains: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    concept_metadata = []

    for concept_index, cid in enumerate(concepts):
        spans = spans_by_concept[cid]
        for span in spans:
            labels[span.global_start:span.global_end, concept_index] = True
            per_protein_domains[span.accession][cid].append(
                (span.local_start, span.local_end)
            )
        concept_metadata.append(
            {
                "concept": cid,
                "feature_type": "Motif",
                "description": cid.split("::", 1)[1],
                "grain": "motif",
                "n_domains": len(spans),
                "n_positive_residues": int(labels[:, concept_index].sum()),
            }
        )

    effective_zf = (
        require_zinc_finger_overlap
        if require_zinc_finger_overlap is not None
        else any(spec.require_zinc_finger_overlap for spec in motif_specs)
    )
    metadata = {
        "concepts": concepts,
        "concept_metadata": concept_metadata,
        "per_protein_domains": {
            acc: dict(by_concept) for acc, by_concept in per_protein_domains.items()
        },
        "motif_specs": [
            {
                "name": spec.name,
                "pattern": spec.pattern,
                "require_zinc_finger_overlap": (
                    False
                    if require_zinc_finger_overlap is False
                    else (
                        True
                        if require_zinc_finger_overlap is True and spec.name == "TGEKP"
                        else spec.require_zinc_finger_overlap
                    )
                ),
            }
            for spec in motif_specs
        ],
        "require_zinc_finger_overlap": bool(effective_zf),
        "n_proteins_scanned": len(proteins),
    }
    return labels, metadata


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Loading index + UniProt cache...")
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)

    annotation_records = None
    uniprot_path = config.RESULTS / "uniprot_raw.json"
    if uniprot_path.exists():
        with open(uniprot_path) as f:
            annotation_records = json.load(f)

    proteins = [(p["accession"], p["sequence"]) for p in index["proteins"]]
    offsets = {key: tuple(value) for key, value in index["offsets"].items()}
    labels, metadata = build_motif_labels(
        proteins,
        offsets,
        int(index["total_residues"]),
        annotation_records=annotation_records,
        # Default: ZF-context for TGEKP when annotations are present.
        require_zinc_finger_overlap=None,
    )

    np.savez_compressed(config.RESULTS / "motif_annotations.npz", labels=labels)
    with open(config.RESULTS / "motif_annotations_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(config.RESULTS / "concepts_motifs.json", "w") as f:
        json.dump(metadata["concept_metadata"], f, indent=2)

    print(
        f"[{time.strftime('%H:%M:%S')}] Wrote {labels.shape[1]} motif concepts "
        f"for {labels.shape[0]:,} residues:"
    )
    for concept in metadata["concept_metadata"]:
        print(
            f"  {concept['concept']:20s} domains={concept['n_domains']:5d} "
            f"residues={concept['n_positive_residues']:7d}"
        )


if __name__ == "__main__":
    main()
