"""Foundation-model candidates for the roadmap comparison experiment.

The original pipeline is intentionally narrow: ESM-2-8M hidden states plus an
InterPLM SAE. This registry makes the next comparison explicit and keeps
modality-specific caveats out of ad hoc experiment scripts.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class FoundationModelSpec:
    key: str
    family: str
    modality: str
    model_id: str | None
    representation_unit: str
    hidden_size: int | None
    recommended_layers: tuple[int, ...]
    max_length: int
    preprocessing: str
    special_token_policy: str
    has_public_sae: bool
    sae_loader_key: str | None
    comparable_to_baseline: bool
    readiness_score: int
    immediate_question: str
    known_concepts: tuple[str, ...]
    actionability_probe: str
    caveats: tuple[str, ...]


MODEL_REGISTRY: dict[str, FoundationModelSpec] = {
    "esm2_8m": FoundationModelSpec(
        key="esm2_8m",
        family="ESM-2",
        modality="protein",
        model_id="facebook/esm2_t6_8M_UR50D",
        representation_unit="amino-acid residue",
        hidden_size=320,
        recommended_layers=(4,),
        max_length=512,
        preprocessing="protein_raw",
        special_token_policy="esm",
        has_public_sae=True,
        sae_loader_key="esm2-8m",
        comparable_to_baseline=True,
        readiness_score=4,
        immediate_question="Baseline reproduced by this repository.",
        known_concepts=("Swiss-Prot residue features", "ProteinGym DMS scores"),
        actionability_probe="single-latent layer-4 forward-hook ablation",
        caveats=("Small model; the negative causal result may be scale-limited.",),
    ),
    "esm2_650m": FoundationModelSpec(
        key="esm2_650m",
        family="ESM-2",
        modality="protein",
        model_id="facebook/esm2_t33_650M_UR50D",
        representation_unit="amino-acid residue",
        hidden_size=1280,
        recommended_layers=(1, 9, 18, 24, 30, 33),
        max_length=1024,
        preprocessing="protein_raw",
        special_token_policy="esm",
        has_public_sae=True,
        sae_loader_key="esm2-650m",
        comparable_to_baseline=True,
        readiness_score=5,
        immediate_question=(
            "Does the dark-feature negative ablation result persist when model "
            "scale changes but the protein labels, SAE family, and ablation "
            "protocol stay comparable?"
        ),
        known_concepts=("Swiss-Prot residue features", "ProteinGym DMS scores"),
        actionability_probe="same InterPLM SAE ablation protocol at matched layers",
        caveats=("Larger hidden states make the full F1 sweep and ablation more expensive.",),
    ),
    "prot_t5_xl_uniref50": FoundationModelSpec(
        key="prot_t5_xl_uniref50",
        family="ProtTrans",
        modality="protein",
        model_id="Rostlab/prot_t5_xl_uniref50",
        representation_unit="amino-acid residue",
        hidden_size=1024,
        recommended_layers=(12, 24),
        max_length=1024,
        preprocessing="prot_t5",
        special_token_policy="suffix_eos",
        has_public_sae=False,
        sae_loader_key=None,
        comparable_to_baseline=False,
        readiness_score=3,
        immediate_question=(
            "Do dark residue-level motifs recur in a non-ESM architecture, or "
            "are they ESM/InterPLM-specific?"
        ),
        known_concepts=("Swiss-Prot residue features", "ProteinGym DMS scores"),
        actionability_probe="train a matched SAE or compare raw-neuron F1 before ablation",
        caveats=(
            "Requires a new SAE or a raw-hidden-state baseline.",
            "ProtT5 tokenization replaces rare amino acids with X and spaces residues.",
        ),
    ),
    "chemberta_77m_mlm": FoundationModelSpec(
        key="chemberta_77m_mlm",
        family="ChemBERTa",
        modality="chemistry",
        model_id="DeepChem/ChemBERTa-77M-MLM",
        representation_unit="SMILES token or molecular substructure",
        hidden_size=None,
        recommended_layers=(6, 12),
        max_length=512,
        preprocessing="smiles",
        special_token_policy="tokenizer_default",
        has_public_sae=False,
        sae_loader_key=None,
        comparable_to_baseline=False,
        readiness_score=2,
        immediate_question=(
            "Can an analogous dark-feature workflow recover under-annotated "
            "chemical substructures instead of protein motifs?"
        ),
        known_concepts=("RDKit substructure labels", "MoleculeNet property labels"),
        actionability_probe="mask/substitute substructures and measure property-head changes",
        caveats=(
            "Not residue-level; concept alignment must move from UniProt spans to molecular graphs.",
            "SMILES tokens do not map one-to-one to atoms without extra bookkeeping.",
        ),
    ),
    "mace_mp": FoundationModelSpec(
        key="mace_mp",
        family="MACE",
        modality="physics/materials",
        model_id=None,
        representation_unit="atom or local atomic environment",
        hidden_size=None,
        recommended_layers=(),
        max_length=0,
        preprocessing="atomic_structure",
        special_token_policy="not_applicable",
        has_public_sae=False,
        sae_loader_key=None,
        comparable_to_baseline=False,
        readiness_score=1,
        immediate_question=(
            "Can sparse features in atomistic foundation models identify local "
            "environments tied to stability, forces, or defects?"
        ),
        known_concepts=("coordination number", "local geometry", "formation energy"),
        actionability_probe="perturb atomic environments and evaluate energy/force consistency",
        caveats=(
            "Requires a separate structural dataset and graph/geometry feature alignment.",
            "Mechanisms are physical state variables, not sequence motifs.",
        ),
    ),
}


def get_model_spec(key: str) -> FoundationModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        options = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model key {key!r}. Available keys: {options}") from exc


def ranked_candidates(include_baseline: bool = False) -> list[FoundationModelSpec]:
    specs = list(MODEL_REGISTRY.values())
    if not include_baseline:
        specs = [spec for spec in specs if spec.key != "esm2_8m"]
    return sorted(
        specs,
        key=lambda spec: (
            spec.readiness_score,
            spec.comparable_to_baseline,
            spec.has_public_sae,
            spec.modality == "protein",
        ),
        reverse=True,
    )


def preprocess_sequence(sequence: str, spec: FoundationModelSpec) -> str:
    """Apply the model-specific sequence representation expected by tokenizers."""
    if spec.preprocessing == "protein_raw":
        return sequence
    if spec.preprocessing == "prot_t5":
        normalized = re.sub(r"[UZOB]", "X", sequence)
        return " ".join(normalized)
    if spec.preprocessing in {"smiles", "atomic_structure"}:
        return sequence
    raise ValueError(f"Unsupported preprocessing mode: {spec.preprocessing}")


def residue_token_slice(
    sequence_length: int,
    max_length: int,
    special_token_policy: str,
) -> tuple[int, int]:
    """Return the token slice containing residue-level representations.

    ESM tokenizers add a prefix CLS token and a suffix EOS token. ProtT5 uses a
    suffix EOS token after one token per spaced residue. The returned bounds are
    half-open and account for truncation.
    """
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if special_token_policy == "esm":
        if max_length < 3:
            raise ValueError("ESM extraction needs room for CLS, at least one residue, and EOS")
        kept = min(sequence_length, max_length - 2)
        return 1, 1 + kept
    if special_token_policy == "suffix_eos":
        if max_length < 2:
            raise ValueError("suffix-EOS extraction needs room for at least one residue and EOS")
        kept = min(sequence_length, max_length - 1)
        return 0, kept
    if special_token_policy in {"tokenizer_default", "not_applicable"}:
        kept = min(sequence_length, max_length) if max_length else sequence_length
        return 0, kept
    raise ValueError(f"Unsupported special token policy: {special_token_policy}")


def specs_as_dicts(specs: Iterable[FoundationModelSpec]) -> list[dict]:
    return [asdict(spec) for spec in specs]


def markdown_table(specs: Iterable[FoundationModelSpec]) -> str:
    headers = [
        "key",
        "modality",
        "model",
        "unit",
        "public SAE",
        "readiness",
        "immediate question",
    ]
    rows = [
        [
            spec.key,
            spec.modality,
            spec.model_id or spec.family,
            spec.representation_unit,
            "yes" if spec.has_public_sae else "no",
            str(spec.readiness_score),
            spec.immediate_question,
        ]
        for spec in specs
    ]
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        table.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--model-key", help="Show one model spec instead of the ranked table")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)

    if args.model_key:
        specs = [get_model_spec(args.model_key)]
    else:
        specs = ranked_candidates(include_baseline=args.include_baseline)

    if args.format == "json":
        print(json.dumps(specs_as_dicts(specs), indent=2))
    else:
        print(markdown_table(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
