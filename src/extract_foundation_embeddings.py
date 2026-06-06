"""Extract residue-level hidden states for registered protein foundation models.

This is a model-comparison entry point for the roadmap item "explore other
foundation models." It writes the same core files used by the baseline pipeline:

  - hidden_states.npz
  - index.json
  - optionally sae_activations.npz, when an InterPLM SAE is available

The script is intentionally conservative: it supports residue-level protein
models first, because those can reuse the Swiss-Prot and ProteinGym evaluations.
Chemistry and physics candidates need different concept schemas and are tracked
in src.foundation_model_registry.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from src import config
from src.data import load_swissprot_subset
from src.foundation_model_registry import (
    get_model_spec,
    markdown_table,
    preprocess_sequence,
    ranked_candidates,
    residue_token_slice,
)


def choose_layer(model_key: str, requested_layer: int | None) -> int:
    spec = get_model_spec(model_key)
    if requested_layer is not None:
        if spec.recommended_layers and requested_layer not in spec.recommended_layers:
            layers = ", ".join(str(layer) for layer in spec.recommended_layers)
            raise ValueError(
                f"{requested_layer} is not a registered layer for {model_key}. "
                f"Expected one of: {layers}"
            )
        return requested_layer
    if not spec.recommended_layers:
        raise ValueError(f"{model_key} has no registered residue-level layer")
    return spec.recommended_layers[-1]


def load_transformer(model_key: str, device: str):
    spec = get_model_spec(model_key)
    if spec.modality != "protein":
        raise ValueError(
            f"{model_key} is a {spec.modality} model. This extractor currently "
            "supports residue-level protein models only."
        )
    if spec.model_id is None:
        raise ValueError(f"{model_key} does not define a HuggingFace model id")

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, do_lower_case=False)
    if spec.family == "ProtTrans":
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(spec.model_id)
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(spec.model_id, add_pooling_layer=False)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)
    return tokenizer, model


def extract_batch_hidden_states(
    sequences: list[str],
    model_key: str,
    tokenizer,
    model,
    layer: int,
    device: str,
) -> list[np.ndarray]:
    import torch

    spec = get_model_spec(model_key)
    processed = [preprocess_sequence(sequence, spec) for sequence in sequences]
    enc = tokenizer(
        processed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=spec.max_length,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    hidden_states = out.hidden_states
    if layer >= len(hidden_states):
        raise ValueError(
            f"Layer {layer} is unavailable for {model_key}; model returned "
            f"{len(hidden_states)} hidden-state tensors"
        )
    hidden = hidden_states[layer]

    arrays: list[np.ndarray] = []
    for i, sequence in enumerate(sequences):
        start, end = residue_token_slice(len(sequence), spec.max_length, spec.special_token_policy)
        arr = hidden[i, start:end].detach().cpu().float().numpy()
        expected_len = end - start
        if arr.shape[0] != expected_len:
            raise RuntimeError(
                f"Token slicing mismatch for {model_key}: expected {expected_len}, got {arr.shape[0]}"
            )
        arrays.append(arr)
    return arrays


def save_embedding_outputs(
    output_dir: Path,
    model_key: str,
    layer: int,
    proteins: list[tuple[str, str]],
    hidden_by_protein: list[np.ndarray],
) -> None:
    spec = get_model_spec(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets: dict[str, tuple[int, int]] = {}
    total = 0
    for (accession, _sequence), hidden in zip(proteins, hidden_by_protein):
        offsets[accession] = (total, hidden.shape[0])
        total += hidden.shape[0]

    hidden_all = np.concatenate(hidden_by_protein, axis=0).astype(np.float16)
    np.savez_compressed(output_dir / "hidden_states.npz", hidden=hidden_all)
    with open(output_dir / "index.json", "w") as f:
        json.dump(
            {
                "proteins": [
                    {"accession": accession, "sequence": sequence}
                    for accession, sequence in proteins
                ],
                "offsets": {key: list(value) for key, value in offsets.items()},
                "total_residues": total,
                "hidden_dim": int(hidden_all.shape[1]),
                "foundation_model_key": model_key,
                "model_id": spec.model_id,
                "model_family": spec.family,
                "model_layer": layer,
                "preprocessing": spec.preprocessing,
                "special_token_policy": spec.special_token_policy,
            },
            f,
            indent=2,
        )


def save_interplm_sae_outputs(
    output_dir: Path,
    model_key: str,
    layer: int,
    hidden_by_protein: Iterable[np.ndarray],
    device: str,
) -> None:
    spec = get_model_spec(model_key)
    if not spec.has_public_sae or spec.sae_loader_key is None:
        raise ValueError(f"{model_key} has no registered public InterPLM SAE")

    import torch
    from scipy import sparse

    from interplm.sae.inference import load_sae_from_hf

    sae = load_sae_from_hf(plm_model=spec.sae_loader_key, plm_layer=layer)
    sae = sae.to(device)
    sae.eval()

    sparse_features = []
    with torch.no_grad():
        for hidden in hidden_by_protein:
            hidden_t = torch.from_numpy(hidden).to(device)
            feats = sae.encode(hidden_t)
            sparse_features.append(sparse.csr_matrix(feats.cpu().float().numpy()))

    sparse.save_npz(output_dir / "sae_activations.npz", sparse.vstack(sparse_features, format="csr"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="esm2_650m", help="Registered model key")
    parser.add_argument("--layer", type=int, help="Layer to extract; defaults to the registry recommendation")
    parser.add_argument("--n-proteins", type=int, default=25, help="Swiss-Prot subset size for this run")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, help="Defaults to results/foundation_models/<model>_layer<layer>")
    parser.add_argument("--with-interplm-sae", action="store_true", help="Also encode public InterPLM SAE features")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading model weights")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    if args.list_models:
        print(markdown_table(ranked_candidates(include_baseline=True)))
        return 0

    layer = choose_layer(args.model_key, args.layer)
    spec = get_model_spec(args.model_key)
    output_dir = args.output_dir or (
        config.RESULTS / "foundation_models" / f"{args.model_key}_layer{layer}"
    )
    print(f"[{time.strftime('%H:%M:%S')}] model={args.model_key} layer={layer}")
    print(f"  model_id: {spec.model_id}")
    print(f"  output:   {output_dir}")
    print(f"  question: {spec.immediate_question}")

    if args.dry_run:
        sample = "ACDEFGHIKLMNPQRSTVWY"
        print(f"  proteins: {args.n_proteins} requested")
        print(f"  sample:   length={len(sample)} processed={preprocess_sequence(sample[:12], spec)!r}")
        print("Dry run complete; no model weights loaded.")
        return 0

    proteins = load_swissprot_subset(n=args.n_proteins)
    print(f"  proteins: {len(proteins)}")
    if proteins:
        acc, seq = proteins[0]
        print(f"  first:    {acc} length={len(seq)} processed={preprocess_sequence(seq[:12], spec)!r}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_transformer(args.model_key, device)
    hidden_by_protein: list[np.ndarray] = []
    for start in range(0, len(proteins), args.batch_size):
        batch = proteins[start:start + args.batch_size]
        accs, seqs = zip(*batch)
        print(f"[{time.strftime('%H:%M:%S')}] extracting {start + 1}-{start + len(batch)} / {len(proteins)}")
        hidden_by_protein.extend(
            extract_batch_hidden_states(list(seqs), args.model_key, tokenizer, model, layer, device)
        )

    save_embedding_outputs(output_dir, args.model_key, layer, proteins, hidden_by_protein)
    if args.with_interplm_sae:
        save_interplm_sae_outputs(output_dir, args.model_key, layer, hidden_by_protein, device)
    print(f"[{time.strftime('%H:%M:%S')}] Saved foundation-model outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
