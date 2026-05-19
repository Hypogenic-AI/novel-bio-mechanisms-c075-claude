"""Generate biological hypotheses for top dark SAE features using Claude.

For each top dark feature, extract the top-activating sequence windows
(±10 residues around the maximum-activation residue, in each of the
top-activating proteins), plus UniProt protein names if available, and
ask Claude Sonnet for a biological hypothesis.

The prompt allows "no clear pattern" as a valid answer (anti-hallucination).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from src import config


CLAUDE_MODEL = "claude-sonnet-4-5"
OPENAI_MODEL = "gpt-5"  # fallback if Anthropic key unavailable

PROMPT_TEMPLATE = """You are a structural biologist examining a candidate "feature"
extracted by a sparse autoencoder from the protein language model ESM-2-8M (layer 4).
This feature does NOT match any well-characterised Swiss-Prot annotation category
(active site, binding site, disulfide, motif, domain, transmembrane, etc.) at our
F1 > 0.2 threshold, yet it fires consistently on a small set of human proteins.

Below are the top-activating residue windows for this feature, one per protein.
Each window is a ±10-residue context with the maximally-activating residue marked
**in brackets** and the activation value in parentheses.

Your task:
1. Look at the windows.  Is there a *consistent* biological pattern across them?
   Look for: sequence motifs, charge/hydrophobicity patterns, known but
   under-annotated structural features (e.g., turn types, capping motifs,
   linker classes, IDR sub-patterns), evolutionary conservation hints, or
   modifications.
2. If you find one, state a single concrete biological hypothesis.  Include:
   (a) The pattern you noticed (sequence, biophysical, or contextual).
   (b) The candidate biological role.
   (c) How you would experimentally test the hypothesis in 1 sentence.
3. If you do NOT find a clear pattern, say so explicitly — do not invent a story.
4. Rate your confidence: low / medium / high.

=== FEATURE {feat_id} — top-{n_windows} activating proteins ===
{windows}
"""


def build_windows_for_feature(
    feat_id: int,
    feats_csc: sparse.csc_matrix,
    proteins: list,
    offsets: dict,
    n_windows: int = 20,
    context: int = 10,
) -> List[str]:
    col = feats_csc.getcol(int(feat_id)).toarray().ravel()
    nonzero = np.where(col > 0)[0]
    if len(nonzero) == 0:
        return []
    # Sort residues by activation
    sorted_nz = nonzero[np.argsort(-col[nonzero])]

    starts = np.asarray([offsets[p["accession"]][0] for p in proteins])
    lengths = np.asarray([offsets[p["accession"]][1] for p in proteins])

    seen = set()
    windows = []
    for r in sorted_nz:
        pi = int(np.searchsorted(starts, r, side="right") - 1)
        if pi < 0 or pi >= len(proteins):
            continue
        acc = proteins[pi]["accession"]
        seq = proteins[pi]["sequence"]
        if acc in seen:
            continue
        prot_pos = int(r - starts[pi])
        if not (0 <= prot_pos < len(seq)):
            continue
        seen.add(acc)
        s = max(0, prot_pos - context)
        e = min(len(seq), prot_pos + context + 1)
        left = seq[s:prot_pos]
        right = seq[prot_pos + 1:e]
        center = seq[prot_pos]
        win = f"{acc}  pos={prot_pos+1}  act={col[r]:.2f}  ...{left}[{center}]{right}..."
        windows.append(win)
        if len(windows) >= n_windows:
            break
    return windows


def query_llm(prompt: str) -> Tuple[str, str]:
    """Query the LLM and return (model_used, response_text)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return CLAUDE_MODEL, resp.content[0].text
    elif os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return OPENAI_MODEL, resp.choices[0].message.content
    else:
        raise RuntimeError("No LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) set")


def main():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("WARNING: No LLM API key set; skipping LLM annotation step.")
        return

    print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")
    feats_csc = feats.tocsc()
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)
    proteins = index["proteins"]
    offsets = {k: tuple(v) for k, v in index["offsets"].items()}

    with open(config.RESULTS / "dark_features.json") as f:
        dark_meta = json.load(f)
    top_dark_idx = dark_meta["top_dark_indices"][:10]
    print(f"  Annotating {len(top_dark_idx)} top dark features...")

    responses: Dict[int, dict] = {}
    for feat_id in top_dark_idx:
        windows = build_windows_for_feature(feat_id, feats_csc, proteins, offsets,
                                             n_windows=15, context=10)
        if len(windows) == 0:
            continue
        prompt = PROMPT_TEMPLATE.format(
            feat_id=feat_id,
            n_windows=len(windows),
            windows="\n".join(windows),
        )
        print(f"  Asking LLM about feature {feat_id} ({len(windows)} windows)...")
        try:
            model_used, response = query_llm(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR: {e}")
            model_used, response = "error", f"[error] {e}"
        responses[int(feat_id)] = {
            "feature_id": int(feat_id),
            "n_windows": len(windows),
            "windows": windows,
            "model": model_used,
            "response": response,
        }

    with open(config.RESULTS / "llm_annotations.json", "w") as f:
        json.dump(responses, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Saved {len(responses)} annotations.")


if __name__ == "__main__":
    main()
