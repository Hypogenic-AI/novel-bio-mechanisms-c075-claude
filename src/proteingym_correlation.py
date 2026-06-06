"""Test whether top dark SAE features predict mutational sensitivity (ProteinGym).

For each ProteinGym assay we have:
  - The wild-type sequence (`target_seq`)
  - Per-mutant DMS_score (continuous fitness measurement)

We compute per-residue **mutational sensitivity** as
    sensitivity[i] = mean over single-mutants at position i of |DMS_score|

We then:
  1. Run ESM-2-8M + InterPLM SAE on the WT sequence -> per-residue feature activations.
  2. For each top dark feature (and a control set of random features), compute the
     Spearman correlation between its activation and the per-residue sensitivity.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from src import config
from src.foundation_model_registry import get_model_spec


ASSAYS = [
    "BLAT_ECOLX_Stiffler_2015.csv",
    "GFP_AEQVI_Sarkisyan_2016.csv",
    "SPG1_STRSG_Olson_2014.csv",
]


def extract_position(mutant: str) -> int | None:
    """`H24Y` -> 24 (1-indexed)."""
    m = re.match(r"^[A-Z](\d+)[A-Z]$", mutant.strip())
    return int(m.group(1)) if m else None


def per_residue_sensitivity(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    sensitivity = np.zeros(seq_len, dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.int64)
    for _, row in df.iterrows():
        # Multi-mutants are separated by ":"
        mutant = str(row["mutant"]).strip()
        if ":" in mutant:
            continue
        pos = extract_position(mutant)
        if pos is None or not (1 <= pos <= seq_len):
            continue
        sensitivity[pos - 1] += abs(float(row["DMS_score"]))
        counts[pos - 1] += 1
    valid = counts > 0
    sensitivity[valid] /= counts[valid]
    return sensitivity, valid


def get_sae_activations(
    seq: str,
    model: EsmModel,
    tokenizer,
    sae,
    device: str,
    layer: int,
    max_length: int,
    special_token_policy: str,
) -> np.ndarray:
    enc = tokenizer([seq], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer]
        attn = enc["attention_mask"].bool()[0]
        h = hidden[0][attn]
        if special_token_policy == "esm":
            h = h[1:-1]
        else:
            raise ValueError(
                "ProteinGym correlation currently supports ESM-style token positions only; "
                f"got special_token_policy={special_token_policy!r}"
            )
        feats = sae.encode(h).cpu().float().numpy()
    return feats  # (L, dict_size)


def main():
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)

    foundation_model_key = index.get("foundation_model_key", "esm2_8m")
    model_id = index.get("model_id", index.get("esm_model", config.ESM_MODEL))
    model_layer = int(index.get("model_layer", index.get("esm_layer", config.ESM_LAYER)))
    special_token_policy = index.get("special_token_policy", "esm")
    model_spec = get_model_spec(foundation_model_key)
    if not model_spec.has_public_sae or model_spec.sae_loader_key is None:
        raise ValueError(f"{foundation_model_key} has no registered InterPLM SAE for ProteinGym")

    print(f"[{time.strftime('%H:%M:%S')}] Loading model + SAE...")
    print(f"  model={foundation_model_key} ({model_id}), layer={model_layer}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id, add_pooling_layer=False)
    model.eval().to(config.DEVICE)

    from interplm.sae.inference import load_sae_from_hf
    sae = load_sae_from_hf(plm_model=model_spec.sae_loader_key, plm_layer=model_layer)
    sae.eval().to(config.DEVICE)

    print(f"[{time.strftime('%H:%M:%S')}] Loading ProteinGym reference and dark features...")
    ref = pd.read_csv(config.PROTEINGYM_REF)
    with open(config.RESULTS / "dark_features.json") as f:
        dark_meta = json.load(f)
    top_dark_idx = dark_meta["top_dark_indices"]
    print(f"  Using top {len(top_dark_idx)} dark features for correlation analysis")

    # Pick a random "control" set of ANY active features (not dark)
    rng = np.random.default_rng(config.SEED)
    f1_data = np.load(config.RESULTS / "f1_sae.npz")
    F1 = f1_data["F1"]
    max_f1 = F1.max(axis=1)
    bright_idx = np.where(max_f1 >= 0.5)[0]
    bright_sample = rng.choice(bright_idx, size=min(30, len(bright_idx)), replace=False)
    # And a random set of features (any)
    random_sample = rng.choice(F1.shape[0], size=min(30, F1.shape[0]), replace=False)

    all_results = []

    for assay_name in ASSAYS:
        path = config.PROTEINGYM_DIR / assay_name
        df = pd.read_csv(path)
        rec = ref[ref["DMS_filename"] == assay_name].iloc[0]
        target_seq = rec["target_seq"]
        L = len(target_seq)
        print(f"\n[{time.strftime('%H:%M:%S')}] Assay: {assay_name}  (n_mutants={len(df)}, seq_len={L})")

        sensitivity, valid = per_residue_sensitivity(df, L)
        print(f"  covered residues: {valid.sum()}/{L}; sensitivity quantiles "
              f"50%={np.percentile(sensitivity[valid], 50):.3f}  "
              f"95%={np.percentile(sensitivity[valid], 95):.3f}")

        # ESM-2 has max length 1024, GFP=238 BLAT=286 SPG1=448 OK
        feats = get_sae_activations(
            target_seq,
            model,
            tokenizer,
            sae,
            config.DEVICE,
            model_layer,
            model_spec.max_length,
            special_token_policy,
        )
        if feats.shape[0] != L:
            print(f"  WARNING shape mismatch {feats.shape[0]} vs {L}")
            min_len = min(feats.shape[0], L)
            feats = feats[:min_len]
            sensitivity = sensitivity[:min_len]
            valid = valid[:min_len]

        sens_v = sensitivity[valid]
        # Correlate each dark feature with sensitivity (only on residues with measurements)
        def corr_for_group(idx_arr, group_name):
            results = []
            for fi in idx_arr:
                act = feats[:, int(fi)][valid]
                if (act > 0).sum() < 5:
                    continue
                rho, p = stats.spearmanr(act, sens_v)
                results.append({"feature": int(fi), "spearman_r": float(rho), "p": float(p),
                                "n_active_on_assay": int((act > 0).sum())})
            return results

        for group, idx_arr in [
            ("dark", top_dark_idx),
            ("bright", bright_sample.tolist()),
            ("random", random_sample.tolist()),
        ]:
            res = corr_for_group(idx_arr, group)
            for r in res:
                r["assay"] = assay_name
                r["group"] = group
                all_results.append(r)

    # Save
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(config.RESULTS / "proteingym_correlation.csv", index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Saved correlation results to proteingym_correlation.csv")
    print(f"  rows: {len(df_out)}")
    if len(df_out):
        # Per-group summary
        summary = df_out.groupby(["group", "assay"]).agg(
            n_features=("feature", "size"),
            mean_r=("spearman_r", "mean"),
            median_r=("spearman_r", "median"),
            p99_abs_r=("spearman_r", lambda x: float(np.percentile(np.abs(x), 99))),
            frac_p_lt_05=("p", lambda x: float((x < 0.05).mean())),
            frac_r_gt_0_2=("spearman_r", lambda x: float((x > 0.2).mean())),
        ).reset_index()
        print(summary.to_string(index=False))
        summary.to_csv(config.RESULTS / "proteingym_summary.csv", index=False)


if __name__ == "__main__":
    main()
