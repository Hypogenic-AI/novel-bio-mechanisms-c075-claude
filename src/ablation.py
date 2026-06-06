"""Causal ablation experiment using a forward hook.

Approach: register a forward hook on the output of layer L (so we intercept the
hidden state *just before* it gets fed to layer L+1). Inside the hook we
subtract `act_val * W_dec[:, f]` from a target token position. The rest of the
forward pass then proceeds normally — through layers 5/6, the final layer norm,
and the LM head — yielding the new logits.

For each top dark feature we measure:
  - KL(orig || ablated) at the activating residue
  - KL(orig || ablated) at N_CONTROL_RESIDUES_PER_PROT random non-activating
    residues in the same protein (with the same-magnitude perturbation applied
    there as a controlled-magnitude null)
  - Wilcoxon signed-rank test on paired KL_activating vs mean KL_control
  - BH-FDR correction across features
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from src import config
from src.foundation_model_registry import get_model_spec


class AblationHook:
    """Hook that, when activated, subtracts `delta` from `hidden_state[:, token_pos, :]`.

    The hook is registered on the OUTPUT of a chosen encoder layer.
    """

    def __init__(self):
        self.active = False
        self.delta = None  # tensor of shape (D,)
        self.token_pos = None  # int

    def __call__(self, module, _input, output):
        if not self.active:
            return output
        # ESM layer output is the hidden state directly (a Tensor)
        if isinstance(output, tuple):
            h = output[0]
            modified = h.clone()
            modified[:, self.token_pos, :] = modified[:, self.token_pos, :] - self.delta
            return (modified, *output[1:])
        else:
            modified = output.clone()
            modified[:, self.token_pos, :] = modified[:, self.token_pos, :] - self.delta
            return modified


def main():
    with open(config.RESULTS / "index.json") as f:
        index = json.load(f)

    foundation_model_key = index.get("foundation_model_key", "esm2_8m")
    model_id = index.get("model_id", index.get("esm_model", config.ESM_MODEL))
    model_layer = int(index.get("model_layer", index.get("esm_layer", config.ESM_LAYER)))
    special_token_policy = index.get("special_token_policy", "esm")
    if special_token_policy != "esm":
        raise ValueError(
            "Ablation currently supports ESM-style token positions only; "
            f"got special_token_policy={special_token_policy!r}"
        )

    model_spec = get_model_spec(foundation_model_key)
    if not model_spec.has_public_sae or model_spec.sae_loader_key is None:
        raise ValueError(f"{foundation_model_key} has no registered InterPLM SAE for ablation")

    print(f"[{time.strftime('%H:%M:%S')}] Loading model + SAE + data...")
    print(f"  model={foundation_model_key} ({model_id}), layer={model_layer}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id)
    model.eval()
    model = model.to(config.DEVICE)

    from interplm.sae.inference import load_sae_from_hf
    sae = load_sae_from_hf(plm_model=model_spec.sae_loader_key, plm_layer=model_layer)
    sae.eval()
    sae = sae.to(config.DEVICE)

    feats = sparse.load_npz(config.RESULTS / "sae_activations.npz")
    feats_csc = feats.tocsc()
    proteins = index["proteins"]
    offsets = {k: tuple(v) for k, v in index["offsets"].items()}

    with open(config.RESULTS / "dark_features.json") as f:
        dark_meta = json.load(f)
    top_dark_idx = dark_meta["top_dark_indices"]
    print(f"  Top {len(top_dark_idx)} dark features for ablation: {top_dark_idx[:5]}...")

    # Pre-compute starts array for protein lookup
    prot_starts = [(p["accession"], p["sequence"], offsets[p["accession"]][0],
                    offsets[p["accession"]][0] + offsets[p["accession"]][1])
                   for p in proteins]
    starts = np.asarray([ps[2] for ps in prot_starts])

    # Decoder weight (D, F)
    W_dec = sae.decoder.weight.detach().to(config.DEVICE)
    expected_hidden_dim = int(index.get("hidden_dim", config.HIDDEN_DIM))
    expected_sae_dim = feats.shape[1]
    assert W_dec.shape == (expected_hidden_dim, expected_sae_dim), W_dec.shape

    # Register hook on layer model_layer-1 (zero-indexed)
    # transformer's encoder.layer[k] output = hidden_states[k+1].
    # We want to intercept the state AT model_layer, so we hook
    # encoder.layer[model_layer-1]'s output.
    hook = AblationHook()
    target_layer = model.esm.encoder.layer[model_layer - 1]
    handle = target_layer.register_forward_hook(hook)

    rng = np.random.default_rng(config.SEED)
    results: Dict[int, Dict] = {}

    # Build per-protein-sequence cache
    seq_by_acc = {p["accession"]: p["sequence"] for p in proteins}

    for feat_id in tqdm(top_dark_idx, desc="ablating features"):
        col = feats_csc.getcol(int(feat_id)).toarray().ravel()
        nonzero = np.where(col > 0)[0]
        if len(nonzero) < 5:
            continue
        sorted_nz = nonzero[np.argsort(-col[nonzero])]

        seen_proteins: set = set()
        targets: List[Tuple[str, int, float]] = []
        for r in sorted_nz:
            pi = int(np.searchsorted(starts, r, side="right") - 1)
            acc, seq, prot_start, prot_end = prot_starts[pi]
            if acc in seen_proteins:
                continue
            prot_pos = int(r - prot_start)
            if not (0 <= prot_pos < len(seq)):
                continue
            seen_proteins.add(acc)
            targets.append((acc, prot_pos, float(col[r])))
            if len(targets) >= config.N_ABLATION_PROTEINS:
                break

        if len(targets) < 5:
            continue

        kl_activating: List[float] = []
        kl_control: List[float] = []
        top1_change_activating: List[int] = []
        top1_change_control: List[int] = []

        for acc, prot_pos, act_val in targets:
            seq = seq_by_acc[acc]
            enc = tokenizer([seq], return_tensors="pt", padding=True)
            enc = {k: v.to(config.DEVICE) for k, v in enc.items()}

            # 1) original forward pass (hook inactive)
            hook.active = False
            with torch.no_grad():
                out_orig = model(**enc)
            logits_orig = out_orig.logits  # (1, L+2, V)

            token_pos = prot_pos + 1  # +1 to account for CLS

            # 2) ablation at activating residue
            hook.active = True
            hook.token_pos = token_pos
            hook.delta = act_val * W_dec[:, int(feat_id)]
            with torch.no_grad():
                out_abl = model(**enc)
            logits_abl = out_abl.logits
            hook.active = False

            # KL at activating residue
            p_o = torch.softmax(logits_orig[0, token_pos].float(), dim=-1)
            p_a = torch.softmax(logits_abl[0, token_pos].float(), dim=-1)
            kl = torch.sum(p_o * (torch.log(p_o + 1e-12) - torch.log(p_a + 1e-12))).item()
            kl_activating.append(kl)
            top1_change_activating.append(int(int(p_o.argmax()) != int(p_a.argmax())))

            # 3) control residues - random positions in same protein where the feature does NOT activate
            global_start = offsets[acc][0]
            global_len = offsets[acc][1]
            local_col = col[global_start:global_start + global_len]
            non_active = np.where(local_col == 0)[0]
            if len(non_active) < config.N_CONTROL_RESIDUES_PER_PROT:
                continue
            ctrl_positions = rng.choice(non_active, size=config.N_CONTROL_RESIDUES_PER_PROT, replace=False)

            for cp in ctrl_positions:
                ctrl_token_pos = int(cp) + 1
                hook.active = True
                hook.token_pos = ctrl_token_pos
                hook.delta = act_val * W_dec[:, int(feat_id)]
                with torch.no_grad():
                    out_c = model(**enc)
                logits_c = out_c.logits
                hook.active = False

                p_o_c = torch.softmax(logits_orig[0, ctrl_token_pos].float(), dim=-1)
                p_c = torch.softmax(logits_c[0, ctrl_token_pos].float(), dim=-1)
                kl_c = torch.sum(p_o_c * (torch.log(p_o_c + 1e-12) - torch.log(p_c + 1e-12))).item()
                kl_control.append(kl_c)
                top1_change_control.append(int(int(p_o_c.argmax()) != int(p_c.argmax())))

        if len(kl_activating) < 5:
            continue
        kl_act_arr = np.array(kl_activating)
        kl_ctl_arr = np.array(kl_control).reshape(-1, config.N_CONTROL_RESIDUES_PER_PROT).mean(axis=1)
        try:
            w_stat, w_p = stats.wilcoxon(kl_act_arr, kl_ctl_arr, alternative="greater")
        except ValueError:
            w_stat, w_p = 0.0, 1.0

        results[int(feat_id)] = {
            "n_proteins": int(len(kl_act_arr)),
            "kl_act_mean": float(np.mean(kl_act_arr)),
            "kl_act_median": float(np.median(kl_act_arr)),
            "kl_ctl_mean": float(np.mean(kl_ctl_arr)),
            "kl_ctl_median": float(np.median(kl_ctl_arr)),
            "log2_ratio_median": float(np.log2(max(np.median(kl_act_arr), 1e-9) /
                                               max(np.median(kl_ctl_arr), 1e-9))),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "top1_change_rate_activating": float(np.mean(top1_change_activating)),
            "top1_change_rate_control": float(np.mean(top1_change_control)),
        }

    handle.remove()

    # BH-FDR correction
    ids = list(results.keys())
    if ids:
        p_values = np.array([results[i]["wilcoxon_p"] for i in ids])
        order = np.argsort(p_values)
        n = len(p_values)
        ranks = np.empty(n, dtype=np.int32)
        ranks[order] = np.arange(1, n + 1)
        bh = np.minimum(1.0, p_values * n / ranks)
        # Monotone (BH step-up)
        bh_sorted = np.minimum.accumulate(bh[order][::-1])[::-1]
        q_values = np.empty(n)
        q_values[order] = bh_sorted
        for i, q in zip(ids, q_values):
            results[i]["wilcoxon_q_BH"] = float(q)
            results[i]["passes_FDR_05"] = bool(q < 0.05)
            results[i]["passes_FDR_10"] = bool(q < 0.10)

    with open(config.RESULTS / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    n_pass_05 = sum(1 for r in results.values() if r.get("passes_FDR_05", False))
    n_pass_10 = sum(1 for r in results.values() if r.get("passes_FDR_10", False))
    print(f"[{time.strftime('%H:%M:%S')}] Tested {len(results)} features; "
          f"{n_pass_05}/{len(results)} pass FDR<0.05, {n_pass_10}/{len(results)} pass FDR<0.10")
    print("  Top 10 by log2(KL_act / KL_ctl):")
    for i, r in sorted(results.items(), key=lambda kv: -kv[1]["log2_ratio_median"])[:10]:
        print(f"    feat={i}  log2_ratio={r['log2_ratio_median']:+.2f}  "
              f"KL_act={r['kl_act_median']:.4f}  KL_ctl={r['kl_ctl_median']:.4f}  "
              f"p={r['wilcoxon_p']:.2e}  q={r.get('wilcoxon_q_BH', 1.0):.2e}  "
              f"Δtop1: {r['top1_change_rate_activating']:.2f} vs {r['top1_change_rate_control']:.2f}")


if __name__ == "__main__":
    main()
