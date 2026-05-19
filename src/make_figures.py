"""Generate the figures for the final report.

Figure 1: SAE features dominate raw neurons on Swiss-Prot concept alignment.
Figure 2: Dark-feature fraction & activation entropy vs random null.
Figure 3: Causal ablation - KL ratios for activating vs control residues.
Figure 4: ProteinGym variant-effect correlation by feature group.
Figure 5: Top dark feature activation heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

from src import config


sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


def figure1_f1_comparison():
    sae = np.load(config.RESULTS / "f1_sae.npz")
    neu = np.load(config.RESULTS / "f1_neurons.npz")
    F1_sae = sae["F1"]
    F1_neu = neu["F1"]
    concepts = list(sae["concepts"])
    n_concepts = len(concepts)
    max_per_feat = F1_sae.max(axis=1)
    max_per_neu = F1_neu.max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # (a) Histogram of max-F1
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(max_per_feat, bins=bins, alpha=0.7, label=f"SAE features (n={len(max_per_feat)})",
            color="C0", density=True)
    ax.hist(max_per_neu, bins=bins, alpha=0.7, label=f"ESM neurons (n={len(max_per_neu)})",
            color="C1", density=True)
    ax.set_xlabel("Max-F1 across Swiss-Prot concepts")
    ax.set_ylabel("Density")
    ax.set_title("(a) Per-unit best concept alignment")
    ax.legend()
    ax.set_xlim(0, 1)

    # (b) Cumulative count at F1 >= threshold
    ax = axes[1]
    thresholds = np.linspace(0.1, 0.9, 41)
    sae_counts = [(max_per_feat >= t).sum() for t in thresholds]
    neu_counts = [(max_per_neu >= t).sum() for t in thresholds]
    ax.plot(thresholds, sae_counts, "o-", color="C0", label=f"SAE features (max={sae_counts[0]})")
    ax.plot(thresholds, neu_counts, "s-", color="C1", label=f"ESM neurons (max={neu_counts[0]})")
    ax.set_yscale("symlog")
    ax.set_xlabel("F1 threshold")
    ax.set_ylabel("# units with max-F1 ≥ threshold")
    ax.set_title("(b) Concept-aligned units at varying F1 threshold")
    ax.legend()

    fig.suptitle("SAE features vs raw ESM-2-8M neurons on Swiss-Prot concept alignment",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig1_f1_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_f1_comparison.png")


def figure2_dark_feature_entropy():
    data = np.load(config.RESULTS / "dark_features.npz")
    entropies = data["dark_entropy"]
    null = data["null_entropies"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(float(null.max()), float(entropies.max())) + 0.5, 60)
    ax.hist(null, bins=bins, alpha=0.6, label=f"Null (uniform residue selection, n={len(null)})",
            color="0.5", density=True)
    ax.hist(entropies, bins=bins, alpha=0.7, label=f"Dark feature entropies (n={len(entropies)})",
            color="C2", density=True)
    p5 = float(np.percentile(null, 5))
    ax.axvline(p5, color="k", linestyle="--", lw=1.0,
               label=f"Null 5th percentile = {p5:.2f}")
    ax.set_xlabel("Protein-activation entropy (lower = more concentrated)")
    ax.set_ylabel("Density")
    ax.set_title("Dark features concentrate on fewer proteins than random")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig2_dark_entropy.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_dark_entropy.png")


def figure3_ablation():
    p = config.RESULTS / "ablation_results.json"
    if not p.exists():
        print("  (skipping fig3: ablation_results.json missing)")
        return
    with open(p) as f:
        ablation = json.load(f)
    if not ablation:
        print("  (skipping fig3: empty ablation results)")
        return
    rows = []
    for feat_id, r in ablation.items():
        rows.append(
            {
                "feature": int(feat_id),
                "kl_act": r["kl_act_median"],
                "kl_ctl": r["kl_ctl_median"],
                "log2_ratio": r["log2_ratio_median"],
                "q": r.get("wilcoxon_q_BH", 1.0),
                "passes": r.get("passes_FDR_05", False),
            }
        )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(df["kl_ctl"], df["kl_act"],
               c=df["passes"].map({True: "C3", False: "0.5"}),
               s=40, alpha=0.8)
    mx = max(df["kl_act"].max(), df["kl_ctl"].max()) * 1.1
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.set_xlabel("KL(orig||ablated) at control residues (median)")
    ax.set_ylabel("KL(orig||ablated) at activating residue (median)")
    ax.set_title("(a) Per-feature ablation effect (above diagonal = causal)")

    ax = axes[1]
    df_sorted = df.sort_values("log2_ratio")
    colors = df_sorted["passes"].map({True: "C3", False: "0.5"})
    ax.barh(np.arange(len(df_sorted)), df_sorted["log2_ratio"], color=colors)
    ax.set_yticks(np.arange(len(df_sorted)))
    ax.set_yticklabels([f"f/{i}" for i in df_sorted["feature"]], fontsize=7)
    ax.set_xlabel("log₂(KL_act / KL_ctl)  —  positive = feature is causal")
    ax.set_title("(b) Causal effect ranking (red = FDR<0.05)")

    fig.suptitle("Causal ablation of top dark SAE features", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig3_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_ablation.png")


def figure4_proteingym():
    p = config.RESULTS / "proteingym_correlation.csv"
    if not p.exists():
        print("  (skipping fig4: proteingym_correlation.csv missing)")
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"dark": "C2", "bright": "C0", "random": "0.5"}
    for group, color in palette.items():
        sub = df[df["group"] == group]
        if not len(sub):
            continue
        ax.hist(sub["spearman_r"], bins=np.linspace(-0.5, 0.7, 30),
                alpha=0.6, label=f"{group} (n={len(sub)})", color=color, density=True)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ (feature activation vs |DMS_score|)")
    ax.set_ylabel("Density")
    ax.set_title("Correlation between SAE feature activations and variant fitness sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig4_proteingym.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_proteingym.png")


def figure5_concept_coverage():
    sae = np.load(config.RESULTS / "f1_sae.npz")
    neu = np.load(config.RESULTS / "f1_neurons.npz")
    F1_sae = sae["F1"]
    F1_neu = neu["F1"]
    concepts = list(sae["concepts"])
    counts_sae = [(F1_sae[:, ci] >= 0.5).sum() for ci in range(len(concepts))]
    counts_neu = [(F1_neu[:, ci] >= 0.5).sum() for ci in range(len(concepts))]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(concepts))
    w = 0.4
    ax.bar(x - w / 2, counts_sae, w, label="SAE features", color="C0")
    ax.bar(x + w / 2, counts_neu, w, label="ESM neurons", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("# units with F1 ≥ 0.5")
    ax.set_yscale("symlog")
    ax.set_title("Per-concept coverage by SAE features vs raw ESM neurons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig5_concept_coverage.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_concept_coverage.png")


def main():
    print("Generating figures...")
    figure1_f1_comparison()
    figure5_concept_coverage()
    if (config.RESULTS / "dark_features.npz").exists():
        figure2_dark_feature_entropy()
    figure3_ablation()
    figure4_proteingym()


if __name__ == "__main__":
    main()
