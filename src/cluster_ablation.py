"""
cluster_ablation.py

Proposed extension to src/ablation.py that addresses the single-feature ablation
limitation identified in ANALYSIS.md.

The existing ablation script (src/ablation.py) tests one SAE feature at a time.
This fails in ESM-2-8M because collinear features compensate for the removed one,
making every feature appear causally inert (median log2(KL_act/KL_ctl) = -3.86).

This script proposes a two-phase replacement:

Phase 1 - Lasso feature selection:
    Fit a Lasso regression predicting per-residue KL divergence from per-residue
    SAE activations across the full dataset. Lasso's L1 penalty zeros out collinear
    features, retaining only those with independent causal contribution.

Phase 2 - Co-ablation:
    Ablate all Lasso-selected features simultaneously in a single forward pass,
    removing the redundant signal that neighboring features currently absorb.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LassoCV
from src.config import RESULTS_DIR


def load_activations(results_dir: Path) -> np.ndarray:
    """
    Load per-residue SAE feature activations.
    Shape: (n_residues, n_features) = (422146, 10240)
    Already computed by src/extract_activations.py.
    """
    data = np.load(results_dir / "sae_activations.npz")
    return data["activations"]  # sparse: most values are zero


def compute_kl_all_residues(results_dir: Path) -> np.ndarray:
    """
    Compute per-residue KL divergence under a standard perturbation.

    The existing src/ablation.py only computes KL for 30 features x 30 proteins.
    This function extends that to all residues using the same forward-hook approach,
    applying a fixed perturbation magnitude at every residue position.

    Returns:
        kl_scores: shape (n_residues,)
    """
    # TODO: extend forward-hook ablation from src/ablation.py to all residues
    # Suggested approach:
    #   1. For each protein, register a hook on ESM-2 layer 4 output
    #   2. Apply perturbation of fixed magnitude at every residue position
    #   3. Compute KL(original_logits || perturbed_logits) at each position
    #   4. Concatenate across proteins in the same order as sae_activations.npz
    raise NotImplementedError("See src/ablation.py for single-residue implementation")


def select_causal_features(
    activations: np.ndarray,
    kl_scores: np.ndarray,
    cv: int = 5,
) -> np.ndarray:
    """
    Use Lasso regression to identify SAE features with independent causal contribution.

    Fits:
        kl_scores ~ activations @ beta  subject to L1 penalty on beta

    Lasso zeros out collinear features, retaining one representative per redundant
    cluster. Features with nonzero beta are those whose activation pattern predicts
    KL divergence independently of all other features.

    Args:
        activations: shape (n_residues, n_features)
        kl_scores:   shape (n_residues,)
        cv:          number of cross-validation folds for lambda selection

    Returns:
        important_features: indices of features with nonzero Lasso coefficients
    """
    print(f"Fitting LassoCV on {activations.shape[0]} residues x "
          f"{activations.shape[1]} features...")

    lasso = LassoCV(cv=cv, max_iter=10000, n_jobs=-1)
    lasso.fit(activations, kl_scores)

    important_features = np.where(lasso.coef_ != 0)[0]
    print(f"Selected {len(important_features)} causally independent features "
          f"(lambda={lasso.alpha_:.4f})")

    return important_features


def co_ablate_features(
    model,
    sae,
    protein_sequence: str,
    feature_indices: np.ndarray,
    target_residue: int,
) -> float:
    """
    Ablate all selected features simultaneously at a single residue position.

    Unlike single-feature ablation, this removes the combined decoder contribution
    of all collinear-redundant features at once, preventing compensation.

    Args:
        model:            ESM-2 model (transformers AutoModel)
        sae:              InterPLM SAE (loaded from HuggingFace)
        protein_sequence: amino acid string
        feature_indices:  indices of features to co-ablate (from select_causal_features)
        target_residue:   residue position to ablate at

    Returns:
        kl_divergence: KL(original_logits || co_ablated_logits) at target_residue
    """
    # TODO: implement co-ablation forward hook
    # Suggested approach:
    #   1. Compute combined perturbation vector:
    #      perturbation = sum over j in feature_indices of (activation_j * W_dec[:, j])
    #   2. Register hook on ESM-2 layer 4 output that subtracts perturbation
    #      at target_residue token position
    #   3. Run forward pass, collect logits at target_residue
    #   4. Compute and return KL divergence vs original logits
    #
    # Reference: src/ablation.py lines ~45-80 for single-feature hook implementation
    raise NotImplementedError


if __name__ == "__main__":
    activations = load_activations(RESULTS_DIR)
    kl_scores = compute_kl_all_residues(RESULTS_DIR)
    important_features = select_causal_features(activations, kl_scores)

    np.save(RESULTS_DIR / "lasso_causal_features.npy", important_features)
    print(f"Saved {len(important_features)} causal feature indices to "
          f"results/lasso_causal_features.npy")
