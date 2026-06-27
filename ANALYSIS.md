# Analysis: Critique of Negative Results and Proposed Next Step

## Summary
This project tests whether ESM-2-8M, a protein language model, encodes undiscovered biological mechanisms detectable via sparse autoencoders (SAEs). Using InterPLM's pretrained SAE, 1,500 human proteins were evaluated across three axes: statistical structure, causal ablation, and LLM-grounded annotation.

## What I Learned
Dark features — SAE features with no Swiss-Prot concept alignment — are not random noise (H1 supported). However, single-feature causal ablation failed for all 30 features tested (H2 not supported), and ProteinGym correlation was inconclusive due to insufficient statistical power (H3). Strikingly, GPT-5 confidently identified known biological sub-motifs — TGEKP zinc-finger linkers and GPCR DRY/NPxxY motifs — in 10/10 top dark features (H4 strongly supported). The key insight is that dark features encode known biology at finer granularity than Swiss-Prot annotates, making SAEs annotation-refinement tools rather than novel discovery engines in this regime.

## Are the Negative Results Convincing?
The H2 failure is real but not conclusive. Two methodological confounds undermine the ablation design. First, ESM-2-8M is heavily superposed: with only 320 hidden dimensions encoding 10,240 SAE features, collinear features redundantly carry overlapping information. Removing one feature leaves neighboring features to compensate, making any individual feature appear causally inert regardless of its true importance. Second, single-feature ablation is therefore the wrong causal test for this architecture — it systematically underestimates causal role in small, superposed models.

## Proposed Next Step
Co-ablate clusters of collinear features rather than individual features. Group SAE decoder vectors by cosine similarity and ablate all members of a cluster simultaneously. This removes the redundant signal that neighboring features currently absorb, giving the causal test genuine sensitivity. If cluster-level ablation resurfaces disruption at activating residues, it would directly resolve the central ambiguity in this report and reopen the question of whether dark features are mechanistically important.
