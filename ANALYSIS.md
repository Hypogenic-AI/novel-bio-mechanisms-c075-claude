# Analysis: Critique of Negative Results and Proposed Next Step

## Summary
This project tests whether ESM-2-8M encodes undiscovered biological mechanisms 
detectable via sparse autoencoders (SAEs). Using InterPLM's pretrained SAE, 1,500 
human Swiss-Prot proteins were evaluated across three axes: statistical structure, 
causal ablation, and LLM-grounded annotation.

## What I Learned
Dark features — SAE features with no Swiss-Prot concept alignment — are not random 
noise (H1 supported). However, single-feature causal ablation failed for all 30 
features tested (H2 not supported), and ProteinGym correlation was inconclusive due 
to insufficient statistical power (H3). Strikingly, GPT-5 identified known biological 
sub-motifs — TGEKP zinc-finger linkers and GPCR DRY/NPxxY motifs — in 10/10 top dark 
features (H4 strongly supported). The key insight is that dark features encode known 
biology at finer granularity than Swiss-Prot annotates, making SAEs annotation-
refinement tools rather than novel discovery engines in this small-model regime.

## Are the Negative Results Convincing?
The H2 failure is real but methodologically confounded. Two issues undermine the 
ablation design. First, ESM-2-8M is heavily superposed: with only 320 hidden 
dimensions encoding 10,240 SAE features, collinear features redundantly carry 
overlapping information. Removing one feature leaves its neighbors to compensate, 
making any individual feature appear causally inert regardless of its true importance. 
Second, single-feature ablation is therefore the wrong causal test — it 
systematically underestimates causal role whenever features share decoder directions.

## Proposed Next Step
Replace single-feature ablation with Lasso regression over the full residue 
population. Concretely: use per-residue SAE activations (shape: n_residues × 10,240) 
as input features and per-residue KL divergence under perturbation as the target. 
Lasso's L1 penalty drives coefficients of collinear features to exactly zero, 
selecting one representative from each redundant cluster and assigning it the full 
independent causal credit. Features with nonzero Lasso coefficients are the ones 
whose causal contribution is not recoverable from their neighbors — the true causal 
set. These should then be co-ablated simultaneously to give the causal test genuine 
sensitivity. This approach replaces a geometric heuristic with a statistically 
grounded selection procedure that directly targets the collinearity problem.
