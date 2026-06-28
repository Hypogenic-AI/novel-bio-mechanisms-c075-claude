# Analysis: Reflection on Negative Results and Proposed Next Step

## What I Learned

What struck me first was scale: what the model learns in minutes is so much 
more than what humans learned in decades. The 80% dark feature rate is shocking 
— but the three-axis investigation reframes it methodically, confirming the 
features are real (H1), testing whether they drive predictions (H2 not 
supported), and asking whether they can be named (H4 strongly supported).

The tension between H2 and H4 is the most interesting result. The same features 
that fail causal ablation are confidently identified as real biological sub-motifs. 
This suggests a third interpretation beyond what the paper offers: the model may 
learn whatever it's given as deeply as possible, independently of what its 
prediction task actually needs.

The annotation-refinement conclusion is still meaningful. There are finer logics 
behind the logics we've already uncovered — not proving current explanations 
wrong, just incomplete. The way relativity doesn't disprove Newton but explains 
the errors that accumulate at edge cases.

## Proposed Next Step

The H2 failure is real but not conclusive — single-feature ablation is a 
potentially flawed instrument in heavily superposed models. Collinear features 
may compensate for the removed one, though this remains a hypothesis.

I propose replacing it with Lasso regression. Unlike co-ablation clustering — 
which assumes geometrically similar decoder vectors contribute similarly — Lasso 
does not take this as granted. It fits a regression predicting KL divergence 
from all 10,240 feature activations simultaneously, letting the data determine 
actual redundancy. Features with nonzero coefficients have genuinely independent 
causal signal. These get co-ablated together.

A positive result is not guaranteed. But it would be more credible. A skeleton 
implementation is included in `src/cluster_ablation.py`.
