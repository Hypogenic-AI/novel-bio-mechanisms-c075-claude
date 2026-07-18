# What Was Done
At a high level, my goal was to strengthen the results found in this analysis (specifically H1/H2).
The original code used a library of 15 concepts (of the original 16, with one omitted for ubiquity).
Of these, only 8 of which were covered at the F1 ≥ 0.2 level in the SAE features.
However, most of the top "dark" features in the original run were found to be known biology.
This meant that, firstly, the causal ablation run doesn't tell us much about the dark features (though it was still useful, as we'll see) and, secondly, that we don't really have a good picture of how significant the actual dark features are.
My thought was that it would be possible to obtain a more accurate result by covering a broader space of concepts.
This would also largely erase the need for an LLM judge for feature discovery, which I feel is a relative weakness of the original design, even if it was integrated into the experimental design by virtue of a fourth hypothesis.
## Changes Made
I expanded the concept dictionary from covering 16 basic concepts to covering the entire set of UniProt concepts. The dictionary was constructed as follows:
1. Cached Uniprot feature annotations were read from uniprot_raw.json and ubiquitous types were dropped.
2. Remaining types were given a `Type::ANY` label and `Type::description` subtypes, as features could activate at a coarser or finer level.
3. Infrequent types were dropped.
4. As the original report singled out DRY, NPxxY, and TGEKP sequences as "dark" features, I added another layer to annotate these as I was not sure if they were covered by the UniProt annotations.
The expanded dictionary contained 167 concepts; of these 28 were types, 136 were subtypes, and 3 were constructed annotations. 
Out of 10,240 SAE features, 1,024 reached a maximum F1 of at least 0.5 (versus 165 in the original), covering 47 distinct concepts.

For convenience, I also modified the code to run on Modal compute (as the rerun would have been inconveniently long on my machine).
## Results
As a result of the greater granularity of concepts, the number of active features with maximum F1 below 0.2 and at least 100 activating residues fell from 1,382 to 701.
28 features were covered by the additional motif layer but *not* the UniProt annotations; all were linked to the DRY motif.

### H1: Residual dark features are structured

H1 remained supported after removing features explained by the expanded dictionary. 
Of the 701 residual dark candidates, 623 (88.9%) had protein-activation entropy below the random-residue null's fifth percentile.
This is similar to the original proportion (1,207/1,382, or 87.3%), despite the elimination of explained candidates. 
Residual-dark median entropy was 3.49, compared with a null median of 5.54 and a null fifth percentile of 4.63. 
Thus, the remaining dark features still activate on a more restricted set of proteins than expected from randomly placing the same number of activations.

None of the old top 30 "dark" features remained in the candidate pool in the new run.
However, as we still see structured darks, there is stronger evidence supporting this hypothesis.

### H2: Residual dark features are causally important

H2 remained unsupported. Single-feature forward-hook ablation was repeated on the top 30 structured residual-dark features. 
None passed Benjamini-Hochberg FDR at either 0.05 or 0.10; the smallest uncorrected Wilcoxon p-value was 0.191.
Only 3/30 features had a positive median log2 ratio of KL divergence at activating versus control residues.
These results are not so suprising since, by explaining the old top features, the new top features fire more weakly/less often, and we would thus expect them to have less of a causal impact on the logits.

The median log2(KL-activating/KL-control) across features was -1.31, compared with -3.63 in the original run. 
Median KL divergence was approximately 1.2e-5 at activating residues and 4.3e-5 at controls, compared with 3.0e-4 and 2.2e-3 respectively in the original run. 
The new top features also had about half the median nonzero activation magnitude (0.28 versus 0.55), although they activated at a similar number of residues (median 200 versus 186).

We now know that after eliminating known biology concepts from the old set, ablation still fails.
Furthermore, the new dark features perturbed outputs less than the old "dark" features.

# Next Steps
However, the original results do show that a negative result in causal ablation does not mean that a feature is not biological.
The top "dark" features in the previous run were all biological and none of them were determined to causally affect outputs, so the ablation result cannot reliably be used as the standard for biological interpretability.
The divergence can be explained by the fact that the model itself is not a proxy for biological interpretability; just because some motif has a function in the protein does not mean that this particular model "utilizes" this function, even if the motif shows up in the weights.
If the model is the limitation, it would be useful to run this procedure on a larger model (as noted in the roadmap), which has the potential to encode a greater amount of structural information, but there may be other approaches.
For example, we could look at the position/prevalence of strongly activating residues within protein families, as consistency would provide evidence for functionality.
Furthermore, to eliminate the posibility of a "dark" feature being an artefact of this particular SAE, we could try to verify against different SAE architectures and model layers.

# What I Learned
To be honest, I wasn't aware of how similar the problems language and protein modeling are, but interpretability research into protein models seems like a really worthwile direction to take.
I also learned a bit more about the techniques used in interpretability research—I hadn't worked with SAEs and ablative techniques before.
I was surprised by how difficult it is to even reach a conclusion regarding the presence of dark features in the model. 
I wouldn't really expect unknown biology to be encoded in a small protein model but even refuting the hypothesis requires a lot of work!

