# Planning: Discovering Novel Biological Mechanisms from Protein Language Models

## Motivation & Novelty Assessment

### Why This Research Matters
Protein language models (ESM-2, ProtT5, etc.) achieve state-of-the-art on structure and
variant-effect prediction without ever seeing structural labels — their hidden states must
encode a great deal of biology. Recent sparse-autoencoder (SAE) interpretability work (InterPLM,
Simon & Zou, 2024) shows that ESM-2 layer-4 activations decompose into thousands of monosemantic
features, but **only ~20%** map cleanly to Swiss-Prot concepts (143/433 evaluated). The remaining
~80% of features ("dark features") may either be noise or encode mechanisms that have not yet
been catalogued by biologists. If even a small fraction of them are real, they constitute
**model-generated scientific hypotheses** — a potential new pipeline for biological discovery.

### Gap in Existing Work
The InterPLM paper enumerates dark features but does not systematically test whether they encode
biology. Three weaknesses in the current state of the art:
1. **No causal validation**: feature-concept correlation does not imply mechanistic role
   (Bricken et al., 2026; Haque et al., 2026 — Antibody SAE).
2. **No downstream functional test**: nobody has asked whether dark features predict variant
   fitness on residues they activate on.
3. **No systematic LLM-grounded annotation** of dark features that links them back to literature.

### Our Novel Contribution
We provide the **first systematic three-axis validation** of dark SAE features in a PLM:
(i) **Statistical structure** — are dark features *more* than random superposition noise?
(ii) **Causal role** — do they affect ESM-2's own predictions when ablated?
(iii) **Functional relevance** — do they correlate with experimentally measured fitness changes?
We then use a frontier LLM (Claude) to generate biological hypotheses for the top-scoring
dark features, providing a concrete handoff to experimental biologists.

### Experiment Justification
- **Experiment 1 (Sanity-reproduce InterPLM F1 alignment)**: confirm the pretrained SAE
  reproduces the known feature-concept structure on our Swiss-Prot subset. Otherwise the rest
  of the analysis is meaningless.
- **Experiment 2 (Dark feature identification + structure analysis)**: separate genuinely
  structured dark features from random/dead ones using activation entropy and consistency.
- **Experiment 3 (Causal ablation)**: zero each top dark feature in the SAE reconstruction
  path and measure perturbation of ESM-2's masked-LM logits at the very residues where the
  feature activated vs. matched control residues. A causally-relevant feature should perturb
  predictions specifically at its activating positions.
- **Experiment 4 (ProteinGym functional correlation)**: dark features that activate at
  residues with experimentally measured high fitness sensitivity (|DMS_score|) are
  functionally important even without Swiss-Prot annotation.
- **Experiment 5 (LLM-grounded hypothesis annotation)**: feed the top dark features'
  most-activating sequence windows to Claude and ask for a biological hypothesis. This is
  the *output* of the model-as-scientist pipeline.

## Research Question
Do protein language models contain SAE-decomposable internal features that
(a) do not align with any Swiss-Prot annotation,
(b) nonetheless exhibit consistent and *causally important* activation patterns, and
(c) correlate with experimentally measured fitness?

## Hypothesis Decomposition
- **H1 (structure)**: Some dark features (F1 < 0.2 vs every Swiss-Prot concept) have
  significantly more consistent activation patterns than expected under a random-feature null.
- **H2 (causal)**: Ablating a top dark feature on its activating residues changes ESM-2's
  masked-LM logits at those residues significantly more than ablating at random control
  residues.
- **H3 (functional)**: For ProteinGym DMS-assayed proteins, residues where dark features
  activate have higher mean |DMS_score| (variant sensitivity) than non-activating residues.
- **H4 (interpretable)**: An LLM, given each top dark feature's top-activating sequence windows,
  can generate consistent and plausible biological hypotheses (vs. nonsense or "no pattern").

## Proposed Methodology

### Approach
We use **pretrained InterPLM SAEs** (`Elana/InterPLM-esm2-8m`, layer 4) — the canonical 10,240-dim
ReLU SAE trained on ESM-2-8M layer-4 activations across UniRef50. This skips weeks of training
and tests our hypothesis on the exact features that have already been validated as
"interpretable in part." We use **Swiss-Prot human-reviewed proteome** (20,442 sequences, already
downloaded) as the evaluation corpus, and a subset of **ProteinGym** assays for the variant-effect
test.

### Experimental Steps
1. **Setup**: Install torch+CUDA, transformers, fair-esm, the local interPLM package, biopython,
   plus matplotlib/scikit-learn/UMAP for analysis.
2. **Load**: ESM-2-8M from HuggingFace; InterPLM layer-4 SAE from HuggingFace.
3. **Subset Swiss-Prot**: take a manageable subset (~1,500 proteins, length 50–500) from the
   human proteome — large enough for statistics, small enough to fit on one A6000 in <30 min.
4. **Extract per-residue ESM-2 layer-4 embeddings + SAE feature activations** for the subset.
   Save as a single tensor of shape (N_residues, N_features). Use mixed precision (fp16/bf16).
5. **Fetch Swiss-Prot annotations** from the UniProt REST API for the same subset proteins
   (binding sites, active sites, motifs, domains, secondary structure, disulfides, mod residues,
   transmembrane). Binarize per residue, by concept.
6. **Feature-concept F1**: for each feature × concept pair, threshold the feature at the 99th
   percentile of nonzero activations; compute precision per amino acid, recall per concept-domain
   (InterPLM's domain-adjusted recall).
7. **Sanity check (Exp 1)**: distribution of max F1 per feature should look like InterPLM's:
   a long tail of well-aligned features (>0.5) and a bulk of low-F1 features.
8. **Dark feature filter (Exp 2)**: features with max F1 < 0.2 across all concepts. Within these,
   sort by *activation consistency*: features whose top-100 activating residues come from few
   distinct proteins (low entropy) and whose mean activation is high. Drop dead features
   (mean activation ≈ 0).
9. **Causal ablation (Exp 3)**: for the top-10 surviving dark features:
   - For 30 proteins where the feature activates strongly, identify the top-activating residue.
   - Compute the SAE reconstruction of the layer-4 hidden state, *with* and *without* that
     feature's contribution (zero out its decoder column).
   - Pass both back through the rest of ESM-2 (layers 5+) and get masked-LM logits.
   - Compute KL divergence between original and ablated logit distributions at the activating
     residue, and at 10 random non-activating control residues.
   - Compare paired (activating vs control) differences with a Wilcoxon signed-rank test.
10. **Variant-effect correlation (Exp 4)**: for each of 3 ProteinGym DMS assays, compute SAE
    activations on the wild-type. For each top dark feature, compute the Pearson correlation
    between per-residue feature activation and per-residue mean |DMS_score|.
11. **LLM hypothesis generation (Exp 5)**: for each top dark feature, extract the 50 most
    strongly-activating residues with ±10 sequence context, plus the parent protein name and
    Swiss-Prot description. Feed to Claude Opus and ask for a biological hypothesis. Save
    responses verbatim.
12. **Compare against baselines**:
    - **Raw ESM-2 neurons (320-dim)**: redo Step 6 directly on the neurons (no SAE), measure
      max F1 distribution. Expect dramatically fewer concept-aligned units (this is InterPLM
      Figure 3).
    - **Shuffled SAE features**: shuffle each feature's activations across residues; redo
      ablation. Expect KL divergence at activating residues to drop to control level.

### Baselines
1. **Raw ESM-2 neuron-concept F1** — the original InterPLM headline comparison.
2. **Activation-shuffled SAE features** — null for the causal-ablation experiment.
3. **Random control residues** within each protein — null for the per-residue ablation effect.

### Evaluation Metrics
- **Primary**: number of dark features that pass both H2 (causal p < 0.01) and H3 (Pearson r > 0.2
  on at least one DMS assay).
- **Secondary**: % of dark features that are not dead (mean activation > 0.01) and structured
  (activation entropy < random null).
- **Tertiary**: SAE features (max F1 > 0.5) count vs raw neurons — sanity reproduction.

### Statistical Analysis Plan
- All p-values from non-parametric tests (Wilcoxon, Spearman) — protein activations are
  non-Gaussian and heavy-tailed.
- Multiple-testing correction: Benjamini–Hochberg FDR < 0.05 across the 10,240 features when
  scanning for novel signals.
- Effect sizes (Cohen's d for ablation KL, Pearson r for fitness correlation) reported alongside
  p-values.
- Bootstrap CIs (1,000 resamples) for the headline numbers.

## Expected Outcomes
- **Supportive**: a small but non-trivial fraction (e.g., 5–20 of 10,240) of dark features pass
  both causal and functional tests. LLM hypotheses for these features are coherent and
  literature-plausible. This supports the project's central claim.
- **Partial**: dark features show statistical structure but fail causal validation. This
  partially supports the *Interpretability without Actionability* critique and refines the
  research scope.
- **Refutation**: dark features behave like random superposition residue. The hypothesis that
  PLMs encode unknown mechanisms accessible via SAE-only is then unsupported in the small-model
  regime; larger models (ESM-2 650M, 3B) would be the natural follow-up.

## Timeline and Milestones
- Setup + data load: 20 min
- Embedding extraction (1,500 proteins, ESM-2-8M layer 4, A6000): 10 min
- Annotation fetch + F1 evaluation: 30 min
- Dark feature analysis + ablation: 45 min
- ProteinGym correlation: 20 min
- LLM annotation + figure generation + writeup: 60 min
- Total budget: ~3 hours

## Potential Challenges
- **ESM-2-8M is small**: dark features may be model artifacts, not biology. Mitigation: explicit
  random-weight and shuffled controls.
- **InterPLM SAE was trained on UniRef50, evaluated on Swiss-Prot**: distribution shift could
  exaggerate dark-feature fraction. Mitigation: focus on relative comparison vs baselines.
- **UniProt API latency**: large feature-fetch calls may hit rate limits. Mitigation: batch in
  groups of 200, exponential backoff, cache results.
- **ProteinGym sequences may not match human proteome**: we fetch SAE activations directly on
  the WT sequence from each assay (already available locally).
- **LLM hallucination on biology**: explicitly note in prompt that "no clear pattern" is an
  acceptable answer; do not require Claude to invent a story.

## Success Criteria
The research is successful if:
1. We reproduce InterPLM's sanity result (SAE features dominate raw neurons on concept-F1) —
   confirms the pipeline is correct.
2. We provide a quantitative answer to the dark-feature question (how many pass causal + fitness
   tests) — this is the primary scientific contribution.
3. We have a documented procedure that others can rerun on a larger model (ESM-2-650M, 3B) to
   scale the discovery.
