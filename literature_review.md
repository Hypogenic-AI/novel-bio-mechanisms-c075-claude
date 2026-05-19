# Literature Review

**Project**: Discovering Novel Biological Mechanisms from Protein Language Models
**Hypothesis**: Protein language models contain internal representations that correspond to
previously unknown biological mechanisms. By identifying latent features or circuits that strongly
influence protein function predictions but do not align with existing biological annotations, it is
possible to uncover new motifs, interaction patterns, or evolutionary constraints not yet
characterized experimentally.

## 1. Research Area Overview

Protein language models (PLMs) are transformer-based self-supervised models trained on tens to
hundreds of millions of natural protein sequences. The flagship ESM-2 family (Lin et al., 2022,
8M–15B params) achieves state-of-the-art on contact prediction, structure prediction (via
ESMFold), and variant-effect prediction, without ever seeing structural labels. **The internal
representations of these models are known to encode rich structural, functional, and evolutionary
information**, but how that information is *organized* inside the network is only beginning to be
understood.

A small but rapidly growing line of work (2024–2026) applies **mechanistic interpretability**
tools — primarily sparse autoencoders (SAEs), but also concept activation vectors (CAVs), attention
analysis, and cross-layer transcoders — to PLMs. The dominant finding is that, like LLM neurons,
**PLM neurons are polysemantic**: a single neuron mixes many unrelated concepts via superposition.
Decomposing those neurons with SAEs produces thousands of "monosemantic" latent features per layer,
many of which align cleanly with known biology (catalytic active sites, zinc fingers, disulfide
bonds, beta barrels, kinase binding loops, mitochondrial targeting sequences, phosphorylated
residues, disordered regions, etc.).

Critically, the seminal **InterPLM** paper (Simon & Zou, 2024) reports that of ~10,420 SAE features
extracted per ESM-2 layer, up to **2,548 features per layer have F1 > 0.5 against at least one
Swiss-Prot concept** — but the *total* number of Swiss-Prot concepts that any feature aligns with
is only 143 out of 433 evaluated. **The remaining ~80% of features have *no* strong alignment to
any known annotation, yet they exhibit consistent activation patterns** (often clustering in
3D space within a protein, or activating across structurally homologous proteins). These are
the candidates for novel mechanisms that this research project targets.

The line of work has now branched in several directions:
- **SAE architecture variants** (TopK, JumpReLU, Ordered/Matryoshka) trading off sparsity vs
  feature granularity and steerability.
- **Cross-layer transcoders / circuit tracing** for following information through the network
  rather than just inventorying features at a single layer.
- **Downstream-task validation**: do SAE features improve fitness prediction, variant-effect
  prediction, or generative design? (Yes, sometimes substantially.)
- **Domain-specific PLMs**: antibodies, enzymes, IDPs.
- **Caveats**: Bricken-style "interpretability without actionability" concerns that
  feature-concept correlations don't imply causal control.

## 2. Key Papers

### Paper 1: InterPLM (Simon & Zou, 2024) — **CORE**

- **Authors**: Elana Simon, James Zou (Stanford)
- **Year**: 2024 (Nov), published in Nature Methods 2025
- **arXiv**: 2412.12101 · 92 citations
- **File**: `papers/InterPLM_Discovering_Interpretable_Features_in_PLMs_via_SAE.pdf`
- **Key contribution**: First systematic framework for extracting and analyzing SAE features from
  protein language models. Demonstrates that ESM-2 represents most biological concepts *in
  superposition* (only 46 neurons/layer cleanly map to 15 concepts), while SAE expansion (320 →
  10,420 latents) reveals up to 2,548 monosemantic features per layer covering 143 concepts.
- **Methodology**:
  - Train ReLU SAE on per-residue ESM-2 embeddings, expansion factor 32× (320 → 10,240, +180 extra).
  - Loss: MSE reconstruction + L1 sparsity. λ = 0.08–0.1; renormalize decoder rows to unit norm
    after each step.
  - Layer-wise: 6 separate SAEs, one per ESM-2-8M layer. Best concept-association at layer 4.
  - Evaluation metric: **modified F1** — precision per amino acid, recall *per domain*, because
    multi-residue concepts (like a 200-aa kinase domain) penalize features that activate on only
    one conserved position within the domain. This is the InterPLM-style evaluation that every
    follow-up paper uses.
  - **Structural vs sequential clustering**: for each feature's top activating residues in proteins
    with AlphaFold structures, compute mean activation in ±2 sequence neighborhood vs ≤6 Å
    3D neighborhood. Compare to permuted null. Cohen's d effect sizes classify each feature.
  - **Automated feature description**: Claude-3.5 Sonnet given top-activating protein sequence
    windows, asked to write a biological hypothesis for what the feature represents.
- **Datasets**: UniRef50 (training, ~5M tokens), Swiss-Prot (concept evaluation, 433 concepts
  binarized per residue), AlphaFold-DB (structural-proximity analysis).
- **Results**:
  - SAE features (1.2K–2.5K per layer at F1 > 0.5) >> ESM neurons (8–46 per layer).
  - Hierarchical relationships: e.g., feature f/1503 detects TBDR beta barrels specifically, while
    f/2469 detects all transmembrane sugar-channel barrels. Sub-clusters within "kinase binding
    site" features distinguish active vs pseudo-kinase residues.
  - Steering: clamping a feature to its 90th-percentile activation propagates and changes
    predicted logits at *other* positions, consistent with a causal/circuit role.
  - Novel motif: glycosyltransferase cluster (f/9047 family) — features fire consistently on a
    cluster of glycosyltransferases despite no Swiss-Prot annotation linking these proteins to a
    shared motif.
- **Code**: <https://github.com/ElanaPearl/interPLM>. **Pretrained SAEs on HuggingFace** for all
  layers of ESM-2-8M (1–6) and ESM-2-650M (1, 9, 18, 24, 30, 33).
- **Relevance**: This is the methodological foundation for the entire research project. The
  pretrained SAEs let us skip the costly training step and go straight to feature analysis.

### Paper 2: Sparse Autoencoders Uncover Biologically Interpretable Features in PLMs (Adams et al., 2025)

- **PNAS**, 39 citations, paper-finder relevance score 3.
- **Why it matters**: Parallel/independent demonstration of the same hypothesis as InterPLM,
  published in PNAS — gives the field stronger external validation. (PDF not downloadable due to
  publisher restrictions; abstract retrieved.)
- **Relevance**: Confirms reproducibility of the SAE-on-PLM finding. Cite as supporting evidence
  in any writeup; not strictly needed for implementation since InterPLM is more thorough.

### Paper 3: Interpreting and Steering PLMs through Sparse Autoencoders (Villegas Garcia & Ansuini, 2025)

- **arXiv**: 2502.09135 · 16 citations
- **File**: `papers/Interpreting_Steering_PLMs_through_Sparse_Autoencoders.pdf`
- **Key contribution**: Adds two methodological refinements: (a) layer selection via **intrinsic
  dimension plateau** (Facco et al. 2017 estimator) rather than the typical "use the middle layer"
  heuristic, and (b) demonstration that *zinc finger domains* can be reliably steered by SAE-latent
  manipulation. Uses 0.80 precision/recall threshold for feature-concept association.
- **Datasets**: SCOPe 2.08 (~15K non-redundant sequences for training), UniProt for concept labels.
- **Relevance**: The intrinsic-dimension heuristic is directly transferable — saves having to train
  SAEs for every layer and pick the best one in hindsight.

### Paper 4: Sparse Autoencoders for Low-N Protein Function Prediction and Design (Tsui, Talreja, & Aghazadeh, 2025)

- **arXiv**: 2508.18567 · paper-finder relevance score 3
- **File**: `papers/Sparse_Autoencoders_LowN_Protein_Function_Prediction.pdf`
- **Key contribution**: Repositions SAEs from "interpretability tool" to "predictor". Shows that
  SAEs trained on **LoRA-fine-tuned ESM-2-650M embeddings** outperform raw ESM-2 baselines in 58%
  of low-N fitness extrapolation tasks (N = 8, 24, 96, 384). Steering predictive SAE latents yields
  top-fitness variants in 83% of cases vs ESM-2 direct design.
- **Methodology**:
  - SAEs with **TopK activation** (k = 128 per residue position), dictionary size 4,096, expansion
    8× over the 1,280-dim ESM-2-650M hidden.
  - Auxiliary loss: reconstruction error of top-256 dead latents to mitigate dead-latent collapse.
  - Train a linear probe (Lasso) on mean-pooled SAE latents to predict fitness.
  - 5 extrapolation regimes: random, position, mutation, regime, score.
- **Datasets**: 6 DMS assays from **ProteinGym** (GFP, GB1×2, PDZ, SH3, antitoxin ParD3).
- **Code**: <https://github.com/amirgroup-codes/LowNSAE>
- **Relevance**: Provides the downstream-validation protocol. If our "novel" SAE features really
  capture biology, they should improve fitness prediction *exactly* on residues they activate on.
  This is the most natural quantitative test of the research hypothesis.

### Paper 5: Mechanistic Interpretability of Antibody Language Models Using SAEs (Haque et al., 2025/2026)

- **arXiv**: 2512.05794 · 1 citation
- **File**: `papers/Mechanistic_Interpretability_Antibody_LMs_Using_SAEs.pdf`
- **Key contribution**: Compares **TopK SAEs vs Ordered SAEs (O-SAEs)** on autoregressive antibody
  models (p-IgGen, IgLM, ProGen2-OAS). Finds that TopK SAEs identify interpretable features but
  high feature-concept correlation **does not guarantee causal steerability**. O-SAEs (nested
  hierarchical structure with monotonically decreasing truncation weights) impose explicit
  hierarchy and reliably yield steerable latents.
- **Datasets**: Observed Antibody Space (OAS, 1.8M paired VH/VL sequences), CoV-AbDab, PLAbDab.
- **Code**: Built on `EleutherAI/sparsify`.
- **Relevance**: Two takeaways: (1) we should be cautious about treating high feature-concept F1
  as proof of mechanistic role — causal intervention is the stronger test; (2) Ordered SAE
  architecture is worth comparing against vanilla TopK for our experiments.

### Paper 6: Automated Protein Motif Localization using Concept Activation Vectors in PLM Embedding Space (Shamail & McWhite, 2025)

- **arXiv**: 2511.21614 · paper-finder relevance score 3
- **File**: `papers/Automated_Protein_Motif_Localization_CAVs_PLM.pdf`
- **Key contribution**: **Alternative to SAEs.** Uses Concept Activation Vectors (linear classifiers
  in PLM embedding space) instead of SAE features. For each motif, train a logistic regressor to
  separate motif-containing from non-motif windowed embeddings; the classifier weight vector is
  the "CAV" for that motif. Scoring is then a single inner product per window. Achieves >85% F1
  on a curated 69-motif benchmark, with 90% precision / 80% recall.
- **Datasets**: 69 well-characterized Pfam motifs, ESM-C embeddings.
- **Relevance**: Provides a *supervised* lens on the same question. SAE features that don't match
  any CAV → strong novelty signal (one independent method failed to recognize them as known).
  Also: their **motif-as-direction** geometric framing is what we'd want for any "candidate novel
  motif" CAV training: take an SAE feature's top-activating sequences as positives, random
  sequences as negatives, train CAV, check if it generalizes.

### Paper 7: Insights into the Inner Workings of Transformer Models for Protein Function Prediction (Stark et al., 2023)

- **arXiv**: 2309.03631 · 16 citations
- **File**: `papers/Insights_Inner_Workings_Transformer_Protein_Function.pdf`
- **Key contribution**: Pre-SAE-era interpretability for PLMs using **integrated gradients and
  attention attribution** on the protein function prediction task (Gene Ontology terms). Shows
  that PLM attention heads systematically attend to functionally critical residues — providing
  the historical evidence that "PLMs encode known biology" *before* the SAE-based methods could
  enumerate which biology specifically.
- **Relevance**: Useful as historical/baseline context. The attention-based approach is now
  largely superseded by SAE methods, but the integrated-gradients setup is still a clean way to
  validate that an SAE feature has *causal* relevance to a specific downstream prediction.

### Paper 8: BERTology Meets Biology — Interpreting Attention in PLMs (Vig et al., 2020)

- **arXiv**: 2006.15222 · 359 citations
- **File**: `papers/BERTology_Meets_Biology_Interpreting_Attention_PLMs.pdf`
- **Key contribution**: First systematic interpretability study of PLMs (TAPE / pre-ESM-2 era).
  Shows specific attention heads in pre-trained transformer protein models correspond to **contact
  maps** and **binding sites**. Establishes that "PLMs encode biology, even before any
  fine-tuning."
- **Relevance**: Seminal context. The attention-head methodology has been largely displaced by
  SAEs but the result (PLMs encode contacts and binding sites) is foundational.

### Paper 9: Protein Circuit Tracing via Cross-layer Transcoders (2026)

- **arXiv**: 2602.12026 · 0 citations
- **File**: `papers/Protein_Circuit_Tracing_Cross_Layer_Transcoders.pdf`
- **Key contribution**: Moves beyond per-layer feature inventories to actual **circuit analysis**
  — composing transcoders across layers to identify multi-step computations inside ESM-2. This is
  the natural next step after SAE-based feature extraction.
- **Relevance**: The most recent (2026) advance and the most ambitious in scope. For our project,
  cross-layer transcoders are a stretch goal; per-layer SAEs are the realistic starting point.

### Paper 10: Induction Meets Biology — Mechanisms of Repeat Detection in PLMs (2026)

- **arXiv**: 2602.23179 · 0 citations
- **File**: `papers/Induction_Meets_Biology_Repeat_Detection_PLMs.pdf`
- **Key contribution**: Identifies that PLMs implement **induction-head-like circuits** for repeat
  detection, analogous to the induction heads in LLMs (Olsson et al. 2022). Demonstrates a
  specific computational primitive shared across modalities.
- **Relevance**: Excellent example of "novel mechanism" — they identified a circuit, not just a
  feature, and connected it to a known computational primitive from NLP. This is the kind of
  finding the research project aspires to produce.

### Paper 11: Automated Neuron Labelling Enables Generative Steering and Interpretability in PLMs (Parsan et al., 2025)

- **arXiv**: 2507.06458 · 0 citations
- **File**: `papers/Automated_Neuron_Labelling_Generative_Steering_PLMs.pdf`
- **Key contribution**: Pipeline for **LLM-based automated labelling** of PLM neurons/SAE features
  followed by steerable generation in ProGen2. Less novel architecturally than InterPLM but
  productionizes the LLM-assisted annotation loop.
- **Relevance**: Direct template for the "automated annotation" step in our pipeline.

### Paper 12: ProtSAE — Disentangling and Interpreting PLMs via Semantically-Guided Sparse Autoencoders (2025)

- **arXiv**: 2509.05309 · 2 citations
- **File**: `papers/ProtSAE_Disentangling_Interpreting_PLMs_Semantically.pdf`
- **Key contribution**: Adds **semantic regularization** to the SAE objective — penalizes features
  that don't align with at least one of a curated set of biological concept embeddings. Trades
  some "novelty" potential for higher per-feature interpretability.
- **Relevance**: Cautionary lens: too much semantic regularization would *prevent* discovery of
  novel mechanisms. Useful as a contrasting baseline.

### Paper 13: Towards Interpretable Protein Structure Prediction with SAEs (2025)

- **arXiv**: 2503.08764 · 11 citations
- **File**: `papers/Towards_Interpretable_Protein_Structure_Prediction_with_SAEs.pdf`
- **Key contribution**: SAEs applied to ESMFold structure-prediction pathway. Identifies SAE
  latents responsible for specific local structural predictions.
- **Relevance**: Complementary axis — our project focuses on *general* features, but features
  that drive *structure prediction* are a particularly testable class.

### Paper 14: Hierarchical Semantics in Sparse Autoencoder Architectures (2025)

- **arXiv**: 2506.01197
- **File**: `papers/Hierarchical_Semantics_in_SAE_Architectures.pdf`
- **Key contribution**: Matryoshka/nested SAE architectures from the NLP mech-interp community.
  Related to Ordered SAEs used in the antibody paper.
- **Relevance**: Methodological reference for hierarchical SAEs.

### Paper 15: Interpretability without Actionability — Mechanistic Methods Cannot Correct LM Errors (2026)

- **arXiv**: 2603.18353
- **File**: `papers/Interpretability_without_Actionability_Mechanistic.pdf`
- **Key contribution**: Critical paper. Argues that mechanistic-interpretability findings have
  often failed to translate into actionable model edits. Specific to LLMs but applicable here.
- **Relevance**: Caveat. Our claim that an SAE feature represents a "novel mechanism" must be
  backed by *more than feature-concept correlation* — ideally by causal intervention (steering
  experiment) and/or downstream improvement (fitness prediction).

### Paper 16: ProtTrans — Cracking the Language of Life's Code (Elnaggar et al., 2020)

- **arXiv**: 2007.06225 · 1,276 citations
- **File**: `papers/ProtTrans_Cracking_Language_Life_Self_Supervised.pdf`
- **Key contribution**: Foundational paper introducing ProtT5, ProtBERT, ProtAlbert family.
  Alternative to ESM-2 for cross-model comparison.

### Paper 17: Tranception (Notin et al., 2022)

- **arXiv**: 2205.13760
- **File**: `papers/Tranception_Protein_Fitness_Prediction.pdf`
- **Key contribution**: Autoregressive PLM specifically for variant-effect prediction (the
  motivation behind ProteinGym).

### Paper 18: Fine-tuning ESM2 to Understand the Functional Impact of Missense Variants (2024)

- **arXiv**: 2410.10919
- **File**: `papers/Fine_Tuning_ESM2_Missense_Variants_Functional_Impact.pdf`
- **Key contribution**: Practical fine-tuning recipe for ESM-2 on variant-effect labels, used by
  the LowNSAE paper as the base model for SAE training.

### Paper 19: Central Dogma Transformer III — Interpretable AI Across DNA, RNA, and Protein (2026)

- **arXiv**: 2603.23361
- **File**: `papers/Central_Dogma_Transformer_III_DNA_RNA_Protein.pdf`
- **Key contribution**: Multi-modal protein/DNA/RNA model with built-in interpretability.

### Paper 20: Protein Generation with Embedding Learning for Motif Diversification (2025)

- **arXiv**: 2510.18790
- **File**: `papers/Protein_Generation_Embedding_Learning_Motif_Diversification.pdf`
- **Key contribution**: Generative approach using PLM embeddings to design diverse motif-bearing
  proteins — useful for closing the loop from "discovered feature" → "designed novel protein
  containing that feature".

## 3. Common Methodologies

| Approach | Used by | Strengths | Limitations |
|---|---|---|---|
| **Sparse Autoencoders (SAEs)** | InterPLM, ProtSAE, LowNSAE, Antibody-SAE, Adams et al. | Unsupervised; discovers thousands of features per layer; aligns well with known concepts; steerable | Many "dead" latents; high feature-concept F1 ≠ causal control; expansion factor and L1 weight are sensitive hyperparameters |
| **Concept Activation Vectors (CAVs)** | Shamail & McWhite 2025 | Each motif is one vector; trivially extensible; cheap inference; interpretable | Supervised — needs positive/negative examples; can't discover unknown concepts |
| **Attention attribution** | Vig et al. 2020, Stark et al. 2023, MSA Transformer interp | Pre-SAE-era classic; reveals contact / binding-site heads | Per-head granularity is coarse; many heads are uninterpretable |
| **Integrated gradients / probing** | Many | Quantifies which input residues drive a prediction | Confounds feature presence with predictive importance |
| **Cross-layer transcoders / circuit tracing** | 2026 Protein Circuit Tracing paper | Reveals multi-layer computation, not just per-layer features | Compute-intensive; still cutting-edge |

## 4. Standard Baselines

For SAE-based PLM interpretability, the field's standard baselines are:
1. **Raw ESM-2 neurons** (i.e., post-MLP activations of the layer) compared via the same F1
   metric. InterPLM's headline result is SAE 2,548 vs neurons 46 features per layer at F1 > 0.5.
2. **Random-weight ESM-2** (shuffled): isolates what's due to pretraining vs the SAE architecture.
   InterPLM's Figure 3 shows random-weight ESM-2 SAEs have nearly no concept alignment.
3. **PCA / ICA features** on the same embeddings: dense baselines.
4. **Linear probes** on raw embeddings for downstream tasks (ProteinGym fitness, GO terms,
   contacts).

## 5. Evaluation Metrics

| Metric | Use | Note |
|---|---|---|
| **F1 with domain-adjusted recall** | Feature ↔ Swiss-Prot concept alignment | InterPLM's signature metric. Precision per amino acid, recall *per concept domain*. |
| **L0 (avg non-zero latents per token)** | SAE sparsity | Target: <100 for a 10K-feature SAE |
| **% Loss Recovered** | SAE faithfulness | (CE_recon − CE_orig) / (CE_zero − CE_orig). InterPLM SAEs achieve 98–100% |
| **Spearman ρ on ProteinGym** | Downstream fitness | LowNSAE methodology |
| **Steering success rate** | Causal validation | Top-fitness variants in 83% of cases (LowNSAE) |
| **Structural vs sequential Cohen's d** | Feature type characterization | InterPLM, requires AlphaFold |

## 6. Datasets in the Literature

| Dataset | Used by | Role |
|---|---|---|
| **Swiss-Prot / UniProtKB-reviewed** | InterPLM, Villegas-Garcia, all SAE papers | Per-residue ground-truth concept labels (active sites, binding, disulfide, helix, turn, strand, transmembrane, etc.) |
| **UniRef50** | InterPLM, ESM-2 pretraining | SAE training corpus |
| **AlphaFold-DB** | InterPLM | 3D structures for spatial-proximity analysis |
| **ProteinGym** | LowNSAE, Tranception, ESM-1v eval | Downstream fitness benchmark |
| **Pfam** | CAV paper, Encyclopedia of Domains | Motif/domain ground truth |
| **SCOPe 2.08** | Villegas-Garcia 2025 | Smaller, less-redundant SAE training corpus |
| **Observed Antibody Space (OAS)** | Antibody-SAE | Domain-specific PLM corpus |

## 7. Gaps and Opportunities

1. **"Dark" SAE features remain uncharacterized.** Even in InterPLM, ~80% of SAE features have no
   strong alignment to any Swiss-Prot concept. They are not all noise — many have consistent
   spatial / sequential activation patterns. Systematic characterization of these features is the
   research-project opportunity. The current state of the art is **LLM-based hypothesis
   generation** (Claude/GPT), which is fast but unvalidated.
2. **Causal validation is weak.** Most papers stop at feature-concept correlation. Steering
   experiments exist but only on a handful of features. Connecting feature ablation to **measurable
   functional changes** (ProteinGym fitness, GO-term prediction, structural prediction) is rare.
3. **Cross-model universality is barely studied.** Do the same features appear in ESM-2 and
   ProtT5? In ESM-2 vs ESM-3? If yes, the feature is more likely to reflect biological reality
   than model-specific superposition.
4. **Circuit-level analysis is just beginning.** Per-layer SAEs treat each layer in isolation;
   real biology likely lives in cross-layer circuits (early-layer detection → mid-layer
   composition → late-layer prediction). Only the 2026 cross-layer-transcoder paper has begun
   this.
5. **Validation against experimental literature.** Once we have an "unannotated SAE feature
   candidate", how do we test if it represents real biology that just isn't in Swiss-Prot yet?
   Searching for the feature's top-activating sequences in recent (post-Swiss-Prot-snapshot)
   literature is one route — none of the published papers do this.

## 8. Recommendations for Our Experiment

Given the hypothesis ("PLMs contain features for previously unknown mechanisms") and the existing
toolchain, the recommended experimental design is:

### Datasets
- **Primary**: Swiss-Prot human-reviewed proteome (`datasets/swissprot/full_human_proteome.fasta`)
  — 20K sequences, 13 MB. Already downloaded.
- **Validation**: AlphaFold structures for the same proteins (download on demand).
- **Downstream task**: 1–2 ProteinGym DMS assays for variant-effect validation
  (`datasets/proteingym/`).

### Baselines
- **Raw ESM-2 neuron–concept F1** (the original benchmark from InterPLM Fig 3).
- **InterPLM's released SAE features at F1 > 0.5** (the "known" features).
- **Random-weight ESM-2 SAE** as null model.

### Methods
1. **Skip SAE training**: load `Elana/InterPLM-esm2-8m` layer-4 pretrained SAE (~25 MB) from
   HuggingFace.
2. **Reproduce InterPLM's feature-concept F1 evaluation** on Swiss-Prot human subset. Verify
   pretrained features still have the expected concept alignment (sanity check).
3. **Filter to dark features**: SAE features with F1 < 0.2 against every Swiss-Prot concept AND
   spatial-clustering Cohen's d > 0.3 (i.e., they cluster in 3D, so they're not noise).
4. **Cluster dark features by activation pattern** (UMAP on activation profiles, HDBSCAN).
   Clusters of dark features are candidate "novel mechanism families".
5. **LLM-assisted annotation**: feed top-activating sequence windows + AlphaFold context to
   Claude, ask for a hypothesis ("what biological role does this region play?"). Use Claude-Opus
   for highest-quality biological reasoning.
6. **Causal validation**: pick the top-3 candidate features. For each:
   - Pick 3 wild-type sequences where the feature activates.
   - Mutate each activating residue to alanine (in silico). Recompute SAE feature.
   - Pass through ESMFold; check structural change.
   - For sequences in ProteinGym: check whether feature-activating residues have above-average
     |DMS_score|. If yes, the feature is *functionally important* even if no Swiss-Prot
     annotation flags it.
7. **Cross-model universality check**: take the same proteins, extract features from
   `Elana/InterPLM-esm2-650m` (larger model, layers 9/18/24). Are the same dark features
   represented? If yes, the "novel mechanism" claim is much stronger.

### Metrics
- **Primary**: Number of dark features that pass causal validation (per-residue fitness
  correlation, structural change on ablation, cross-model presence).
- **Secondary**: Quality of LLM-generated hypotheses (human-rated, or compared to recent literature
  retrieved by search).
- **Tertiary**: Reproduction of known InterPLM features (sanity).

### Methodological considerations
- The LLM annotation step risks **hallucination** — a hypothesis is not evidence. Treat LLM
  output as a search query for literature validation, not as proof.
- The Bricken "interpretability without actionability" paper is a warning: high feature-concept
  correlation in a probe ≠ causal mechanistic role. Steering / ablation is essential.
- ESM-2-8M is the right starting model: it's small enough to iterate quickly, and pretrained SAEs
  exist. Larger models (650M, 3B) can be a follow-up.
- The biggest *non-obvious* gotcha (from the antibody paper): TopK SAEs give clean per-feature
  interpretability but **Ordered SAEs give more reliable causal steerability**. If we want to
  emphasize causal claims, switch architecture.
