# Downloaded Papers

20 papers, ~154 MB total. All retrieved via arXiv except where noted. PDF chunks for the most
important papers are in `papers/pages/` for chunk-by-chunk reading.

## SAE-based PLM interpretability (CORE)

1. **InterPLM_Discovering_Interpretable_Features_in_PLMs_via_SAE.pdf** —
   Simon & Zou, Stanford, 2024 · arXiv:2412.12101 · 92 citations · published Nature Methods 2025.
   *The seminal paper.* SAE on ESM-2-8M finds 2,548 monosemantic features per layer mapping to 143
   Swiss-Prot concepts (vs 46 neurons covering 15 concepts). Provides pretrained SAEs for all
   layers on HuggingFace. Why: the methodological foundation we will reuse directly.

2. **Interpreting_Steering_PLMs_through_Sparse_Autoencoders.pdf** —
   Villegas Garcia & Ansuini, Area Science Park, 2025 · arXiv:2502.09135 · 16 citations.
   Adds intrinsic-dimension heuristic for layer selection; demonstrates steering toward zinc
   finger domains. Why: principled layer-selection method.

3. **ProtSAE_Disentangling_Interpreting_PLMs_Semantically.pdf** —
   2025 · arXiv:2509.05309 · 2 citations.
   Semantically-guided SAE objective. Why: complementary methodology and cautionary example
   (semantic regularization may suppress novelty discovery).

4. **Sparse_Autoencoders_LowN_Protein_Function_Prediction.pdf** —
   Tsui, Talreja, Aghazadeh, Georgia Tech, 2025 · arXiv:2508.18567.
   SAE latents beat raw ESM-2 embeddings at low-N fitness prediction. Why: provides the downstream
   validation protocol for testing if SAE features capture function.

5. **Hierarchical_Semantics_in_SAE_Architectures.pdf** —
   2025 · arXiv:2506.01197.
   Matryoshka/nested SAE architectures. Why: alternative architecture worth comparing.

6. **Mechanistic_Interpretability_Antibody_LMs_Using_SAEs.pdf** —
   Haque et al., Oxford / Reticular / Harvard, 2025/26 · arXiv:2512.05794.
   Compares TopK vs Ordered SAEs on antibody LMs. Why: shows feature-concept correlation ≠
   steerability; Ordered SAEs more reliable for causal experiments.

## Circuit and mechanistic analysis

7. **Protein_Circuit_Tracing_Cross_Layer_Transcoders.pdf** —
   2026 · arXiv:2602.12026 · 0 citations (very recent).
   Cross-layer transcoders for tracing computation through ESM-2. Why: state-of-the-art beyond
   per-layer SAEs.

8. **Induction_Meets_Biology_Repeat_Detection_PLMs.pdf** —
   2026 · arXiv:2602.23179.
   Identifies induction-head-like circuits for repeat detection. Why: concrete example of "novel
   mechanism" discovery via mech interp.

9. **Automated_Neuron_Labelling_Generative_Steering_PLMs.pdf** —
   Parsan et al., 2025 · arXiv:2507.06458.
   LLM-based automated labelling pipeline for PLM features. Why: template for LLM-assisted
   annotation step.

## Alternative interpretability methods

10. **Automated_Protein_Motif_Localization_CAVs_PLM.pdf** —
    Shamail & McWhite, U Arizona, 2025 · arXiv:2511.21614.
    Concept Activation Vectors instead of SAEs. Why: independent supervised lens — SAE features
    not matched by CAV are stronger novelty candidates.

11. **BERTology_Meets_Biology_Interpreting_Attention_PLMs.pdf** —
    Vig et al., 2020 · arXiv:2006.15222 · 359 citations.
    Seminal attention-based interpretability for PLMs. Why: historical context; baseline.

12. **Insights_Inner_Workings_Transformer_Protein_Function.pdf** —
    Stark et al., 2023 · arXiv:2309.03631 · 16 citations.
    Integrated gradients on protein function prediction. Why: complementary causal-validation tool.

## Foundational PLM models

13. **ProtTrans_Cracking_Language_Life_Self_Supervised.pdf** —
    Elnaggar et al., 2020 · arXiv:2007.06225 · 1,276 citations.
    ProtBERT, ProtT5, ProtAlbert. Why: alternative PLM family for universality studies.

14. **Tranception_Protein_Fitness_Prediction.pdf** —
    Notin et al., 2022 · arXiv:2205.13760.
    Autoregressive PLM for fitness prediction; motivates ProteinGym. Why: foundational benchmark.

15. **Fine_Tuning_ESM2_Missense_Variants_Functional_Impact.pdf** —
    2024 · arXiv:2410.10919.
    Recipe for fine-tuning ESM-2 on variant-effect labels. Why: base model recipe used by LowNSAE.

## Structure and design

16. **Towards_Interpretable_Protein_Structure_Prediction_with_SAEs.pdf** —
    2025 · arXiv:2503.08764 · 11 citations.
    SAEs on ESMFold. Why: structure-prediction-specific extension of the framework.

17. **Protein_Generation_Embedding_Learning_Motif_Diversification.pdf** —
    2025 · arXiv:2510.18790.
    Generative use of PLM embeddings for motif-aware design. Why: closing the discovery → design
    loop.

18. **Central_Dogma_Transformer_III_DNA_RNA_Protein.pdf** —
    2026 · arXiv:2603.23361.
    Multi-modal DNA/RNA/protein interpretable model. Why: cross-modality context.

## Caveats / supporting

19. **Interpretability_without_Actionability_Mechanistic.pdf** —
    2026 · arXiv:2603.18353.
    Argues mech-interp findings often don't translate to actionable model edits. Why: critical
    counterweight; informs validation strategy.

20. **Continuous_Sparse_Activations_PLMs.pdf** —
    2025 · arXiv:2502.07154.
    Alternative continuous sparsification approach. Why: supplementary methodological reference.

## Reading order recommendation for the experiment runner

1. **InterPLM** (#1) — full read, use the pretrained SAEs.
2. **Sparse_Autoencoders_LowN** (#4) — methods + experimental setup.
3. **Automated_Protein_Motif_Localization_CAVs** (#10) — alternative lens.
4. **Mechanistic_Interpretability_Antibody_LMs** (#6) — sections 2–3, caveats and ordered SAE.
5. **Interpretability_without_Actionability** (#19) — what to avoid claiming.

The other 15 papers are reference material; check them as specific questions arise.

## Chunked PDFs

The `papers/pages/` subdirectory contains 3-page-per-chunk splits of the papers used during deep
reading. These are reproducible via:
```bash
python .claude/skills/paper-finder/scripts/pdf_chunker.py papers/<filename>.pdf --pages-per-chunk 3
```
