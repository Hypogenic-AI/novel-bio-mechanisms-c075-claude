"""Orchestrate the expanded-label pipeline (no LLM stage)."""
from __future__ import annotations

import argparse
import time
from typing import List


DEFAULT_STAGES = [
    "compute_f1",
    "dark_features",
    "ablation",
    "make_figures",
]

ALL_STAGES = [
    "build_concepts",
    "build_motifs",
    "extract_activations",
    "compute_f1",
    "dark_features",
    "ablation",
    "proteingym_correlation",
    "make_figures",
]


def run_stage(name: str) -> None:
    print(f"\n===== [{time.strftime('%H:%M:%S')}] STAGE: {name} =====")
    if name == "build_concepts":
        from src.build_concepts import main
    elif name == "build_motifs":
        from src.build_motifs import main
    elif name == "extract_activations":
        from src.extract_activations import main
    elif name == "compute_f1":
        from src.compute_f1 import main
    elif name == "dark_features":
        from src.dark_features import main
    elif name == "ablation":
        from src.ablation import main
    elif name == "proteingym_correlation":
        from src.proteingym_correlation import main
    elif name == "make_figures":
        from src.make_figures import main
    else:
        raise ValueError(f"Unknown stage: {name}")
    main()
    print(f"===== [{time.strftime('%H:%M:%S')}] DONE: {name} =====\n")


def run_pipeline(stages: List[str] | None = None) -> None:
    stages = stages or DEFAULT_STAGES
    print(f"Running stages: {stages}")
    for stage in stages:
        run_stage(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanded-label experiment stages")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=DEFAULT_STAGES,
        help="Pipeline stages to run (default: F1 → dark → ablation → figures)",
    )
    args = parser.parse_args()
    run_pipeline(args.stages)


if __name__ == "__main__":
    main()
