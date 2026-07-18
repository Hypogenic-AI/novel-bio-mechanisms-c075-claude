"""Modal entrypoints for the expanded-label experiment.

Usage (from repo root, with Modal authenticated):

  # 1) Upload local results needed on the volume (activations, labels, index, ...)
  modal run modal_app.py::sync_inputs

  # 2) Run GPU pipeline: F1 → dark → ablation → figures
  modal run modal_app.py::run_pipeline

  # Optional: also rebuild labels / re-extract activations on GPU
  modal run modal_app.py::run_pipeline --stages build_concepts build_motifs extract_activations compute_f1 dark_features ablation make_figures

No OpenAI/Anthropic key is required (LLM stage is omitted).
"""
from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "novel-bio-mechanisms"
VOLUME_NAME = "novel-bio-mechanisms-data"
DATA_ROOT = "/data"
PROJECT_ROOT = "/root/project"

LOCAL_ROOT = Path(__file__).resolve().parent

# Artifacts to sync for the default GPU path (activations already extracted locally).
SYNC_FILES = [
    "results/index.json",
    "results/uniprot_raw.json",
    "results/reference_features.npz",
    "results/reference_features_meta.json",
    "results/concepts_uniprot.json",
    "results/motif_annotations.npz",
    "results/motif_annotations_meta.json",
    "results/concepts_motifs.json",
    "results/sae_activations.npz",
    "results/hidden_states.npz",
]

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy",
        "scipy",
        "tqdm",
        "requests",
        "huggingface-hub",
        "safetensors",
        "einops",
        "h5py",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "biopython",
        "transformers",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(
        # InterPLM install may pull extras; we only need interplm.sae.dictionary.ReLUSAE.
        "pip install 'git+https://github.com/ElanaPearl/interPLM.git' || true"
    )
    .env(
        {
            "NOVEL_BIO_ROOT": DATA_ROOT,
            "HF_HOME": f"{DATA_ROOT}/hf_cache",
            "TRANSFORMERS_CACHE": f"{DATA_ROOT}/hf_cache",
            "PYTHONPATH": PROJECT_ROOT,
        }
    )
    .add_local_dir(
        str(LOCAL_ROOT),
        remote_path=PROJECT_ROOT,
        ignore=[
            ".venv",
            ".git",
            ".claude",
            ".codex",
            ".gemini",
            ".neurico",
            "papers",
            "paper_draft",
            "paper_search_results",
            "logs",
            "figures",
            "**/__pycache__",
            "results/*.npz",
            "results/uniprot_raw.json",
            "results/reference_features_meta.json",
            "uv.lock",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={DATA_ROOT: vol},
    timeout=60 * 60,
)
def _noop_volume_touch() -> str:
    """Ensure volume is created / mountable."""
    Path(DATA_ROOT, "results").mkdir(parents=True, exist_ok=True)
    Path(DATA_ROOT, "figures").mkdir(parents=True, exist_ok=True)
    Path(DATA_ROOT, "logs").mkdir(parents=True, exist_ok=True)
    Path(DATA_ROOT, "hf_cache").mkdir(parents=True, exist_ok=True)
    vol.commit()
    return f"volume ready at {DATA_ROOT}"


@app.local_entrypoint()
def sync_inputs() -> None:
    """Upload local results artifacts onto the Modal volume."""
    print("Ensuring volume directories exist...")
    print(_noop_volume_touch.remote())

    missing = [rel for rel in SYNC_FILES if not (LOCAL_ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing local files required for sync:\n  " + "\n  ".join(missing)
        )

    print(f"Uploading {len(SYNC_FILES)} files to volume '{VOLUME_NAME}'...")
    with vol.batch_upload(force=True) as batch:
        for rel in SYNC_FILES:
            local_path = LOCAL_ROOT / rel
            remote_path = rel  # stored under /data/<rel> when volume is mounted at /data
            print(f"  put {local_path} -> /data/{remote_path}")
            batch.put_file(str(local_path), remote_path)
    print("Sync complete. Volume committed on upload finish.")


@app.function(
    volumes={DATA_ROOT: vol},
    gpu="A10G",
    timeout=60 * 60 * 6,
    memory=32768,
)
def run_pipeline_remote(stages: list[str] | None = None) -> dict:
    """Run experiment stages on GPU with data under /data."""
    import os
    import sys

    os.environ["NOVEL_BIO_ROOT"] = DATA_ROOT
    os.environ["HF_HOME"] = f"{DATA_ROOT}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{DATA_ROOT}/hf_cache"
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src import config
    from src.run_pipeline import DEFAULT_STAGES, run_pipeline

    print(f"DEVICE={config.DEVICE} ROOT={config.ROOT}")
    print(f"RESULTS={config.RESULTS} exists={config.RESULTS.exists()}")
    for required in (
        "index.json",
        "sae_activations.npz",
        "hidden_states.npz",
        "reference_features.npz",
        "motif_annotations.npz",
    ):
        path = config.RESULTS / required
        print(f"  {path}: exists={path.exists()} size={path.stat().st_size if path.exists() else 0}")

    chosen = list(stages) if stages else list(DEFAULT_STAGES)
    run_pipeline(chosen)
    vol.commit()

    summary = {}
    f1_summary = config.RESULTS / "f1_summary.json"
    dark = config.RESULTS / "dark_features.json"
    if f1_summary.exists():
        import json

        summary["f1_summary"] = json.loads(f1_summary.read_text())
    if dark.exists():
        import json

        dark_meta = json.loads(dark.read_text())
        summary["dark"] = {
            k: dark_meta[k]
            for k in (
                "n_bright_uniprot",
                "n_granularity_gap",
                "n_dark_candidate",
                "n_structured_dark",
                "top_dark_indices",
            )
            if k in dark_meta
        }
    return summary


@app.local_entrypoint()
def run_pipeline() -> None:
    """Local CLI wrapper: modal run modal_app.py::run_pipeline"""
    print("Launching GPU pipeline on Modal...")
    summary = run_pipeline_remote.remote()
    print("Pipeline finished.")
    if summary:
        import json

        print(json.dumps(summary, indent=2)[:4000])


@app.local_entrypoint()
def download_results() -> None:
    """Pull key result artifacts from the volume back to local results/."""
    remote_files = [
        "results/f1_sae.npz",
        "results/f1_neurons.npz",
        "results/f1_summary.json",
        "results/dark_features.json",
        "results/dark_features.npz",
        "results/ablation_results.json",
        "figures/fig1_f1_comparison.png",
        "figures/fig2_dark_entropy.png",
        "figures/fig3_ablation.png",
        "figures/fig5_concept_coverage.png",
    ]
    print("Downloading results from volume...")
    for rel in remote_files:
        local_path = LOCAL_ROOT / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Modal volume read API: open files via a tiny remote helper
            data = _read_volume_file.remote(rel)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {rel}: {exc}")
            continue
        if data is None:
            print(f"  missing {rel}")
            continue
        local_path.write_bytes(data)
        print(f"  wrote {local_path} ({len(data):,} bytes)")


@app.function(volumes={DATA_ROOT: vol}, timeout=60 * 30)
def _read_volume_file(rel: str) -> bytes | None:
    path = Path(DATA_ROOT) / rel
    if not path.exists():
        return None
    return path.read_bytes()
