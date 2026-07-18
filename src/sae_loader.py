"""Load InterPLM ReLU SAE checkpoints without the broken train-config import path."""
from __future__ import annotations

from huggingface_hub import hf_hub_download

from src import config


def load_interplm_sae(device: str | None = None, unnormalized: bool = False):
    """Load ESM-2-8M layer SAE from HuggingFace via ReLUSAE.from_pretrained."""
    from interplm.sae.dictionary import ReLUSAE

    device = device or config.DEVICE
    kind = "unnormalized" if unnormalized else "normalized"
    weights_path = hf_hub_download(
        repo_id="Elana/InterPLM-esm2-8m",
        filename=f"layer_{config.ESM_LAYER}/ae_{kind}.pt",
    )
    sae = ReLUSAE.from_pretrained(weights_path, device=device)
    sae.eval()
    for param in sae.parameters():
        param.requires_grad_(False)
    return sae.to(device)
