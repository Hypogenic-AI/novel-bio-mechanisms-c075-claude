"""Data loading helpers for Swiss-Prot and ProteinGym."""
from __future__ import annotations

import re
import random
from pathlib import Path
from typing import List, Tuple, Iterator

from src import config


def parse_fasta(path: Path) -> Iterator[Tuple[str, str, str]]:
    """Yield (accession, header, sequence) triples from a FASTA file.

    Accession is extracted from `sp|ACC|NAME` headers; falls back to the
    full header if the pattern doesn't match.
    """
    acc_re = re.compile(r"sp\|([^|]+)\|")
    header = None
    seq_parts: List[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_parts)
                    m = acc_re.match(header)
                    acc = m.group(1) if m else header.split()[0]
                    yield acc, header, seq
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            seq = "".join(seq_parts)
            m = acc_re.match(header)
            acc = m.group(1) if m else header.split()[0]
            yield acc, header, seq


def load_swissprot_subset(
    n: int = config.N_PROTEINS,
    min_len: int = config.SEQ_LEN_MIN,
    max_len: int = config.SEQ_LEN_MAX,
    seed: int = config.SEED,
) -> List[Tuple[str, str]]:
    """Load a random subset of Swiss-Prot human proteins filtered by length."""
    rng = random.Random(seed)
    all_records: List[Tuple[str, str]] = []
    for acc, _header, seq in parse_fasta(config.SWISSPROT_FASTA):
        # Filter standard amino acids only
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
            continue
        if min_len <= len(seq) <= max_len:
            all_records.append((acc, seq))
    rng.shuffle(all_records)
    return all_records[:n]
