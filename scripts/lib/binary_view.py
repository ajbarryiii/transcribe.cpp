"""Reader for parakeet-binary-v1 exports: the scoring view of a binary-weight
Parakeet (kea-b-0.1), as its exporter loads it. Numpy only, so both the
NeMo reference dumper and the GGUF weight check can use it.

Binary layer `name` stores `name.codes` uint8 [out, ceil(in/8)] (code j is
bit j % 8 of byte j // 8, 1 = +1, 0 = -1) and `name.scale` FP32 [out]; its
weight is code * scale. Every other tensor is stored as the model uses it.
This reader is deliberately independent of lib/keab.py, the converter's
keab-8bit-v1 bundle reader, so it can serve as the oracle for it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "parakeet-binary-v1"
TOKENIZER_FILES = {"model_path": "tokenizer.model", "vocab_path": "vocab.txt",
                   "spe_tokenizer_vocab": "tokenizer.vocab"}


def is_binary_view(path: Path) -> bool:
    manifest = Path(path) / "manifest.json"
    return manifest.is_file() and json.loads(manifest.read_text()).get("format") == FORMAT


def read_binary_view(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """(manifest, state_dict) with every weight dequantized to FP32 and
    pointwise convs in NeMo's Conv1d layout [out, in, 1]. Integer buffers
    (BatchNorm num_batches_tracked) are kept as stored."""
    from safetensors.numpy import load_file

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{root}: format {manifest.get('format')!r}, expected {FORMAT!r}")
    for rel, digest in manifest["files"].items():
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{root / rel}: sha256 {actual} != manifest {digest}")
    tensors = load_file(str(root / manifest["file"]))
    state: dict[str, np.ndarray] = {}
    for name, layer in manifest["quantized_layers"].items():
        out_f, in_f = layer["shape"]
        if layer["bias"]:
            raise ValueError(f"{name}: binary layers with biases are not supported")
        codes = tensors.pop(f"{name}.codes")
        scale = tensors.pop(f"{name}.scale").astype(np.float32)
        bits = np.unpackbits(codes, axis=1, bitorder="little")[:, :in_f]
        w = (bits.astype(np.float32) * 2.0 - 1.0) * scale[:, None]
        if w.shape != (out_f, in_f):
            raise ValueError(f"{name}: weight {w.shape} != {(out_f, in_f)}")
        state[f"{name}.weight"] = w[:, :, None] if layer["kind"] == "pointwise_conv1d" else w
    for name, a in tensors.items():
        state[name] = a.astype(np.float32) if np.issubdtype(a.dtype, np.floating) else a
    return manifest, state
