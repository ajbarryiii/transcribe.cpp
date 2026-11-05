"""Reader for keab-8bit-v1 export bundles (kea-b binary Parakeet TDT).

A bundle directory holds bundle.safetensors, manifest.json and tokenizer/.
The manifest pins the SHA-256 of every file. Three weight encodings:

  binary  <layer>.codes uint8 [out, ceil(in/8)] + <layer>.scale fp16 [out].
          Code j of a row is bit j % 8 of byte j // 8 (LSB first); bit 1 is
          +1, bit 0 is -1. W[o, j] = scale[o] * code[o, j]. No biases.
  int8    <name>.int8 int8 [out, in] + <name>.int8_scale fp32 [out].
          The forward weight is W8 = fp16(q * s), computed in fp32.
  float   every other floating tensor, fp16 (fp32 feature constants).

Every check fails loudly: an unexpected format, hash, tensor, dtype or
shape raises ValueError instead of being skipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "keab-8bit-v1"
MODEL_CLASS = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
BIT_LAYOUT = (
    "uint8 [out, ceil(in/8)]; code j of a row is bit j % 8 of byte j // 8 "
    "(least significant bit first); bit 1 = +1, bit 0 = -1; unused tail bits 0"
)
INT8_FORMAT = (
    "symmetric per output channel (row): s = max(max|W_c| / 127, 1e-30), "
    "q = round-half-even(W / s) clamped to [-127, 127]; forward weight W8 = fp16(q * s)"
)
BINARY_KINDS = ("linear", "pointwise_conv1d")

# ggml block layouts written directly by the converter. gguf-py 0.18 does
# not know Q1_0, so the type id and block geometry are pinned here against
# ggml/src/ggml-common.h (block_q1_0, block_q8_0).
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q1_0 = 41
QK8_0 = 32
QK1_0 = 128


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BinaryLayer:
    kind: str           # "linear" or "pointwise_conv1d"
    codes: np.ndarray   # uint8 [out, in / 8]
    scale: np.ndarray   # float16 [out]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.codes.shape[0]), int(self.codes.shape[1]) * 8

    def signs(self) -> np.ndarray:
        """{-1, +1} float32 [out, in]."""
        bits = np.unpackbits(self.codes, axis=1, bitorder="little")
        return bits.astype(np.float32) * 2.0 - 1.0

    def dequantize(self) -> np.ndarray:
        """float32 [out, in]: scale[o] * code[o, j], exact in fp32."""
        return self.signs() * self.scale.astype(np.float32)[:, None]


@dataclass
class Int8Tensor:
    q: np.ndarray       # int8 [out, in]
    s: np.ndarray       # float32 [out]

    def dequantize(self) -> np.ndarray:
        """float32 [out, in] of W8 = fp16(q * s)."""
        w = self.q.astype(np.float32) * self.s[:, None]
        return w.astype(np.float16).astype(np.float32)


@dataclass
class KeabBundle:
    root: Path
    manifest: dict
    binary: dict[str, BinaryLayer]      # NeMo module name -> layer
    int8: dict[str, Int8Tensor]         # NeMo weight name -> tensor
    floats: dict[str, np.ndarray]       # NeMo name -> fp16/fp32 array
    integers: dict[str, np.ndarray]     # BN num_batches_tracked buffers

    @property
    def config(self) -> dict:
        return self.manifest["config"]

    @property
    def tokenizer_model(self) -> Path:
        return self.root / "tokenizer" / "tokenizer.model"

    def dequantized_state_dict(self) -> dict[str, np.ndarray]:
        """NeMo state_dict of float32 arrays with every weight dequantized.
        Binary pointwise convs keep NeMo's Conv1d layout [out, in, 1]."""
        sd: dict[str, np.ndarray] = {}
        for name, layer in self.binary.items():
            w = layer.dequantize()
            sd[f"{name}.weight"] = w[:, :, None] if layer.kind == "pointwise_conv1d" else w
        for name, t in self.int8.items():
            sd[name] = t.dequantize()
        for name, a in self.floats.items():
            sd[name] = a.astype(np.float32)
        sd.update(self.integers)
        return sd


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"keab bundle: {msg}")


def read_bundle(root: Path) -> KeabBundle:
    from safetensors import safe_open

    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    _require(manifest.get("format") == FORMAT, f"format {manifest.get('format')!r}, expected {FORMAT!r}")
    _require(manifest.get("model_class") == MODEL_CLASS, f"model_class {manifest.get('model_class')!r}")
    _require(manifest.get("bit_layout") == BIT_LAYOUT, f"unexpected bit_layout {manifest.get('bit_layout')!r}")
    _require(manifest.get("binary_scale_dtype") == "float16",
             f"binary_scale_dtype {manifest.get('binary_scale_dtype')!r}, expected 'float16'")
    _require(manifest.get("int8", {}).get("format") == INT8_FORMAT,
             f"unexpected int8 format {manifest.get('int8', {}).get('format')!r}")
    files = manifest.get("files") or {}
    weights = manifest.get("file")
    _require(weights in files and files[weights] == manifest.get("sha256"),
             f"manifest file/sha256 do not match files[{weights!r}]")
    _require("tokenizer/tokenizer.model" in files, "files lacks tokenizer/tokenizer.model")
    for rel, digest in sorted(files.items()):
        actual = sha256_file(root / rel)
        _require(actual == digest, f"{rel}: sha256 {actual} != manifest {digest}")

    with safe_open(str(root / weights), framework="np") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    described = manifest["tensors"]
    _require(set(described) == set(raw),
             f"tensors differ from the manifest: only in file {sorted(set(raw) - set(described))[:5]}, "
             f"only in manifest {sorted(set(described) - set(raw))[:5]}")
    for name, info in described.items():
        a = raw[name]
        _require(str(a.dtype) == info["dtype"] and list(a.shape) == list(info["shape"]),
                 f"{name}: {a.dtype} {list(a.shape)} != manifest {info['dtype']} {info['shape']}")

    claimed: set[str] = set()

    def take(name: str, dtype, shape: tuple[int, ...]) -> np.ndarray:
        _require(name in raw, f"missing tensor {name}")
        _require(name not in claimed, f"tensor {name} claimed twice")
        a = raw[name]
        _require(a.dtype == dtype and tuple(a.shape) == shape,
                 f"{name}: {a.dtype} {tuple(a.shape)}, expected {np.dtype(dtype)} {shape}")
        claimed.add(name)
        return a

    binary: dict[str, BinaryLayer] = {}
    for name, info in manifest["binary_layers"].items():
        _require(info.get("kind") in BINARY_KINDS, f"{name}: unknown binary kind {info.get('kind')!r}")
        _require(info.get("bias") is False, f"{name}: binary layers with biases are not supported")
        out_f, in_f = (int(x) for x in info["shape"])
        _require(in_f % 8 == 0, f"{name}: in_features {in_f} is not a multiple of 8")
        codes = take(f"{name}.codes", np.uint8, (out_f, in_f // 8))
        scale = take(f"{name}.scale", np.float16, (out_f,))
        _require(bool(np.isfinite(scale).all()), f"{name}: non-finite scale")
        binary[name] = BinaryLayer(info["kind"], codes, scale)

    int8: dict[str, Int8Tensor] = {}
    for name, info in manifest["int8"]["tensors"].items():
        _require(info.get("channel_axis") == 0, f"{name}: channel_axis {info.get('channel_axis')!r}, expected 0")
        shape = tuple(int(x) for x in info["shape"])
        _require(len(shape) == 2, f"{name}: int8 tensors must be 2-D, got {shape}")
        q = take(info["codes"], np.int8, shape)
        s = take(info["scales"], np.float32, (shape[0],))
        _require(int(np.abs(q.astype(np.int16)).max()) <= 127, f"{name}: code -128 outside [-127, 127]")
        _require(bool(np.isfinite(s).all()) and bool((s > 0).all()), f"{name}: non-positive or non-finite scale")
        int8[name] = Int8Tensor(q, s)

    floats: dict[str, np.ndarray] = {}
    integers: dict[str, np.ndarray] = {}
    fp32_buffers = set(manifest.get("fp32_buffers") or [])
    for name in sorted(set(raw) - claimed):
        a = raw[name]
        if name.endswith(".num_batches_tracked"):
            _require(a.dtype == np.int64 and a.shape == (), f"{name}: {a.dtype} {a.shape}")
            integers[name] = a
        elif name in fp32_buffers:
            _require(a.dtype == np.float32, f"{name}: fp32 buffer stored as {a.dtype}")
            floats[name] = a
        else:
            _require(a.dtype == np.float16, f"{name}: unexpected dtype {a.dtype}")
            floats[name] = a
        _require(not name.endswith((".codes", ".scale", ".int8", ".int8_scale")),
                 f"{name}: quantized tensor not described by the manifest")

    return KeabBundle(root, manifest, binary, int8, floats, integers)


def q1_0_bytes(layer: BinaryLayer) -> np.ndarray:
    """ggml Q1_0 rows for a binary layer: uint8 [out, in / 128 * 18].

    block_q1_0 is {fp16 d; uint8 qs[16]} with weight j = qs[j/8] bit j%8,
    1 -> +d, 0 -> -d: the bundle's bit order and sign. Each row's 16-byte
    chunks become blocks whose d is that row's scale, so the encoding is
    lossless."""
    out_f, in_f = layer.shape
    if in_f % QK1_0:
        raise ValueError(f"Q1_0 needs rows of a multiple of {QK1_0} weights, got {in_f}")
    nb = in_f // QK1_0
    d = np.broadcast_to(layer.scale.astype("<f2").view(np.uint8).reshape(out_f, 1, 2), (out_f, nb, 2))
    qs = layer.codes.reshape(out_f, nb, QK1_0 // 8)
    return np.ascontiguousarray(np.concatenate([d, qs], axis=2).reshape(out_f, nb * (2 + QK1_0 // 8)))


def q8_0_bytes(t: Int8Tensor) -> np.ndarray:
    """ggml Q8_0 rows for an int8 tensor: uint8 [out, in / 32 * 34], the
    bundle's codes with d = fp16(s) in every block of the row. Dequantizes
    to q * fp16(s), which differs from W8 = fp16(q * s) by fp16 rounding."""
    out_f, in_f = t.q.shape
    if in_f % QK8_0:
        raise ValueError(f"Q8_0 needs rows of a multiple of {QK8_0} weights, got {in_f}")
    nb = in_f // QK8_0
    d = np.broadcast_to(t.s.astype("<f2").view(np.uint8).reshape(out_f, 1, 2), (out_f, nb, 2))
    qs = t.q.view(np.uint8).reshape(out_f, nb, QK8_0)
    return np.ascontiguousarray(np.concatenate([d, qs], axis=2).reshape(out_f, nb * (2 + QK8_0)))
