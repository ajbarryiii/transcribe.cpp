#!/usr/bin/env python3
"""Weight parity of a kea-b GGUF against its parakeet-binary-v1 export.

Dequantizes every tensor of a GGUF written by
`convert-parakeet.py --keab-weights {q1_0,f32}` and compares it with the
export's own weights (lib/binary_view.py, independent of the converter's
bundle reader). Q1_0, F16 and F32 tensors must match exactly. Q8_0 tensors
hold the int8 codes with d = fp16(s), so they may differ from the export's
fp16(q * s) by fp16 rounding; their worst error is reported and bounded.

    uv run --project scripts/envs/parakeet scripts/check_keab_gguf.py \\
        GGUF EXPORT_VIEW_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.binary_view import read_binary_view  # noqa: E402

GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q1_0 = 0, 1, 8, 41
TYPE_NAMES = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q8_0: "Q8_0", GGML_Q1_0: "Q1_0"}
# Q8_0: q * fp16(s) vs fp16(q * s) is two fp16 roundings apart at most.
Q8_0_MAX_REL = 2.0 ** -10


def read_gguf_tensors(path: Path) -> dict[str, tuple[int, tuple[int, ...], np.ndarray]]:
    """name -> (ggml type, ne, raw bytes). A minimal reader: gguf-py 0.18
    rejects ggml type 41 (Q1_0)."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    pos = 0

    def take(fmt: str):
        nonlocal pos
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, data, pos)
        pos += size
        return value[0] if len(value) == 1 else value

    def string() -> str:
        nonlocal pos
        n = take("<Q")
        s = bytes(data[pos:pos + n]).decode()
        pos += n
        return s

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}

    def value(vtype: int):
        if vtype == 8:
            return string()
        if vtype == 9:
            itype, n = take("<I"), take("<Q")
            return [value(itype) for _ in range(n)]
        return take(scalar[vtype])

    if bytes(data[:4]) != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    pos = 4
    version, n_tensors, n_kv = take("<I"), take("<Q"), take("<Q")
    if version != 3:
        raise ValueError(f"{path}: GGUF version {version}, expected 3")
    alignment = 32
    for _ in range(n_kv):
        key = string()
        v = value(take("<I"))
        if key == "general.alignment":
            alignment = int(v)
    infos = []
    for _ in range(n_tensors):
        name = string()
        ne = tuple(take("<Q") for _ in range(take("<I")))
        infos.append((name, ne, take("<I"), take("<Q")))
    base = (pos + alignment - 1) // alignment * alignment
    block = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_Q8_0: (32, 34), GGML_Q1_0: (128, 18)}
    out = {}
    for name, ne, ggml_type, offset in infos:
        if ggml_type not in block:
            raise ValueError(f"{name}: unexpected ggml type {ggml_type}")
        per, size = block[ggml_type]
        n = int(np.prod(ne))
        if ne[0] % per:
            raise ValueError(f"{name}: row of {ne[0]} is not a multiple of {per}")
        start = base + offset
        out[name] = (ggml_type, ne, data[start:start + n // per * size])
    return out


def dequantize(ggml_type: int, ne: tuple[int, ...], raw: np.ndarray) -> np.ndarray:
    """FP32 array in numpy order (ne reversed)."""
    shape = tuple(reversed(ne))
    if ggml_type == GGML_F32:
        return raw.view("<f4").reshape(shape)
    if ggml_type == GGML_F16:
        return raw.view("<f2").astype(np.float32).reshape(shape)
    if ggml_type == GGML_Q8_0:
        blocks = raw.reshape(-1, 34)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        return (blocks[:, 2:].view(np.int8).astype(np.float32) * d).reshape(shape)
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    bits = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little").astype(np.float32)
    return ((bits * 2.0 - 1.0) * d).reshape(shape)


def gguf_to_nemo_names() -> dict[str, list[str]]:
    """GGUF name -> NeMo source names (two for the summed LSTM bias), from
    the converter's own tables."""
    spec = importlib.util.spec_from_file_location(
        "convert_parakeet", Path(__file__).resolve().parent / "convert-parakeet.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    names = {g: [n] for n, g in conv.PRE_ENCODE_TABLE + conv.JOINT_TABLE}
    names["pred.embed.weight"] = ["decoder.prediction.embed.weight"]
    for i in range(64):
        for n, g in conv.ENCODER_BLOCK_TABLE:
            names[f"enc.blocks.{i}.{g}"] = [f"encoder.layers.{i}.{n}"]
        for stat in ("running_mean", "running_var"):
            names[f"enc.blocks.{i}.conv.bn.{stat}"] = [f"encoder.layers.{i}.conv.batch_norm.{stat}"]
        lstm = f"decoder.prediction.dec_rnn.lstm"
        names[f"pred.lstm.{i}.Wx"] = [f"{lstm}.weight_ih_l{i}"]
        names[f"pred.lstm.{i}.Wh"] = [f"{lstm}.weight_hh_l{i}"]
        names[f"pred.lstm.{i}.bias"] = [f"{lstm}.bias_ih_l{i}", f"{lstm}.bias_hh_l{i}"]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("view", type=Path, help="parakeet-binary-v1 export directory")
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args()

    tensors = read_gguf_tensors(args.gguf)
    _, state = read_binary_view(args.view)
    names = gguf_to_nemo_names()
    used: set[str] = set()
    stats: dict[str, dict] = defaultdict(lambda: {"tensors": 0, "params": 0, "bytes": 0,
                                                  "max_abs": 0.0, "max_rel": 0.0})
    failures = []
    for gname, (ggml_type, ne, raw) in tensors.items():
        if gname not in names:
            failures.append(f"{gname}: no NeMo source")
            continue
        src = names[gname]
        expect = state[src[0]] if len(src) == 1 else state[src[0]] + state[src[1]]
        used.update(src)
        got = dequantize(ggml_type, ne, raw)
        # Pointwise convs keep the conv layout [out, in, 1] as F32/F16 (the
        # loader requires it) and are 2-D [out, in] when block-quantized;
        # both hold the same row-major order.
        pointwise_2d = expect.ndim == 3 and expect.shape[2] == 1 and got.shape == expect.shape[:2]
        quantized = ggml_type not in (GGML_F32, GGML_F16)
        if got.shape != expect.shape and not (pointwise_2d and quantized):
            failures.append(f"{gname}: {got.shape} vs source {expect.shape}")
            continue
        if not np.isfinite(got).all():
            failures.append(f"{gname}: non-finite values")
            continue
        diff = np.abs(got.reshape(-1).astype(np.float64) - expect.reshape(-1))
        s = stats[TYPE_NAMES[ggml_type]]
        s["tensors"] += 1
        s["params"] += int(got.size)
        s["bytes"] += int(raw.size)
        s["max_abs"] = max(s["max_abs"], float(diff.max()))
        rel = float((diff / np.maximum(np.abs(expect.reshape(-1)), 1e-30)).max())
        s["max_rel"] = max(s["max_rel"], rel)
        if ggml_type == GGML_Q8_0:
            if rel > Q8_0_MAX_REL:
                failures.append(f"{gname}: Q8_0 relative error {rel:.3g} > {Q8_0_MAX_REL:.3g}")
        elif diff.max() != 0.0:
            failures.append(f"{gname}: {TYPE_NAMES[ggml_type]} differs by {diff.max():.3g}")
    ignored = {k for k in state if k.startswith("preprocessor.") or k.endswith(".num_batches_tracked")}
    missing = sorted(set(state) - used - ignored)
    if missing:
        failures.append(f"{len(missing)} export tensors not in the GGUF, e.g. {missing[:3]}")

    report = {"gguf": str(args.gguf), "gguf_bytes": args.gguf.stat().st_size,
              "tensors": len(tensors), "by_type": dict(stats), "failures": failures}
    for t, s in sorted(stats.items()):
        print(f"{t:5} {s['tensors']:4d} tensors {s['params']:11d} params {s['bytes'] / 1e6:8.2f} MB  "
              f"max_abs {s['max_abs']:.3g}  max_rel {s['max_rel']:.3g}")
    print(f"{len(tensors)} tensors, {report['gguf_bytes']} bytes")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    for f in failures:
        print(f"FAIL {f}")
    print("weight parity:", "FAIL" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
