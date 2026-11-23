"""Helpers to build tiny fake model folders for tests.

Pure standard library: fake .safetensors files are written by hand (header +
raw bytes), so no torch / safetensors / network is needed.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path


def make_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    """Write a minimal valid .safetensors file from raw tensor bytes."""
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, data in tensors.items():
        n = len(data)
        assert n % 4 == 0, "test tensors must have a 4-byte aligned size"
        header[name] = {"dtype": "F32", "shape": [n // 4], "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)  # 8-byte aligned header, as the real writer does
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for data in tensors.values():
            f.write(data)


def make_model_folder(
    folder: Path,
    files: dict[str, dict[str, bytes]],
    extra: dict[str, str] | None = None,
) -> Path:
    """Build a fake model folder: .safetensors shards + config.json + extras."""
    folder.mkdir(parents=True, exist_ok=True)
    for fname, tensors in files.items():
        make_safetensors(folder / fname, tensors)
    (folder / "config.json").write_text(json.dumps({"model_type": "fake-model"}))
    for name, content in (extra or {}).items():
        (folder / name).write_text(content)
    return folder


def raw_bytes(data_size: int, seed: int = 0) -> bytes:
    """A deterministic byte pattern of the given size."""
    return bytes((seed + i) % 256 for i in range(data_size))
