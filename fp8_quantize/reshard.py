"""Reshard a model's .safetensors files into shards of a target size.

Tensors are copied byte-for-byte (no dtype conversion, no model loading), one
tensor at a time, so memory use stays tiny. Every tensor in every
*.safetensors file in the source folder ends up in the new shards and the new
index; all other files (config.json, tokenizer, processor configs, chat
template, ...) are copied as-is. The source folder is only read, never
modified.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .utils import INDEX_NAME, copy_support_files, fail, read_header

COPY_CHUNK = 64 * 1024 * 1024  # 64 MB


@dataclass
class ReshardStats:
    tensors: int
    shards: int
    total_bytes: int


def shards_fit_limit(src: Path, max_bytes: int) -> bool:
    """True if every .safetensors shard in src is already at most max_bytes."""
    for f in src.glob("*.safetensors"):
        if f.stat().st_size > max_bytes:
            return False
    return True


def reshard_folder(src: Path, dst: Path, max_shard_gb: float) -> ReshardStats:
    """Rewrite all tensors from ``src`` into shards of at most ``max_shard_gb`` in ``dst``.

    A single tensor bigger than the limit gets its own shard. ``dst`` must not
    contain .safetensors files of its own (support files are fine and are
    kept).
    """
    src, dst = src.resolve(), dst.resolve()
    if src == dst:
        fail("source and destination must be different folders")
    files = sorted(src.glob("*.safetensors"))
    if not files:
        fail(f"no .safetensors files found in {src}")
    dst.mkdir(parents=True, exist_ok=True)
    if list(dst.glob("*.safetensors")):
        fail(f"{dst} already contains .safetensors files")

    # Collect every tensor: (name, source file, data start, header info)
    tensors, seen = [], set()
    for f in files:
        try:
            header, data_start = read_header(f)
        except (json.JSONDecodeError, struct.error, OSError) as exc:
            fail(f"cannot read {f.name}: {exc}")
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if name in seen:
                fail(f"tensor {name} appears in more than one file")
            seen.add(name)
            tensors.append((name, f, data_start, info))

    # Group into shards of at most max_bytes (a single bigger tensor gets its own shard)
    max_bytes = int(max_shard_gb * 1e9)
    shards, current, size = [], [], 0
    for t in tensors:
        nbytes = t[3]["data_offsets"][1] - t[3]["data_offsets"][0]
        if current and size + nbytes > max_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(t)
        size += nbytes
    if current:
        shards.append(current)

    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        header, offset = {"__metadata__": {"format": "pt"}}, 0
        for name, _, _, info in shard:
            nbytes = info["data_offsets"][1] - info["data_offsets"][0]
            header[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
            weight_map[name] = fname
        total += offset

        hbytes = json.dumps(header, separators=(",", ":")).encode()
        hbytes += b" " * (-len(hbytes) % 8)  # pad header to 8-byte alignment
        with open(dst / fname, "wb") as out:
            out.write(struct.pack("<Q", len(hbytes)))
            out.write(hbytes)
            for _, srcfile, data_start, info in shard:
                start, end = info["data_offsets"]
                with open(srcfile, "rb") as fin:
                    fin.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        chunk = fin.read(min(COPY_CHUNK, remaining))
                        if not chunk:
                            fail(f"{srcfile.name} is truncated")
                        out.write(chunk)
                        remaining -= len(chunk)
        print(f"Wrote {fname} ({offset / 1e9:.2f} GB, {len(shard)} tensors)")

    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (dst / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    copy_support_files(src, dst)

    print(f"Resharded {len(tensors)} tensors into {len(shards)} shards ({total / 1e9:.1f} GB)")
    return ReshardStats(len(tensors), len(shards), total)
