"""Shared helpers: errors, raw safetensors header parsing, file copying.

Only the standard library is used at module import time so the package stays
importable (and testable) on machines without torch / CUDA installed.
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
RESTORED_NAME = "restored-bf16.safetensors"
FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}

# Folders that exist inside HF snapshots but never belong in a model folder.
_NON_MODEL_NAMES = {".cache", "refs"}


def fail(msg: str) -> None:
    """Print an error message and exit non-zero."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read_header(path: Path) -> tuple[dict, int]:
    """Parse a .safetensors header.

    Returns ``(tensors, data_start)`` where ``tensors`` maps tensor name to its
    header entry (dtype, shape, data_offsets) and ``data_start`` is the byte
    offset where the tensor data section begins.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def read_headers(folder: Path) -> dict[str, tuple[str, str]]:
    """Map tensor name -> (file name, dtype) from safetensors headers only.

    Tensors are never loaded; this reads headers, so it is fast and needs no
    tensor library.
    """
    info: dict[str, tuple[str, str]] = {}
    for f in sorted(folder.glob("*.safetensors")):
        try:
            header, _ = read_header(f)
        except (json.JSONDecodeError, struct.error, OSError) as exc:
            fail(f"cannot read {f.name}: {exc}")
        for name, meta in header.items():
            info[name] = (f.name, meta["dtype"])
    return info


def copy_support_files(src: Path, dst: Path) -> list[str]:
    """Copy every non-weight file (tokenizer, configs, chat template, ...).

    Weights, the weight index and hidden files (``.cache``, ``.git``, ...) are
    skipped. Existing files in ``dst`` are never overwritten.
    """
    copied = []
    for item in sorted(src.iterdir()):
        name = item.name
        if (
            name.endswith(".safetensors")
            or name in {INDEX_NAME, *_NON_MODEL_NAMES}
            or name.startswith(".")
        ):
            continue
        target = dst / name
        if target.exists():
            continue  # never overwrite what the quantizer/resharder wrote
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied.append(name)
    return copied


def check_index(folder: Path) -> list[str]:
    """Verify that the shard index and the .safetensors files on disk agree.

    Returns a list of human-readable problems (empty if the folder is fine).
    """
    idx_path = folder / INDEX_NAME
    if not idx_path.is_file():
        return [f"{folder} has no {INDEX_NAME}"]
    try:
        index = json.loads(idx_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{INDEX_NAME} is not valid JSON: {exc}"]
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        return [f"{INDEX_NAME} has an empty weight_map"]

    on_disk = read_headers(folder)
    shard_files = {f.name for f in folder.glob("*.safetensors")}

    bad_file = sorted(n for n, f in weight_map.items() if f not in shard_files)
    if bad_file:
        problems = [f"{len(bad_file)} tensors point at missing shard files, e.g. {bad_file[:3]}"]
    else:
        problems = []

    missing = sorted(set(weight_map) - set(on_disk))
    extra = sorted(set(on_disk) - set(weight_map))
    if missing:
        problems.append(f"{len(missing)} indexed tensors are missing on disk, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} on-disk tensors are not indexed, e.g. {extra[:3]}")

    mismatch = sorted(
        n for n in set(weight_map) & set(on_disk) if weight_map[n] != on_disk[n][0]
    )
    if mismatch:
        problems.append(
            f"{len(mismatch)} tensors are in a different shard than the index says, e.g. {mismatch[:3]}"
        )
    return problems
