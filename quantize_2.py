"""
Split a model's safetensors shards into smaller ones, without loading the model.

Tensors are copied byte-for-byte (no dtype conversion, no torch needed), one at
a time, so memory use stays tiny. Every tensor in every *.safetensors file in
the folder ends up in the new shards and the new index. All other files
(config.json, tokenizer, processor configs, chat template, ...) are copied as-is.
The source folder is only read, never modified.

Usage:
  python reshard_safetensors.py --model /path/to/finetuned --out /path/to/finetuned-5gb --max-shard-gb 5
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
COPY_CHUNK = 64 * 1024 * 1024  # 64 MB


def read_header(path: Path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n  # tensors, byte offset where data starts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model folder to reshard")
    p.add_argument("--out", required=True, help="new folder for the resharded model")
    p.add_argument("--max-shard-gb", type=float, default=4.0)
    args = p.parse_args()

    src, dst = Path(args.model).resolve(), Path(args.out).resolve()
    if src == dst:
        sys.exit("ERROR: --out must be different from --model")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"ERROR: {dst} is not empty")
    files = sorted(src.glob("*.safetensors"))
    if not files:
        sys.exit(f"ERROR: no .safetensors files in {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # Collect every tensor: (name, source file, data start, info)
    tensors, seen = [], set()
    for f in files:
        header, data_start = read_header(f)
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if name in seen:
                sys.exit(f"ERROR: tensor {name} appears in more than one file")
            seen.add(name)
            tensors.append((name, f, data_start, info))

    # Group into shards of at most max_bytes (a single bigger tensor gets its own shard)
    max_bytes = int(args.max_shard_gb * 1e9)
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
            header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + nbytes]}
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
                            sys.exit(f"ERROR: {srcfile.name} is truncated")
                        out.write(chunk)
                        remaining -= len(chunk)
        print(f"Wrote {fname} ({offset / 1e9:.2f} GB, {len(shard)} tensors)")

    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (dst / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

    # Copy everything else unchanged
    for item in sorted(src.iterdir()):
        if item.name.endswith(".safetensors") or item.name == INDEX_NAME or item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)

    print(f"\nDone: {len(tensors)} tensors in {len(shards)} shards ({total / 1e9:.1f} GB) -> {dst}")


if __name__ == "__main__":
    main()
