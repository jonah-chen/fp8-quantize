"""
Quantize a finetuned Qwen3.5 / Qwen3.6 model (dense or MoE) from BF16 to
block-wise FP8 (128x128 weight blocks, dynamic activations), the same scheme
as Qwen's official *-FP8 releases.

The output is a complete model folder that vLLM, SGLang and transformers
(with compressed-tensors installed) can load directly:
  * quantized weights + config.json with quantization_config
  * every other file from the source folder (tokenizer, chat template,
    preprocessor / video preprocessor configs, generation config, ...)
  * any tensor the quantizer didn't write (e.g. MTP heads) restored in BF16

Usage:
  pip install -U llmcompressor compressed-tensors safetensors
  python quantize_qwen35_fp8.py --model /path/to/finetuned --out /path/to/finetuned-FP8

The source folder is only read, never modified.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Kept in BF16: output head, embeddings, vision tower, gated-delta-net
# linear attention, MoE routers, and the MTP draft head.
IGNORE = [
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*visual.*",
    "re:.*linear_attn.*",
    "re:.*mlp.gate$",
    "re:.*shared_expert_gate$",
    "re:.*mtp.*",
]

INDEX_NAME = "model.safetensors.index.json"
RESTORED_NAME = "restored-bf16.safetensors"
FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def read_headers(folder: Path) -> dict[str, tuple[str, str]]:
    """tensor name -> (file name, dtype), from safetensors headers only."""
    info = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                info[name] = (f.name, sf.get_slice(name).get_dtype())
    return info


def copy_support_files(src: Path, dst: Path) -> list[str]:
    """Copy everything except weights, the weight index and config.json."""
    copied = []
    for item in sorted(src.iterdir()):
        name = item.name
        if (
            name.endswith(".safetensors")
            or name in {INDEX_NAME, "config.json"}
            or name.startswith(".")  # .git, .cache, ...
        ):
            continue
        target = dst / name
        if target.exists():
            continue  # never overwrite what the quantizer wrote
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(name)
    return copied


def restore_missing_tensors(src: Path, src_info: dict, dst: Path) -> list[str]:
    """Copy (unchanged) any source tensor that is absent from the output."""
    dst_info = read_headers(dst)
    missing = sorted(set(src_info) - set(dst_info))
    if not missing:
        return []

    by_file: dict[str, list[str]] = {}
    for name in missing:
        by_file.setdefault(src_info[name][0], []).append(name)

    tensors = {}
    for fname, names in by_file.items():
        with safe_open(src / fname, framework="pt") as sf:
            for name in names:
                tensors[name] = sf.get_tensor(name)
    save_file(tensors, dst / RESTORED_NAME, metadata={"format": "pt"})

    # Rebuild the index from what is actually on disk so every tensor resolves.
    weight_map = {name: f for name, (f, _) in read_headers(dst).items()}
    total = sum(f.stat().st_size for f in dst.glob("*.safetensors"))
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (dst / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    return missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="finetuned BF16 model folder (or HF repo id)")
    p.add_argument("--out", required=True, help="output folder for the FP8 model")
    p.add_argument("--max-workers", type=int, default=4, help="parallel shard workers")
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: GPU if available)")
    p.add_argument("--overwrite", action="store_true", help="allow a non-empty --out folder")
    args = p.parse_args()

    src = Path(args.model)
    if not src.is_dir():
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.model} ...")
        src = Path(snapshot_download(args.model))
    src = src.resolve()
    dst = Path(args.out).resolve()

    # ---- checks before doing any work ----
    if not (src / "config.json").is_file():
        fail(f"{src} has no config.json - point --model at the model folder, not a file")
    if src == dst:
        fail("--out must be different from --model")
    if dst.exists() and any(dst.iterdir()) and not args.overwrite:
        fail(f"{dst} is not empty (use --overwrite to write into it anyway)")

    src_info = read_headers(src)
    if not src_info:
        fail(f"no .safetensors files found in {src}")
    src_has_mtp = any(n.startswith("mtp.") for n in src_info)
    is_moe = any(".experts." in n for n in src_info)
    print(f"Source: {len(src_info)} tensors | MoE: {is_moe} | MTP head: {src_has_mtp}")
    if not src_has_mtp:
        print("  note: no mtp.* tensors in the source, so MTP speculative decoding won't be available")

    # ---- quantize ----
    try:
        from llmcompressor import model_free_ptq
    except ImportError:
        from llmcompressor.entrypoints.model_free import model_free_ptq

    dst.mkdir(parents=True, exist_ok=True)
    model_free_ptq(
        model_stub=str(src),
        save_directory=str(dst),
        scheme="FP8_BLOCK",
        ignore=IGNORE,
        max_workers=args.max_workers,
        device=args.device,
    )

    # ---- complete the output folder ----
    copied = copy_support_files(src, dst)
    print(f"Copied {len(copied)} support files: {', '.join(copied) or '(none)'}")

    restored = restore_missing_tensors(src, src_info, dst)
    if restored:
        print(f"Restored {len(restored)} tensors the quantizer didn't write into {RESTORED_NAME}")

    # ---- verify ----
    problems = []
    cfg_path = dst / "config.json"
    if not cfg_path.is_file():
        problems.append("output has no config.json")
    elif "quantization_config" not in json.loads(cfg_path.read_text()):
        problems.append("output config.json has no quantization_config")

    dst_info = read_headers(dst)
    fp8 = [n for n, (_, dt) in dst_info.items() if dt in FP8_DTYPES]
    if not fp8:
        problems.append("no FP8 tensors in the output - nothing was quantized")
    if is_moe and not any(".experts." in n for n in fp8):
        problems.append("MoE expert weights were not quantized (check the expert tensor layout)")
    if src_has_mtp and not any(n.startswith("mtp.") for n in dst_info):
        problems.append("MTP tensors are missing from the output")
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        if (src / name).exists() and not (dst / name).exists():
            problems.append(f"{name} is missing from the output")

    src_gb = sum(f.stat().st_size for f in src.glob("*.safetensors")) / 1e9
    dst_gb = sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9
    print(f"Size: {src_gb:.1f} GB -> {dst_gb:.1f} GB ({dst_gb / src_gb:.0%}) | FP8 tensors: {len(fp8)}")

    if problems:
        print("\nPROBLEMS FOUND:")
        for msg in problems:
            print(f"  - {msg}")
        sys.exit(1)
    print(f"\nDone. Serve with: sglang serve --model-path {dst}")


if __name__ == "__main__":
    main()
