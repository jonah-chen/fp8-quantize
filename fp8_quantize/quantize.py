"""Quantize a BF16 model folder to block-wise FP8 with llmcompressor.

Scheme: FP8_BLOCK (128x128 weight blocks, dynamic activations), the same
scheme as Qwen's official *-FP8 releases. The output is a complete model
folder that vLLM, SGLang and transformers (with compressed-tensors installed)
can load directly:
  * quantized weights + config.json with quantization_config
  * every other file from the source folder (tokenizer, chat template,
    preprocessor / video preprocessor configs, generation config, ...)
  * any tensor the quantizer didn't write (e.g. MTP heads) restored in BF16

The source folder is only read, never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

from .utils import (
    FP8_DTYPES,
    INDEX_NAME,
    RESTORED_NAME,
    copy_support_files,
    fail,
    read_headers,
)

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


def quantize_model(
    src: Path, dst: Path, *, max_workers: int = 4, device: str | None = None
) -> None:
    """Run llmcompressor's model-free PTQ: ``src`` (BF16) -> ``dst`` (FP8)."""
    try:
        from llmcompressor import model_free_ptq
    except ImportError:
        try:
            from llmcompressor.entrypoints.model_free import model_free_ptq
        except ImportError:
            fail("llmcompressor is not installed - run `uv sync` (or: pip install llmcompressor)")
    dst.mkdir(parents=True, exist_ok=True)
    model_free_ptq(
        model_stub=str(src),
        save_directory=str(dst),
        scheme="FP8_BLOCK",
        ignore=IGNORE,
        max_workers=max_workers,
        device=device,
    )


def restore_missing_tensors(src: Path, src_info: dict, dst: Path) -> list[str]:
    """Copy (unchanged, in BF16) any source tensor that is absent from ``dst``."""
    dst_info = read_headers(dst)
    missing = sorted(set(src_info) - set(dst_info))
    if not missing:
        return []

    from safetensors import safe_open
    from safetensors.torch import save_file

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


def finalize_output(src: Path, dst: Path) -> None:
    """Complete the quantized folder: support files + tensors the quantizer skipped."""
    copied = copy_support_files(src, dst)
    print(f"Copied {len(copied)} support files: {', '.join(copied) or '(none)'}")
    src_info = read_headers(src)
    restored = restore_missing_tensors(src, src_info, dst)
    if restored:
        print(f"Restored {len(restored)} tensors the quantizer didn't write into {RESTORED_NAME}")


def verify_output(src: Path, dst: Path) -> tuple[list[str], dict]:
    """Sanity-check a quantized folder. Returns ``(problems, stats)``."""
    problems: list[str] = []
    src_info = read_headers(src)
    dst_info = read_headers(dst)
    is_moe = any(".experts." in n for n in src_info)
    src_has_mtp = any(n.startswith("mtp.") for n in src_info)

    cfg_path = dst / "config.json"
    if not cfg_path.is_file():
        problems.append("output has no config.json")
    elif "quantization_config" not in json.loads(cfg_path.read_text()):
        problems.append("output config.json has no quantization_config")

    fp8 = [n for n, (_, dt) in dst_info.items() if dt in FP8_DTYPES]
    if not fp8:
        problems.append("no FP8 tensors in the output - nothing was quantized")
    if is_moe and not any(".experts." in n for n in fp8):
        problems.append("MoE expert weights were not quantized (check the expert tensor layout)")
    if src_has_mtp and not any(n.startswith("mtp.") for n in dst_info):
        problems.append("MTP tensors are missing from the output")
    for name in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
    ):
        if (src / name).exists() and not (dst / name).exists():
            problems.append(f"{name} is missing from the output")

    src_gb = sum(f.stat().st_size for f in src.glob("*.safetensors")) / 1e9
    dst_gb = sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9
    stats = {
        "src_gb": src_gb,
        "dst_gb": dst_gb,
        "fp8_count": len(fp8),
        "is_moe": is_moe,
        "src_has_mtp": src_has_mtp,
    }
    return problems, stats
