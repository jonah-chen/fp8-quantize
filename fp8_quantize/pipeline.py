"""Orchestrate the full pipeline: [download] -> reshard -> quantize -> reshard.

Steps:
  1. resolve the source (local folder or HF download, with confirmation)
  2. reshard the source into shards of at most ``max_shard_gb`` (skipped when
     the source shards already fit)
  3. quantize BF16 -> FP8 (block 128x128) with llmcompressor, then complete
     and verify the quantized folder
  4. reshard the quantized output into ``out`` with the same shard size

Intermediate data lives in a hidden folder next to ``out`` (or ``--temp-dir``)
and is deleted after a successful run.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import download, quantize
from . import reshard as reshard_mod
from .utils import check_index, fail


@dataclass
class PipelineConfig:
    model: str
    out: Path
    max_shard_gb: float = 4.0
    assume_yes: bool = False
    max_workers: int = 4
    device: str | None = None
    overwrite: bool = False
    keep_temp: bool = False
    temp_dir: Path | None = None
    token: str | None = None
    revision: str | None = None


@dataclass
class _Temp:
    """Tracks the intermediate folders we create, so cleanup is precise."""

    base: Path
    created_base: bool = False
    dirs: list[Path] = field(default_factory=list)

    def subdir(self, name: str) -> Path:
        d = self.base / name
        d.mkdir(parents=True, exist_ok=True)
        self.dirs.append(d)
        return d

    def cleanup(self, keep: bool) -> None:
        if keep:
            if self.dirs:
                print(f"Keeping intermediate folders under {self.base} (delete with rm -rf)")
            return
        for d in self.dirs:
            shutil.rmtree(d, ignore_errors=True)
        if self.created_base and self.base.exists():
            try:
                self.base.rmdir()
            except OSError:
                pass


def _tmp(cfg: PipelineConfig, out: Path) -> _Temp:
    if cfg.temp_dir:
        base = cfg.temp_dir.expanduser().resolve()
    else:
        base = out.parent / f".{out.name}.__tmp__"
    created = not base.exists()
    base.mkdir(parents=True, exist_ok=True)
    return _Temp(base, created_base=created)


def _validate_source(src: Path) -> None:
    if not (src / "config.json").is_file():
        fail(f"{src} has no config.json - point --model at a model folder, not a file")
    if not list(src.glob("*.safetensors")):
        fail(f"no .safetensors files found in {src}")


def _warn_disk_space(out: Path, src: Path) -> None:
    """Print a warning when the scratch volume likely cannot hold the pipeline."""
    try:
        src_gb = sum(f.stat().st_size for f in src.glob("*.safetensors")) / 1e9
        probe = out.parent if out.parent.exists() else Path("/")
        usage = shutil.disk_usage(str(probe))
    except OSError:
        return
    needed = 2 * src_gb  # reshard copy + quantized + final output
    if usage.free < needed:
        print(
            f"WARNING: only {usage.free / 1e9:.1f} GB free on {probe}, "
            f"the pipeline wants ~{needed:.0f} GB of scratch space"
        )


def run(cfg: PipelineConfig) -> Path:
    """Run the pipeline and return the final output folder."""
    out = cfg.out.expanduser().resolve()
    src = download.resolve_source(
        cfg.model,
        assume_yes=cfg.assume_yes,
        token=cfg.token,
        revision=cfg.revision,
    )
    _validate_source(src)
    if src == out or src in out.parents or out in src.parents:
        fail("--out and the source model folder must not be the same or nested in each other")
    if out.exists() and any(out.iterdir()) and not cfg.overwrite:
        fail(f"{out} is not empty (use --overwrite to write into it anyway)")

    tmp = _tmp(cfg, out)
    _warn_disk_space(out, src)
    failed = False
    try:
        max_bytes = int(cfg.max_shard_gb * 1e9)

        # --- step 1: reshard the source so the quantizer sees small shards ---
        if reshard_mod.shards_fit_limit(src, max_bytes):
            print(f"[1/3] source shards are already <= {cfg.max_shard_gb:g} GB - skipping reshard")
            quant_src = src
        else:
            quant_src = tmp.subdir("reshard")
            print(f"[1/3] resharding source to {cfg.max_shard_gb:g} GB shards -> {quant_src}")
            reshard_mod.reshard_folder(src, quant_src, cfg.max_shard_gb)

        # --- step 2: quantize ---
        quant_dir = tmp.subdir("quant")
        print(f"[2/3] quantizing BF16 -> FP8 (128x128 blocks, dynamic activations) -> {quant_dir}")
        quantize.quantize_model(quant_src, quant_dir, max_workers=cfg.max_workers, device=cfg.device)
        quantize.finalize_output(quant_src, quant_dir)
        problems, stats = quantize.verify_output(quant_src, quant_dir)
        if problems:
            print("PROBLEMS FOUND:")
            for msg in problems:
                print(f"  - {msg}")
            fail("the quantized output failed verification")
        print(
            f"Size: {stats['src_gb']:.1f} GB -> {stats['dst_gb']:.1f} GB "
            f"({stats['dst_gb'] / stats['src_gb']:.0%}) | FP8 tensors: {stats['fp8_count']}"
        )
        if not stats["src_has_mtp"]:
            print("note: no mtp.* tensors in the source, so MTP speculative decoding won't be available")

        # --- step 3: reshard the FP8 output into the final folder ---
        print(f"[3/3] resharding FP8 output to {cfg.max_shard_gb:g} GB shards -> {out}")
        reshard_mod.reshard_folder(quant_dir, out, cfg.max_shard_gb)
        problems = check_index(out)
        if problems:
            print("PROBLEMS FOUND:")
            for msg in problems:
                print(f"  - {msg}")
            fail("the final output failed verification")
    except BaseException:
        failed = True
        raise
    finally:
        # on failure keep the temp folders around for debugging
        tmp.cleanup(keep=cfg.keep_temp or failed)

    total = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    n_shards = len(list(out.glob("*.safetensors")))
    print(f"\nDone: {n_shards} shards, {total:.1f} GB -> {out}")
    print(f"Serve with: sglang serve --model-path {out}")
    return out
