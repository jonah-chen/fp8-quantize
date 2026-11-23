"""Command line interface for fp8-quantize (``python -m fp8_quantize``)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .pipeline import PipelineConfig, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fp8-quantize",
        description=(
            "Turn a BF16 model into block-wise FP8 (128x128) with small shards: "
            "download from the Hugging Face Hub (when needed, after asking) -> "
            "reshard -> quantize -> reshard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id (e.g. 'Qwen/Qwen3.5-4B') or a path to a local model folder in BF16",
    )
    p.add_argument("--out", required=True, help="output folder for the final FP8 model")
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="answer yes to the download prompt, so the run needs no user input",
    )
    p.add_argument(
        "--max-shard-gb",
        type=float,
        default=4.0,
        help="target shard size in GB for both reshard passes",
    )
    p.add_argument("--max-workers", type=int, default=4, help="llmcompressor shard workers")
    p.add_argument(
        "--device",
        default=None,
        help="quantization device, e.g. cuda:0 (default: llmcompressor default, first GPU)",
    )
    p.add_argument("--overwrite", action="store_true", help="write into a non-empty --out folder")
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="keep the intermediate reshard/quant folders after a successful run",
    )
    p.add_argument(
        "--temp-dir",
        default=None,
        help="folder for intermediate data (default: a hidden folder next to --out)",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face token for gated models (default: $HF_TOKEN)",
    )
    p.add_argument("--revision", default=None, help="HF branch / tag / commit to download")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_shard_gb <= 0:
        raise SystemExit("ERROR: --max-shard-gb must be > 0")
    if args.max_workers < 1:
        raise SystemExit("ERROR: --max-workers must be >= 1")
    cfg = PipelineConfig(
        model=args.model,
        out=Path(args.out),
        max_shard_gb=args.max_shard_gb,
        assume_yes=args.yes,
        max_workers=args.max_workers,
        device=args.device,
        overwrite=args.overwrite,
        keep_temp=args.keep_temp,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
        token=args.token,
        revision=args.revision,
    )
    run(cfg)


if __name__ == "__main__":
    main()
