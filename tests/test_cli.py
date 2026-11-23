"""Tests for the CLI argument parsing."""
from __future__ import annotations

import pytest

from fp8_quantize.cli import build_parser


def test_defaults() -> None:
    args = build_parser().parse_args(["--model", "Qwen/Qwen3.5-4B", "--out", "/data/out"])
    assert args.model == "Qwen/Qwen3.5-4B"
    assert args.out == "/data/out"
    assert args.yes is False
    assert args.max_shard_gb == 4.0
    assert args.max_workers == 4
    assert args.device is None
    assert args.overwrite is False
    assert args.keep_temp is False
    assert args.temp_dir is None
    assert args.token is None
    assert args.revision is None


def test_yes_short_flag() -> None:
    args = build_parser().parse_args(
        ["--model", "m", "--out", "o", "-y", "--max-shard-gb", "2", "--device", "cuda:1"]
    )
    assert args.yes is True
    assert args.max_shard_gb == 2.0
    assert args.device == "cuda:1"


def test_missing_required_args_fails() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "m"])


def test_invalid_max_shard_gb_fails() -> None:
    from fp8_quantize.cli import main

    with pytest.raises(SystemExit, match="max-shard-gb"):
        main(["--model", "m", "--out", "o", "--max-shard-gb", "0"])
    with pytest.raises(SystemExit, match="max-workers"):
        main(["--model", "m", "--out", "o", "--max-workers", "0"])
