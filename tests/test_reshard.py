"""Tests for the resharder: grouping, byte-faithful copies, index, errors."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fp8_quantize import reshard
from fp8_quantize.utils import INDEX_NAME, read_header
from tests.fake_model import make_model_folder, raw_bytes

DMB = 10**6  # decimal megabyte, matching the code's GB = 1e9


def _shard_tensors(path: Path) -> dict[str, bytes]:
    """Read every tensor's bytes back out of a .safetensors shard."""
    header, data_start = read_header(path)
    out: dict[str, bytes] = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            out[name] = f.read(end - start)
    return out


def test_reshard_groups_tensors_to_target_size(tmp_path: Path) -> None:
    a, b, c = raw_bytes(4 * DMB, 1), raw_bytes(4 * DMB, 2), raw_bytes(4 * DMB, 3)
    src = make_model_folder(
        tmp_path / "src",
        {
            "model-00001-of-00002.safetensors": {"w.a": a, "w.b": b},
            "model-00002-of-00002.safetensors": {"w.c": c},
        },
        extra={"tokenizer.json": "{}"},
    )
    dst = tmp_path / "dst"

    stats = reshard.reshard_folder(src, dst, 0.009)  # 9 MB limit

    assert stats.tensors == 3
    assert stats.shards == 2
    assert stats.total_bytes == 12 * DMB
    shards = sorted(p.name for p in dst.glob("*.safetensors"))
    assert shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]

    index = json.loads((dst / INDEX_NAME).read_text())
    assert index["weight_map"] == {"w.a": shards[0], "w.b": shards[0], "w.c": shards[1]}
    assert index["metadata"]["total_size"] == 12 * DMB

    # bytes are copied verbatim, tensors grouped under the size limit
    assert _shard_tensors(dst / shards[0]) == {"w.a": a, "w.b": b}
    assert _shard_tensors(dst / shards[1]) == {"w.c": c}

    # support files are copied, source untouched
    assert (dst / "tokenizer.json").read_text() == "{}"
    assert (dst / "config.json").exists()
    assert sorted(p.name for p in src.glob("*.safetensors")) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_oversized_tensor_gets_its_own_shard(tmp_path: Path) -> None:
    big, small = raw_bytes(12 * DMB, 1), raw_bytes(2 * DMB, 2)
    src = make_model_folder(
        tmp_path / "src",
        {"a.safetensors": {"w.big": big}, "b.safetensors": {"w.small": small}},
    )
    stats = reshard.reshard_folder(src, tmp_path / "dst", 0.009)

    assert stats.shards == 2
    index = json.loads((tmp_path / "dst" / INDEX_NAME).read_text())
    assert index["weight_map"]["w.big"] == "model-00001-of-00002.safetensors"
    assert index["weight_map"]["w.small"] == "model-00002-of-00002.safetensors"
    assert _shard_tensors(tmp_path / "dst" / index["weight_map"]["w.big"]) == {"w.big": big}


def test_duplicate_tensor_name_fails(tmp_path: Path) -> None:
    data = raw_bytes(DMB, 1)
    src = make_model_folder(
        tmp_path / "src",
        {"a.safetensors": {"w.x": data}, "b.safetensors": {"w.x": data}},
    )
    with pytest.raises(SystemExit):
        reshard.reshard_folder(src, tmp_path / "dst", 0.009)


def test_no_safetensors_fails(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        reshard.reshard_folder(src, tmp_path / "dst", 4.0)


def test_nonempty_dst_fails(tmp_path: Path) -> None:
    src = make_model_folder(tmp_path / "src", {"a.safetensors": {"w": raw_bytes(DMB, 1)}})
    dst = make_model_folder(tmp_path / "dst", {"b.safetensors": {"w": raw_bytes(DMB, 2)}})
    with pytest.raises(SystemExit):
        reshard.reshard_folder(src, dst, 4.0)


def test_shards_fit_limit(tmp_path: Path) -> None:
    src = make_model_folder(tmp_path / "src", {"a.safetensors": {"w": raw_bytes(4 * DMB, 1)}})
    assert reshard.shards_fit_limit(src, 5 * DMB)
    assert not reshard.shards_fit_limit(src, 3 * DMB)
