"""Pipeline tests with a stubbed quantizer: no GPU, no network, no model."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from fp8_quantize import download, pipeline, quantize
from fp8_quantize.utils import INDEX_NAME, read_header
from tests.fake_model import make_model_folder, raw_bytes

DMB = 10**6


def _read_all_tensors(folder: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for f in sorted(folder.glob("*.safetensors")):
        header, data_start = read_header(f)
        with open(f, "rb") as fh:
            for name, info in header.items():
                start, end = info["data_offsets"]
                fh.seek(data_start + start)
                out[name] = fh.read(end - start)
    return out


def _write_shard(path: Path, tensors: dict[str, bytes], dtype: str) -> None:
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, data in tensors.items():
        n = len(data)
        header[name] = {"dtype": dtype, "shape": [n], "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for data in tensors.values():
            f.write(data)


def _fake_quantize(src: Path, dst: Path, **kwargs: object) -> None:
    """Stand-in for llmcompressor: same bytes, FP8 headers, plus config."""
    dst.mkdir(parents=True, exist_ok=True)
    tensors = _read_all_tensors(src)
    _write_shard(dst / "model-00001-of-00001.safetensors", tensors, "F8_E4M3")
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "compressed-tensors", "fmt": "e4m3"}
    (dst / "config.json").write_text(json.dumps(cfg))


def _cfg(model: str, out: Path, **kw: object) -> pipeline.PipelineConfig:
    return pipeline.PipelineConfig(model=model, out=out, assume_yes=True, **kw)


def test_full_pipeline_local_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, c = raw_bytes(4 * DMB, 1), raw_bytes(4 * DMB, 2), raw_bytes(4 * DMB, 3)
    src = make_model_folder(
        tmp_path / "src",
        {
            "model-00001-of-00002.safetensors": {"w.a": a, "w.b": b},
            "model-00002-of-00002.safetensors": {"w.c": c},
        },
        extra={"tokenizer.json": "{}"},
    )
    out = tmp_path / "out"
    monkeypatch.setattr(quantize, "quantize_model", _fake_quantize)

    # 9 MB limit: the 12 MB quantized output must be split back into two shards
    result = pipeline.run(_cfg(str(src), out, max_shard_gb=0.009))

    assert result == out.resolve()
    shards = sorted(p.name for p in out.glob("*.safetensors"))
    assert shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    index = json.loads((out / INDEX_NAME).read_text())
    assert len(index["weight_map"]) == 3

    # bytes survived the whole trip, headers now say FP8
    header, _ = read_header(out / shards[0])
    assert header
    assert all(v["dtype"] == "F8_E4M3" for v in header.values())
    assert _read_all_tensors(out) == {"w.a": a, "w.b": b, "w.c": c}

    # config + support files landed in the final folder
    assert "quantization_config" in json.loads((out / "config.json").read_text())
    assert (out / "tokenizer.json").read_text() == "{}"

    # no download for a local source, temp folders cleaned up
    assert not (out.parent / ".out.__tmp__").exists()


def test_pipeline_keeps_temp_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_model_folder(
        tmp_path / "src", {"model-00001-of-00001.safetensors": {"w.a": raw_bytes(4 * DMB, 1)}}
    )
    out = tmp_path / "out"
    monkeypatch.setattr(quantize, "quantize_model", _fake_quantize)

    pipeline.run(_cfg(str(src), out, keep_temp=True))
    assert (out.parent / ".out.__tmp__" / "quant").is_dir()


def test_pipeline_keeps_temp_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_model_folder(
        tmp_path / "src", {"model-00001-of-00001.safetensors": {"w.a": raw_bytes(4 * DMB, 1)}}
    )
    out = tmp_path / "out"

    def broken(_src: Path, _dst: Path, **_kw: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(quantize, "quantize_model", broken)
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run(_cfg(str(src), out))

    # temp base kept for debugging, final output incomplete
    assert (out.parent / ".out.__tmp__").exists()
    assert not (out / INDEX_NAME).exists()


def test_out_must_be_empty_without_overwrite(tmp_path: Path) -> None:
    src = make_model_folder(
        tmp_path / "src", {"model-00001-of-00001.safetensors": {"w.a": raw_bytes(DMB, 1)}}
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk.txt").write_text("x")
    with pytest.raises(SystemExit):
        pipeline.run(_cfg(str(src), out))


def test_overwrite_allows_nonempty_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_model_folder(
        tmp_path / "src", {"model-00001-of-00001.safetensors": {"w.a": raw_bytes(DMB, 1)}}
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk.txt").write_text("x")
    monkeypatch.setattr(quantize, "quantize_model", _fake_quantize)
    pipeline.run(_cfg(str(src), out, overwrite=True))
    assert (out / INDEX_NAME).is_file()


def test_nested_out_rejected(tmp_path: Path) -> None:
    src = make_model_folder(
        tmp_path / "src", {"model-00001-of-00001.safetensors": {"w.a": raw_bytes(DMB, 1)}}
    )
    with pytest.raises(SystemExit):
        pipeline.run(_cfg(str(src), src / "FP8"))


def test_source_must_have_weights(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        pipeline.run(_cfg(str(src), tmp_path / "out"))


def test_resolve_source_uses_local_dir_without_downloading(tmp_path: Path) -> None:
    src = make_model_folder(
        tmp_path / "src", {"model-00001-of-00001.safetensors": {"w.a": raw_bytes(DMB, 1)}}
    )
    got = download.resolve_source(str(src), assume_yes=False)
    assert got == src.resolve()


def test_resolve_source_missing_local_path_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        download.resolve_source(str(tmp_path / "does-not-exist"), assume_yes=True)
