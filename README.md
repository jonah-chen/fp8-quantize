# fp8-quantize

Take a BF16 LLM — downloaded from the Hugging Face Hub or already on your
machine — and turn it into a block-wise FP8 (128x128) model with 4 GB shards:

```
[download from HF] -> reshard to 4 GB -> quantize (BF16 -> FP8) -> reshard to 4 GB
```

The output is a complete model folder that vLLM, SGLang and transformers
(with compressed-tensors installed) can load directly.

## Requirements

- A CUDA GPU (quantization runs through [llmcompressor](https://github.com/vllm-project/llm-compressor))
- Python 3.10+
- Disk space: roughly **2x the BF16 model size** of scratch space during the run
  (one reshard copy + quantized + final output)

## Install (uv)

```bash
uv sync                 # creates .venv and installs all dependencies
```

…or, if you prefer plain venvs:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

## Usage

```bash
uv run python -m fp8_quantize --model Qwen/Qwen3.5-4B --out /data/Qwen3.5-4B-FP8
```

The program walks through the pipeline in four steps:

1. **resolve the source** — if `--model` is an existing local folder it is used
   as-is (no download). Otherwise it is treated as a Hugging Face repo id and
   the program **asks before downloading**.
2. **reshard** the source into shards of at most `--max-shard-gb` (default 4 GB).
   Skipped when the source shards already fit.
3. **quantize** BF16 -> FP8 with `FP8_BLOCK` (128x128 blocks, dynamic
   activations). Embeddings, `lm_head`, vision towers, gated-delta-net linear
   attention, MoE routers and MTP heads stay BF16; any tensor the quantizer
   didn't write is restored in BF16.
4. **reshard** the quantized output into the final `--out` folder, again at
   `--max-shard-gb`, and verify that the index, config and shards all agree.

Intermediate data lives in a hidden folder next to `--out`
(`.<out-name>.__tmp__`) and is deleted after a successful run.

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--model` | — | HF repo id or path to a local BF16 model folder (required) |
| `--out` | — | output folder for the final FP8 model (required) |
| `--yes`, `-y` | off | answer yes to the download prompt up front (non-interactive) |
| `--max-shard-gb` | `4.0` | target shard size for both reshard passes |
| `--max-workers` | `4` | llmcompressor shard workers |
| `--device` | first GPU | quantization device, e.g. `cuda:0` |
| `--overwrite` | off | allow a non-empty `--out` |
| `--keep-temp` | off | keep intermediate folders after a successful run |
| `--temp-dir` | hidden dir next to `--out` | where intermediate data goes |
| `--token` | `$HF_TOKEN` | HF token for gated models |
| `--revision` | repo default | HF branch / tag / commit to download |

## Non-interactive runs

When `--model` is not a local folder the program asks:

```
Download 'Qwen/Qwen3.5-4B' from the Hugging Face Hub? [y/N]
```

For scripts, CI and anything else without a terminal, pass `--yes` to skip the
prompt entirely. Without `--yes` and without a TTY the program refuses to
download and exits with an error.

## Development

```bash
uv run pytest          # unit tests: no GPU, no network, no model needed
```
