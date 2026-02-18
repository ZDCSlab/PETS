# PETS - Performance Evaluation & Testing Suite

LLM reasoning benchmark evaluation framework using vLLM for inference. Supports multiple-choice and open-ended math benchmarks with confidence tracking and majority voting.

## Supported Benchmarks

| Script | Benchmark | Type | Answer Format |
|--------|-----------|------|---------------|
| `aime24.py` | AIME 2024 | Math | `\boxed{}` |
| `aime25.py` | AIME 2025 | Math | `\boxed{}` |
| `hmmt.py` | HMMT Feb 2025 | Math | `\boxed{}` (equivalence voting) |
| `brumo.py` | BRUMO 2025 | Math | `\boxed{}` (equivalence voting) |
| `gpqa.py` | GPQA Diamond | Multiple Choice | A/B/C/D |
| `mmlu_pro.py` | MMLU-Pro | Multiple Choice | A-P (few-shot CoT) |

## Quick Start

### 1. Install dependencies (includes confidence plugin)

```bash
cd ..
bash install.sh
```

This will install vLLM and the **required** confidence plugin automatically.

### 2. Run a benchmark using launch.sh (recommended)

The `launch.sh` script automatically starts the vLLM server with confidence plugin enabled:

```bash
# Run AIME 2024 with 64 samples on 8 GPUs (TP=1, DP=8)
./launch.sh --model-dir /path/to/your/model --task aime24 --n 64

# Run with custom parallelism: TP=2, DP=4 on 8 GPUs
./launch.sh --model-dir /path/to/your/model --task aime24 --tp-size 2 --dp-size 4

# Run GPQA Diamond with lower temperature
./launch.sh --model-dir /path/to/your/model --task gpqa --n 32 --temp 0.6

# Run MMLU-Pro with custom data path
./launch.sh --model-dir /path/to/your/model --task mmlu_pro --data-path TIGER-Lab/MMLU-Pro

# Start server only (for manual benchmark runs)
./launch.sh --model-dir /path/to/your/model --server-only --tp-size 2 --dp-size 4
```

### 3. Alternative: Manual Server Setup

If you prefer to manage the server separately:

```bash
# 1. Enable the confidence plugin (required)
export VLLM_PLUGINS="confidence_logprobs"
export VLLM_CONF_MODE=stats
export VLLM_CONF_TOPK=20
export VLLM_FLAT_LOGPROBS=1

# 2. Start vLLM server with your desired parallelism
vllm serve /path/to/your/model \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --port 8000 \
  --gpu-memory-utilization 0.97

# 3. Run benchmark (in another terminal)
python aime24.py --n 64 --temperature 1.1 --top_p 0.95
```

### 4. Evaluate existing results

```bash
python aime24.py --out aime24_preds.jsonl --eval-only
python gpqa.py   --out gpqa_preds.jsonl   --eval-only
```

## Common Arguments

All benchmark scripts share these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `localhost` | vLLM server host |
| `--port` | `8000` | vLLM server port |
| `--model_name` / `-m` | auto-detect | Model name on the server |
| `--data_path` | benchmark default | Dataset path (local dir/file or HuggingFace name) |
| `--out` | `<benchmark>_preds.jsonl` | Output JSONL path |
| `--n` | `1` | Number of samples per question |
| `--temperature` | `0.0` | Sampling temperature |
| `--top_p` | `1.0` | Nucleus sampling top-p |
| `--top_logprobs` | `20` | Top-k logprobs for confidence |
| `--max_workers` | auto | Max concurrent requests |
| `--limit` | none | Process only first N questions |
| `--eval-only` | off | Evaluate existing results without inference |

### Model-Specific Behavior

**gpt-oss models** (model name contains `gpt`):
- `temperature` and `top_p` are automatically forced to `1` regardless of user settings
- `extra_body={"reasoning_effort": "high"}` is added to API requests

These overrides are transparent — no special flags needed.

### Benchmark-Specific Arguments

**GPQA**: `--subset` (default: `gpqa_diamond`), `--seed` (default: `42`)

**MMLU-Pro**: `--selected_subjects` / `-sub` (default: `all`), `--ntrain` / `-k` (default: `5`), `--initial_prompt`

## Using launch.sh

The `launch.sh` script provides a complete solution that:
1. Starts vLLM server with confidence plugin enabled
2. Waits for server to be ready
3. Runs the benchmark
4. Cleans up the server on exit

### Command-Line Arguments

```bash
./launch.sh --model-dir /path/to/model --task aime24 [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | *(required)* | Path to the model |
| `--task` | *(required)* | Task: aime24, aime25, hmmt, brumo, gpqa, mmlu_pro |
| `--n` | `64` | Number of samples per question |
| `--tp-size` | `1` | Tensor parallel size |
| `--dp-size` | `1` | Data parallel size |
| `--api-server-count` | `2 * dp-size` | Number of API server processes |
| `--enable-expert-parallel` | `0` | Enable for MoE models (flag => 1, or pass 0/1) |
| `--temp` | `1.1` | Sampling temperature |
| `--top-p` | `0.95` | Nucleus sampling top-p |
| `--top-logprobs` | `20` | Top-k logprobs for confidence |
| `--port` | `8000` | API server port |
| `--out` | auto | Output file path |
| `--data-path` | benchmark default | Dataset path |
| `--max-workers` | `16` | Max concurrent requests |
| `--gpu-memory-utilization` | `0.97` | GPU memory utilization |
| `--server-only` | off | Only start server, don't run benchmark |

### Examples

```bash
# Basic: AIME 2024 with 64 samples on 8 GPUs
./launch.sh --model-dir /path/to/model --task aime24 --n 64

# Custom parallelism: TP=2, DP=4
./launch.sh --model-dir /path/to/model --task aime24 --tp-size 2 --dp-size 4

# Large model: all 8 GPUs for one replica
./launch.sh --model-dir /path/to/huge/model --task aime24 --tp-size 8 --dp-size 1

# MoE model with expert parallel
./launch.sh --model-dir /path/to/moe/model --task aime24 --enable-expert-parallel 1

# Server-only mode for manual testing
./launch.sh --model-dir /path/to/model --server-only --tp-size 2

# Use local AIME24 parquet folder
./launch.sh --model-dir /path/to/model --task aime24 --data-path /path/to/aime24_dir

# Use local AIME25 jsonl file/folder
./launch.sh --model-dir /path/to/model --task aime25 --data-path /path/to/aime25.jsonl
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_CONF_MODE` | Confidence output mode | stats |
| `VLLM_CONF_TOPK` | Top-k for confidence | 20 |
| `SERVER_READY_TIMEOUT` | Server startup timeout (sec) | 7200 |
| `LOG_DIR` | Log directory | ./logs |
| `RESULTS_DIR` | Results directory | ./results |

## Dataset Formats

| Benchmark | Format | Source |
|-----------|--------|--------|
| AIME 2024 | Local parquet directory or HuggingFace | `HuggingFaceH4/aime_2024` |
| AIME 2025 | Local JSONL or HuggingFace | `opencompass/AIME2025` |
| HMMT Feb 2025 | Local parquet directory or HuggingFace | `MathArena/hmmt_feb_2025` |
| BRUMO 2025 | Local parquet directory or HuggingFace | `MathArena/brumo_2025` |
| GPQA Diamond | HuggingFace or local dataset path | `Idavidrein/gpqa` |
| MMLU-Pro | HuggingFace or local dataset path | `TIGER-Lab/MMLU-Pro` |

## Local Dataset Placement

`--data-path` can be absolute or relative. Local datasets do not need to be under `PETS/`; they can be placed anywhere.

Recommended optional layout:

```text
PETS/reasoning/datasets/
  aime24/data/*.parquet
  hmmt/data/*.parquet
  brumo/data/*.parquet
  aime25/*.jsonl
```

Task-specific local path requirements:

- `aime24`, `hmmt`, `brumo`: pass a directory containing `*.parquet` files, or `data/*.parquet`.
- `aime25`: pass a single `.jsonl` file, or a directory containing `.jsonl` files.
- `gpqa`: pass a HuggingFace dataset name, or a local path that `datasets.load_dataset(...)` can read.
- `mmlu_pro`: pass a HuggingFace dataset name, or a local path with `test` and `validation` splits.

## Project Structure

```
PETS/reasoning/
├── common.py          # Shared utilities (client, inference loop, confidence, voting)
├── aime24.py          # AIME 2024
├── aime25.py          # AIME 2025
├── hmmt.py            # HMMT February 2025
├── brumo.py           # BRUMO 2025
├── gpqa.py            # GPQA Diamond
├── mmlu_pro.py        # MMLU-Pro (few-shot CoT)
├── launch.sh          # Complete launcher (server + benchmark)
├── logs/              # vLLM server logs (auto-created)
└── results/           # Benchmark results (auto-created)
```

## Output Format

All benchmarks output JSONL files with one result per line:

```json
{
  "id": "...",
  "problem": "...",
  "answer": "ground_truth",
  "pred": "voted_prediction",
  "answers": ["ans1", "ans2", "..."],
  "raw_outputs": ["full_text1", "full_text2", "..."],
  "trace_confidence": [{"sample_idx": 0, "conf_summary": {...}}, ...],
  "final_trace": {...},
  "success": true
}
```
