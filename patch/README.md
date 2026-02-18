# Patch Directory

Contains patches and plugins for vLLM to extend its functionality.

## Directory Structure

```
patch/
└── vllm_confidence_plugin/   # vLLM confidence plugin
    ├── pyproject.toml        # Python package configuration
    └── vllm_confidence_plugin/
        └── __init__.py       # Plugin main code
```

## vllm_confidence_plugin

vLLM confidence tracking plugin that adds per-token confidence calculation and statistical summaries to the vLLM server's chat completion API.

### Features

- **Confidence Calculation**: Compute confidence scores for each token based on top-k logprobs
- **Statistical Summaries**: Provide various confidence metrics including mean, tail mean, sliding window min, etc.
- **Multiple Modes**: Support different output modes for different use cases
- **High Performance**: Vectorized computation using NumPy for optimized large-scale inference

### Installation

```bash
pip install -e vllm_confidence_plugin
```

### Configuration

Configure the plugin via environment variables on the **vLLM server side**:

```bash
# Set top-k value for confidence calculation (default: 20)
export VLLM_CONF_TOPK=20

# Set output mode (default: per_token)
export VLLM_CONF_MODE=stats
```

### Output Modes

| Mode | Description |
|------|-------------|
| `per_token` | Return confidence scores and full logprobs for each token |
| `summary` | Return confidence scores for each token, but without top_logprobs list |
| `stats` | Return only statistical summary, no per-token content (recommended for production) |
| `empty` | Return empty logprobs (minimal overhead) |

### Statistical Metrics

When using `stats` mode, the following metrics are returned:

- `mean_confidence`: Average confidence of all tokens
- `tail_2048_mean_conf`: Average confidence of the last 2048 tokens
- `min_sliding_2048_mean_conf`: Minimum of sliding window (size 2048) means
- `bottom_0.1_sliding_2048_mean_conf`: Mean of the lowest 10% sliding window means
- `bottom_0.5_sliding_2048_mean_conf`: Mean of the lowest 50% sliding window means

### Usage Example

**Server side:**

```bash
export VLLM_CONF_MODE=stats
vllm serve /path/to/model --port 8000
```

**Client side:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}],
    logprobs=True,
    top_logprobs=20
)

# Access confidence statistics
logprobs = response.choices[0].logprobs
if hasattr(logprobs, 'confidence_summary'):
    print(logprobs.confidence_summary)
```

### Technical Implementation

The plugin registers via vLLM's general plugin mechanism and monkey-patches `OpenAIServingChat._create_chat_logprobs` to inject confidence calculation logic.

Key optimizations:

1. **NumPy Vectorization**: Use prefix sums for sliding window calculations, avoiding loops
2. **Partial Sorting**: Use `np.partition` instead of full sorting for bottom-k mean computation
3. **Lazy Evaluation**: Only compute statistics when needed
