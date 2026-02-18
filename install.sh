#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install --upgrade pip
pip install vllm==0.15.1 datasets nvitop 'math-verify[antlr4_13_2]'

# Install vLLM confidence plugin (required)
echo "[INFO] Installing vLLM confidence plugin..."
pip install -e "${SCRIPT_DIR}/patch/vllm_confidence_plugin"

echo "[INFO] Installation complete!"
