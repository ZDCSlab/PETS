#!/usr/bin/env python3
"""PETS shared utilities for inference, evaluation, and confidence tracking."""

import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


# ── Confidence extraction ────────────────────────────────────────────────

CONF_KEYS = (
    "mean_confidence",
    "tail_2048_mean_conf",
    "min_sliding_2048_mean_conf",
    "bottom_0.1_sliding_2048_mean_conf",
    "bottom_0.5_sliding_2048_mean_conf",
)


def _sanitize_conf(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in CONF_KEYS:
        v = d.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out[k] = float(v)
        else:
            out[k] = None
    return out


def extract_conf_summary(choice) -> Dict[str, Any]:
    """Extract confidence summary from a chat completion choice."""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return {"source": "missing", "summary": None}

    summ = getattr(lp, "confidence_summary", None)
    if summ is None:
        try:
            summ = lp.get("confidence_summary")
        except Exception:
            pass

    if summ is not None:
        if hasattr(summ, "model_dump"):
            summ = summ.model_dump()
        elif not isinstance(summ, dict):
            try:
                summ = dict(summ)
            except Exception:
                summ = None

    if isinstance(summ, dict):
        return {"source": "server_stats", "summary": _sanitize_conf(summ)}
    return {"source": "missing", "summary": None}


# ── Voting ───────────────────────────────────────────────────────────────

def vote_majority(answers: List[str]) -> str:
    """Simple majority vote by exact string match."""
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


# ── Math answer utilities ────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, handling nested braces."""
    key = r"\boxed{"
    last = None
    idx = 0
    text = (text or "").strip()
    while True:
        start = text.find(key, idx)
        if start == -1:
            break
        i = start + len(key)
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[start + len(key): i - 1].strip()
            idx = i
        else:
            break
    return last


def _canon_str(s: str) -> str:
    """Canonicalize LaTeX-ish string for raw majority fallback."""
    s = (s or "").strip()
    if not s:
        return ""
    s = (s.replace("\\dfrac", "\\frac")
          .replace("\\tfrac", "\\frac")
          .replace("\\cfrac", "\\frac")
          .replace("\\displaystyle", "")
          .replace(" ", ""))
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    return s


def _parse_math(s: str):
    """Parse a math expression via math_verify with multiple LaTeX wrappers."""
    from math_verify import parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    base = _canon_str(s)
    if not base:
        return []
    for variant in [base, f"${base}$", f"\\({base}\\)", f"\\[{base}\\]"]:
        try:
            out = parse(
                variant,
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
                parsing_timeout=None,
            )
            if out:
                return out
        except Exception:
            continue
    return []


def vote_majority_equiv(answers: List[str]) -> str:
    """Majority vote by mathematical equivalence, with raw canonical fallback."""
    from math_verify import verify

    # Raw canonical grouping
    canon_groups: Dict[str, List[str]] = {}
    for a in answers:
        a0 = (a or "").strip()
        c = _canon_str(a0)
        if c:
            canon_groups.setdefault(c, []).append(a0)

    # Parsed equivalence clustering
    clusters: List[Dict] = []
    for a in answers:
        a0 = (a or "").strip()
        if not a0:
            continue
        parsed = _parse_math(a0)
        if not parsed:
            continue
        placed = False
        for cl in clusters:
            try:
                if verify(cl["rep"], parsed):
                    cl["count"] += 1
                    cl["texts"].append(a0)
                    placed = True
                    break
            except Exception:
                pass
        if not placed:
            clusters.append({"rep": parsed, "texts": [a0], "count": 1})

    best_parsed, best_pc = "", -1
    if clusters:
        best = max(clusters, key=lambda c: (c["count"], -min(len(t) for t in c["texts"])))
        best_pc = best["count"]
        best_parsed = min(best["texts"], key=len)

    best_raw, best_rc = "", -1
    if canon_groups:
        _, exs = max(
            canon_groups.items(),
            key=lambda kv: (len(kv[1]), -min(len(x) for x in kv[1])),
        )
        best_rc = len(exs)
        best_raw = min(exs, key=len)

    return best_raw if best_rc > best_pc else (best_parsed or best_raw)


# ── Model-specific overrides ──────────────────────────────────────────────

def _is_gpt_oss(model: str) -> bool:
    """Check if the model is a gpt-oss variant."""
    return "gpt" in model.lower()


def _needs_fixed_sampling(model: str) -> bool:
    """gpt-oss models require temperature=1, top_p=1."""
    return _is_gpt_oss(model)


def _build_extra_body(model: str) -> Optional[dict]:
    """Build extra_body for the request based on model type.

    For gpt-oss models, reasoning_effort is fixed to 'high'.
    """
    body = {}
    if _is_gpt_oss(model):
        body["reasoning_effort"] = "high"
    return body or None


# ── Client setup ─────────────────────────────────────────────────────────

def create_client(host: str, port: int, model_name: Optional[str] = None):
    """Create an OpenAI-compatible client and resolve the served model name.

    Returns (client, model_name).
    """
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{host}:{port}/v1",
        timeout=float(os.getenv("VLLM_CLIENT_TIMEOUT", 3600)),
    )
    try:
        available = [m.id for m in client.models.list().data]
    except Exception as e:
        sys.exit(f"Cannot connect to vLLM at {host}:{port} - {e}")

    if not available:
        sys.exit("No models available on the server.")

    if model_name and model_name not in available:
        print(f"Warning: '{model_name}' not in {available}; trying anyway.")
        model = model_name
    else:
        model = model_name or available[0]
    print(f"Using model: {model}\n")
    return client, model


# ── Generic question processing ──────────────────────────────────────────

def process_question(
    client,
    model: str,
    item: Dict[str, Any],
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_logprobs: int,
    prompt_text: str,
    extract_fn: Callable[[str], str],
    vote_fn: Callable[[List[str]], str] = vote_majority,
    system_prompt: str = "You are a helpful assistant.",
    gold_answer=None,
    problem_text=None,
) -> Dict[str, Any]:
    """Send one prompt with ``n=n_samples`` and aggregate answers.

    For gpt-oss models, temperature and top_p are automatically overridden
    to 1, and ``extra_body={"reasoning_effort": "high"}`` is added.
    """
    qid = item.get("id") or item.get("question_id") or item.get("problem_id")
    _problem = problem_text or item.get("problem") or item.get("question") or item.get("prompt")
    _gold = gold_answer if gold_answer is not None else (item.get("answer") or item.get("solution"))

    # Override sampling params for gpt-oss models
    _temperature = 1 if _needs_fixed_sampling(model) else temperature
    _top_p = 1 if _needs_fixed_sampling(model) else top_p
    _extra_body = _build_extra_body(model)

    try:
        request_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            n=n_samples,
            temperature=_temperature,
            top_p=_top_p,
            logprobs=True,
            top_logprobs=top_logprobs,
            timeout=None,
        )
        if _extra_body:
            request_kwargs["extra_body"] = _extra_body

        resp = client.chat.completions.create(**request_kwargs)

        answers, raw_outputs, traces = [], [], []
        for idx, choice in enumerate(resp.choices):
            text = (choice.message.content or "").strip()
            ans = extract_fn(text)
            raw_outputs.append(text)
            answers.append(ans)
            conf = extract_conf_summary(choice)
            traces.append({
                "sample_idx": idx,
                "answer": ans,
                "text_len": len(text),
                "finish_reason": getattr(choice, "finish_reason", None),
                "conf_source": conf["source"],
                "conf_summary": conf["summary"],
            })

        non_empty = [a for a in answers if a]
        final = vote_fn(non_empty) if non_empty else (answers[0] if answers else "")
        final_idx = next((i for i, a in enumerate(answers) if a == final), None) if final else None

        return {
            "id": qid, "problem": _problem, "answer": _gold,
            "pred": final, "answers": answers, "raw_outputs": raw_outputs,
            "trace_confidence": traces,
            "final_trace": traces[final_idx] if final_idx is not None else None,
            "success": True,
        }
    except Exception as e:
        return {
            "id": qid, "problem": _problem, "answer": _gold,
            "pred": "", "answers": [], "raw_outputs": [],
            "error": str(e), "trace_confidence": [], "success": False,
        }


# ── Inference loop ───────────────────────────────────────────────────────

def run_inference(
    dataset: List,
    process_fn: Callable,
    output_path: str,
    max_workers: Optional[int] = None,
    desc: str = "Inference",
):
    """Thread-pool inference loop with progress bar and JSONL output."""
    workers = max_workers or min(32, (os.cpu_count() or 8) * 5)
    stats = {"total": 0, "ok": 0, "fail": 0}
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=workers) as pool, \
         tqdm(total=len(dataset), desc=desc, unit="q") as pbar:
        for res in pool.map(process_fn, dataset):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            stats["total"] += 1
            if res.get("success"):
                stats["ok"] += 1
                pbar.set_postfix(pred=res["pred"][:40])
            else:
                stats["fail"] += 1
                pbar.set_postfix(error="X")
            pbar.update(1)

    dt = time.time() - t0
    print(f"\nDone - {stats['ok']} ok, {stats['fail']} failed "
          f"({dt:.1f}s, {stats['total']/dt:.2f} q/s)")
    print(f"Results: {output_path}")


# ── Dataset loaders ──────────────────────────────────────────────────────

def load_parquet_dir(dataset_dir: str) -> List[Dict[str, Any]]:
    """Load parquet files from *dataset_dir* (supports ``data/`` sub-directory)."""
    import numpy as np
    import pandas as pd

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Not a directory: {dataset_dir}")

    data_sub = os.path.join(dataset_dir, "data")
    search = data_sub if os.path.isdir(data_sub) else dataset_dir

    files = sorted(
        os.path.join(search, f) for f in os.listdir(search) if f.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"No .parquet files in {search}")

    def _to_python(obj):
        """Recursively convert numpy types to native Python for JSON."""
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_python(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    rows = []
    for p in files:
        for rec in pd.read_parquet(p).to_dict(orient="records"):
            rows.append(_to_python(rec))
    return rows


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load from a .jsonl file or all .jsonl files in a directory."""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    if not os.path.isdir(path):
        raise ValueError(f"Path not found: {path}")

    rows = []
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(path, fn), encoding="utf-8") as f:
                rows.extend(json.loads(ln) for ln in f if ln.strip())
    if not rows:
        raise ValueError(f"No .jsonl files in {path}")
    return rows


def load_hf_split(dataset_name: str, split: str = "train", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load one split from a HuggingFace dataset name."""
    from datasets import load_dataset

    ds_args = [dataset_name]
    if subset:
        ds_args.append(subset)

    try:
        ds = load_dataset(*ds_args, split=split)
    except Exception:
        if split != "train":
            raise
        for fallback in ("test", "validation"):
            try:
                ds = load_dataset(*ds_args, split=fallback)
                print(f"Warning: split 'train' not found for {dataset_name}; using '{fallback}'.")
                break
            except Exception:
                ds = None
        if ds is None:
            raise

    return [dict(item) for item in ds]


def load_parquet_or_hf(source: str, split: str = "train") -> List[Dict[str, Any]]:
    """Load parquet rows from a local directory or fallback to a HF dataset name."""
    if os.path.exists(source):
        return load_parquet_dir(source)
    return load_hf_split(source, split=split)


def load_jsonl_or_hf(source: str, split: str = "train", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load JSONL rows from local path, or fallback to a HF dataset name."""
    if os.path.exists(source):
        return load_jsonl(source)
    return load_hf_split(source, split=split, subset=subset)


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load inference results from a JSONL file."""
    if not os.path.exists(path):
        sys.exit(f"Results file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ── Common argparse setup ────────────────────────────────────────────────

def add_common_args(ap):
    """Add server, sampling, and inference arguments shared by all benchmarks."""
    ap.add_argument("--host", default="localhost", help="vLLM server host")
    ap.add_argument("--port", type=int, default=8000, help="vLLM server port")
    ap.add_argument("--model_name", "-m", default=None,
                    help="Model name served by vLLM (auto-detect if omitted)")
    ap.add_argument("--n", type=int, default=1, help="Number of samples per question")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    ap.add_argument("--top_logprobs", type=int, default=20, help="Top-k logprobs for confidence")
    ap.add_argument("--max_workers", type=int, default=None, help="Max concurrent requests")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate existing results only")
    return ap
