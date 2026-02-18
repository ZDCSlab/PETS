# vllm_confidence_plugin/__init__.py
import os
import math
from collections import deque
import numpy as np

_PATCHED = False


def _model_has_field(model_cls, field_name: str) -> bool:
    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        fields = getattr(model_cls, "__fields__", {})
    return field_name in fields


def _model_rebuild(model_cls):
    rebuild = getattr(model_cls, "model_rebuild", None)
    if callable(rebuild):
        rebuild(force=True)
        return
    update_refs = getattr(model_cls, "update_forward_refs", None)
    if callable(update_refs):
        update_refs()


def _resolve_serving_chat_and_protocol():
    """Resolve OpenAIServingChat and protocol for latest vLLM layout only."""
    from vllm.entrypoints.openai.chat_completion import serving, protocol

    if not hasattr(serving, "OpenAIServingChat"):
        raise ImportError(
            "Expected OpenAIServingChat in vllm.entrypoints.openai.chat_completion.serving."
        )
    if not (
        hasattr(protocol, "ChatCompletionLogProbs")
        and hasattr(protocol, "ChatCompletionLogProbsContent")
    ):
        raise ImportError(
            "Expected ChatCompletionLogProbs/Content in "
            "vllm.entrypoints.openai.chat_completion.protocol."
        )

    return serving, protocol


def register():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Confidence uses the top-20 logprobs by default.
    # Kept fixed at 20 unless overridden via environment variable.
    K_FOR_CONF = int(os.environ.get("VLLM_CONF_TOPK", "20"))
    # per_token | summary | empty | stats
    MODE = os.environ.get("VLLM_CONF_MODE", "per_token")

    serving_chat, protocol = _resolve_serving_chat_and_protocol()

    # ---------------------------------------------------------------------
    # 0) Add confidence_summary to ChatCompletionLogProbs (used by stats mode).
    # ---------------------------------------------------------------------
    LogProbs = protocol.ChatCompletionLogProbs
    if not _model_has_field(LogProbs, "confidence_summary"):
        LogProbs.__annotations__["confidence_summary"] = dict[str, float | None] | None
        _model_rebuild(LogProbs)

    # ---------------------------------------------------------------------
    # 1) In-place patch: add a confidence field to ChatCompletionLogProbsContent.
    # ---------------------------------------------------------------------
    Content = protocol.ChatCompletionLogProbsContent
    if not _model_has_field(Content, "confidence"):
        Content.__annotations__["confidence"] = float | None
        _model_rebuild(Content)

    # ---------------------------------------------------------------------
    # 2) stats: high-performance summarization (three optimizations live here).
    # ---------------------------------------------------------------------
    def _summarize_confs_fast_numpy(confs_np):
        """
        confs_np: shape [T], dtype float, may contain NaN
        Return dict[str, float|None]
        """

        WINDOW = 2048
        BOTTOM_FRACS = (0.10, 0.50)

        # valid mask
        valid = np.isfinite(confs_np)
        if not np.any(valid):
            return {
                "mean_confidence": None,
                "tail_2048_mean_conf": None,
                "min_sliding_2048_mean_conf": None,
                "bottom_0.1_sliding_2048_mean_conf": None,
                "bottom_0.5_sliding_2048_mean_conf": None,
            }

        # mean_confidence
        mean_conf = float(np.nanmean(confs_np))

        # tail mean
        tail = confs_np[-WINDOW:] if confs_np.shape[0] > WINDOW else confs_np
        tail_mean = float(np.nanmean(tail)) if np.any(np.isfinite(tail)) else None

        # ---- sliding WINDOW mean (vectorized via prefix sums) ----
        # confs_filled = conf where valid else 0
        confs_filled = np.where(valid, confs_np, 0.0).astype(np.float64)
        cnt = valid.astype(np.int64)

        # prefix sums
        prefix_sum = np.empty(confs_filled.shape[0] + 1, dtype=np.float64)
        prefix_sum[0] = 0.0
        np.cumsum(confs_filled, out=prefix_sum[1:])

        prefix_cnt = np.empty(cnt.shape[0] + 1, dtype=np.int64)
        prefix_cnt[0] = 0
        np.cumsum(cnt, out=prefix_cnt[1:])

        T = confs_np.shape[0]
        idx = np.arange(T, dtype=np.int64)
        start = idx - (WINDOW - 1)
        start[start < 0] = 0

        # window sum/cnt for each position i:
        # sum[i] = prefix_sum[i+1] - prefix_sum[start[i]]
        # cnt[i] = prefix_cnt[i+1] - prefix_cnt[start[i]]
        win_sum = prefix_sum[idx + 1] - prefix_sum[start]
        win_cnt = prefix_cnt[idx + 1] - prefix_cnt[start]

        sliding_means = np.full(T, np.nan, dtype=np.float64)
        ok = win_cnt > 0
        sliding_means[ok] = win_sum[ok] / win_cnt[ok]

        if not np.any(np.isfinite(sliding_means)):
            return {
                "mean_confidence": mean_conf,
                "tail_2048_mean_conf": tail_mean,
                "min_sliding_2048_mean_conf": None,
                "bottom_0.1_sliding_2048_mean_conf": None,
                "bottom_0.5_sliding_2048_mean_conf": None,
            }

        # min_sliding: no sort
        min_sliding = float(np.nanmin(sliding_means))

        # bottom frac mean: use partition (no full sort)
        valid_sliding = sliding_means[np.isfinite(sliding_means)].astype(np.float64)
        n = int(valid_sliding.shape[0])

        def _bottom_frac_mean(frac: float):
            k = int(math.ceil(frac * n))
            if k <= 0:
                return None
            if k >= n:
                return float(valid_sliding.mean())
            # kth smallest partition; take first k (unordered)
            part = np.partition(valid_sliding, k - 1)[:k]
            return float(part.mean())

        bottom_01 = _bottom_frac_mean(BOTTOM_FRACS[0])
        bottom_05 = _bottom_frac_mean(BOTTOM_FRACS[1])

        return {
            "mean_confidence": mean_conf,
            "tail_2048_mean_conf": tail_mean,
            "min_sliding_2048_mean_conf": min_sliding,
            "bottom_0.1_sliding_2048_mean_conf": bottom_01,
            "bottom_0.5_sliding_2048_mean_conf": bottom_05,
        }

    def _summarize_confs_fallback_python(confs):
        """
        Pure Python fallback, keeping output semantics close to the original.
        Includes one small optimization: min without sorting; bottom still sorts.
        """
        WINDOW = 2048
        BOTTOM_FRACS = (0.10, 0.50)

        def _is_valid(x):
            return isinstance(x, (int, float)) and not math.isnan(x)

        valid_tokens = [float(c) for c in confs if _is_valid(c)]
        if not valid_tokens:
            return {
                "mean_confidence": None,
                "tail_2048_mean_conf": None,
                "min_sliding_2048_mean_conf": None,
                "bottom_0.1_sliding_2048_mean_conf": None,
                "bottom_0.5_sliding_2048_mean_conf": None,
            }

        mean_conf = float(sum(valid_tokens) / len(valid_tokens))

        tail = confs[-WINDOW:]
        tail_valid = [float(c) for c in tail if _is_valid(c)]
        tail_mean = float(sum(tail_valid) / len(tail_valid)) if tail_valid else None

        dq = deque()
        rolling_sum = 0.0
        rolling_cnt = 0
        sliding_means = []

        for c in confs:
            dq.append(c)
            if _is_valid(c):
                rolling_sum += float(c)
                rolling_cnt += 1

            if len(dq) > WINDOW:
                old = dq.popleft()
                if _is_valid(old):
                    rolling_sum -= float(old)
                    rolling_cnt -= 1

            sliding_means.append((rolling_sum / rolling_cnt) if rolling_cnt > 0 else None)

        valid_sliding = [float(m) for m in sliding_means if _is_valid(m)]
        if not valid_sliding:
            return {
                "mean_confidence": mean_conf,
                "tail_2048_mean_conf": tail_mean,
                "min_sliding_2048_mean_conf": None,
                "bottom_0.1_sliding_2048_mean_conf": None,
                "bottom_0.5_sliding_2048_mean_conf": None,
            }

        # Min without sorting
        min_sliding = float(min(valid_sliding))

        # Bottom still sorts (acceptable for fallback path)
        valid_sliding.sort()

        def _bottom_frac_mean(frac: float):
            n = len(valid_sliding)
            k = max(1, int(math.ceil(frac * n)))
            return float(sum(valid_sliding[:k]) / k)

        bottom_01 = _bottom_frac_mean(BOTTOM_FRACS[0])
        bottom_05 = _bottom_frac_mean(BOTTOM_FRACS[1])

        return {
            "mean_confidence": mean_conf,
            "tail_2048_mean_conf": tail_mean,
            "min_sliding_2048_mean_conf": min_sliding,
            "bottom_0.1_sliding_2048_mean_conf": bottom_01,
            "bottom_0.5_sliding_2048_mean_conf": bottom_05,
        }

    # ---------------------------------------------------------------------
    # 3) FlatLogprobs: vectorized confidence computation (prefix sums).
    #    For topk=20 and K_FOR_CONF=20, per-segment mean is sufficient.
    # ---------------------------------------------------------------------
    def _flatlogprobs_confidence_fast(top_logprobs, n_pos: int, k_for_conf: int):
        """
        Return confs as:
          - numpy array float32 with NaN for invalid
          - or python list with None (if numpy unavailable)
        """

        fl = top_logprobs
        starts = fl.start_indices
        ends = fl.end_indices
        lps = fl.logprobs

        n_pos = min(n_pos, len(starts), len(ends))

        # numpy fast path
        starts_np = np.asarray(starts[:n_pos], dtype=np.int64)
        ends_np = np.asarray(ends[:n_pos], dtype=np.int64)
        lps_np = np.asarray(lps, dtype=np.float32)

        seg_len = (ends_np - starts_np).astype(np.int64)
        valid_seg = seg_len > 0

        # prefix sum
        prefix_lp = np.empty(lps_np.shape[0] + 1, dtype=np.float64)
        prefix_lp[0] = 0.0
        np.cumsum(lps_np.astype(np.float64), out=prefix_lp[1:])

        seg_sum = prefix_lp[ends_np] - prefix_lp[starts_np]
        confs_np = np.full(n_pos, np.nan, dtype=np.float32)

        # Future-safe when topk != k_for_conf:
        # - seg_len <= k: use direct segment mean
        # - seg_len > k: previous behavior selected top-k by rank; with topk=20
        #   and k=20 we use direct segment mean.
        confs_np[valid_seg] = (-seg_sum[valid_seg] / seg_len[valid_seg]).astype(np.float32)

        return None, confs_np

    def _listlogprobs_confidence(
        self,
        token_ids,
        top_logprobs,
        tokenizer,
        num_output_top_logprobs,
        return_as_token_id,
        k_for_conf: int,
    ):
        """Compute per-token confidence for list-based logprobs format."""
        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )

        confs = []
        n_pos = min(len(token_ids), len(top_logprobs))
        for i in range(n_pos):
            token_id = token_ids[i]
            step_top = top_logprobs[i]

            if step_top is None or not hasattr(step_top, "get") or step_top.get(token_id) is None:
                confs.append(None)
                continue

            top_list = self._get_top_logprobs(
                step_top,
                num_output_top_logprobs,
                tokenizer,
                should_return_as_token_id,
            )

            conf = None
            if top_list:
                vals = [lp.logprob for lp in top_list[:k_for_conf] if lp.logprob is not None]
                if vals:
                    conf = -sum(vals) / len(vals)
            confs.append(conf)

        if len(token_ids) > n_pos:
            confs.extend([None] * (len(token_ids) - n_pos))

        confs_np = np.asarray(
            [np.nan if c is None else float(c) for c in confs],
            dtype=np.float64,
        )
        return confs, confs_np

    # ---------------------------------------------------------------------
    # 4) Patch assembly function
    # ---------------------------------------------------------------------
    def _patched_create_chat_logprobs(
        self,
        token_ids,
        top_logprobs,
        tokenizer,
        num_output_top_logprobs=None,
        return_as_token_id=None,
        *args,
        **kwargs,
    ):
        # Empty mode: skip response payload/serialization overhead.
        if MODE == "empty":
            return protocol.ChatCompletionLogProbs(content=[])

        # Stats mode: return only summary statistics, no per-token payload.
        if MODE == "stats":
            import vllm.logprobs as _lp

            n_pos = len(token_ids)

            if isinstance(top_logprobs, _lp.FlatLogprobs):
                # ---- (1a) FlatLogprobs confidence: vectorized ----
                confs_list, confs_np = _flatlogprobs_confidence_fast(
                    top_logprobs,
                    n_pos=n_pos,
                    k_for_conf=K_FOR_CONF,
                )
            elif isinstance(top_logprobs, (list, tuple)):
                # ---- (1b) List-based logprobs confidence ----
                confs_list, confs_np = _listlogprobs_confidence(
                    self,
                    token_ids,
                    top_logprobs,
                    tokenizer,
                    num_output_top_logprobs,
                    return_as_token_id,
                    K_FOR_CONF,
                )
            else:
                raise ValueError(f"Unknown logprobs type: {type(top_logprobs)}")

            # ---- (2)(3) summarize: no sort + vectorized sliding ----
            if confs_np is not None:
                try:
                    summary = _summarize_confs_fast_numpy(confs_np)
                except Exception:
                    # Rare edge-case fallback
                    confs = [
                        None if (c is None or (isinstance(c, float) and math.isnan(c))) else float(c)
                        for c in (confs_list if confs_list is not None else [])
                    ]
                    summary = _summarize_confs_fallback_python(confs)
            else:
                summary = _summarize_confs_fallback_python(confs_list)

            return protocol.ChatCompletionLogProbs(content=[], confidence_summary=summary)

        # --- Original per_token / summary logic (kept mostly unchanged) ---
        logprobs_content = []
        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )

        for i, token_id in enumerate(token_ids):
            step_top = top_logprobs[i]

            if step_top is None or step_top.get(token_id) is None:
                tok = f"token_id:{token_id}" if should_return_as_token_id else tokenizer.decode(token_id)
                logprobs_content.append(
                    protocol.ChatCompletionLogProbsContent(
                        token=tok,
                        bytes=list(tok.encode("utf-8", errors="replace")),
                        confidence=None,
                    )
                )
                continue

            step_token = step_top[token_id]
            step_decoded = step_token.decoded_token

            top_list = self._get_top_logprobs(
                step_top,
                num_output_top_logprobs,
                tokenizer,
                should_return_as_token_id,
            )

            conf = None
            if top_list:
                vals = [lp.logprob for lp in top_list[:K_FOR_CONF] if lp.logprob is not None]
                if vals:
                    conf = -sum(vals) / len(vals)

            if MODE == "summary":
                top_list = []

            logprobs_content.append(
                protocol.ChatCompletionLogProbsContent(
                    token=self._get_decoded_token(
                        step_token,
                        token_id,
                        tokenizer,
                        should_return_as_token_id,
                    ),
                    logprob=max(
                        step_token.logprob if step_token.logprob is not None else -9999.0,
                        -9999.0,
                    ),
                    bytes=None if step_decoded is None else list(step_decoded.encode("utf-8", errors="replace")),
                    top_logprobs=top_list,
                    confidence=conf,
                )
            )

        return protocol.ChatCompletionLogProbs(content=logprobs_content)

    serving_chat.OpenAIServingChat._create_chat_logprobs = _patched_create_chat_logprobs
